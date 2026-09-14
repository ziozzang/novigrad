#!/usr/bin/env python3
"""Paced camera-file replay through a persistent Novigrad HTTP model.

Measures JPEG read/decode, image adapter, JSON/HTTP, and response decoding.
This replays observations; predicted actions do not alter the environment.
"""
import argparse
import csv
import http.client
import json
import math
from pathlib import Path
import time
import urllib.parse

import numpy as np
from PIL import Image
from image_adapter import ImageRateAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('frames', type=Path, help='directory of JPEG frames')
    parser.add_argument('--api-url', default='http://127.0.0.1:8080')
    parser.add_argument('--model', default='steering')
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--ticks', type=int, default=150)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 0 < args.fps <= 1000 or args.ticks < 1 or args.warmup < 0:
        parser.error('fps must be finite in (0,1000], ticks positive, warmup nonnegative')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('use an empty output directory')
    frames = sorted(args.frames.glob('*.jpg'))
    if not frames:
        parser.error('no JPEG frames found')
    bundle = json.loads(args.bundle.read_text())
    if bundle.get('class_labels') != ['left', 'straight', 'right'] or bundle.get('sequence_length') != 1:
        parser.error('expected whole-image three-action steering bundle')
    adapter = ImageRateAdapter.load(args.bundle.parent / bundle['adapter'])
    means = np.asarray(bundle['train_class_steer_mean'], dtype=np.float64)
    if means.shape != (3,) or not np.isfinite(means).all() or np.any(np.abs(means) > 1):
        parser.error('invalid steering decoder')
    url = urllib.parse.urlsplit(args.api_url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.path not in ('', '/') or url.query or url.fragment or url.username:
        parser.error('api-url must be a local HTTP origin')
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=5)
    endpoint = '/v1/models/' + urllib.parse.quote(args.model, safe='') + '/infer'
    ids = [str(int(value)) for value in adapter.input_ids]

    def step(path):
        with Image.open(path) as source:
            pixels = np.asarray(source.convert('L').resize((28, 28), Image.Resampling.BILINEAR), dtype=np.uint8)[None]
        rates = adapter.transform(pixels)
        body = json.dumps({'inputs': rates.tolist(), 'input_ids': ids})
        connection.request('POST', endpoint, body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f'HTTP inference failed: {response.status}: {payload[:200]!r}')
        result = json.loads(payload)
        probabilities = np.asarray(result['probabilities'], dtype=np.float64)
        if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(), 1, atol=1e-6):
            raise ValueError('invalid API probabilities')
        action = int(result['actions'][0])
        if action != int(probabilities[0].argmax()):
            raise ValueError('API action/probability disagreement')
        return action, float(probabilities[0] @ means), probabilities[0]

    rows = []
    period = 1 / args.fps
    try:
        for index in range(args.warmup):
            step(frames[index % len(frames)])
        started = time.perf_counter()
        scheduled = started
        for index in range(args.ticks):
            remaining = scheduled - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            before = time.perf_counter()
            path = frames[index % len(frames)]
            action, steer, probabilities = step(path)
            after = time.perf_counter()
            next_slot = scheduled + period
            skipped = max(0, math.floor((after - next_slot) / period) + 1) if after >= next_slot else 0
            rows.append({'tick': index, 'frame': path.name, 'action': action, 'normalized_steer': steer,
                         'p_left': float(probabilities[0]), 'p_straight': float(probabilities[1]), 'p_right': float(probabilities[2]),
                         'latency_ms': (after - before) * 1000,
                         'start_lateness_ms': max(0, before - scheduled) * 1000,
                         'deadline_missed': int(after > scheduled + period), 'skipped_slots': skipped})
            scheduled = next_slot + skipped * period
        elapsed = time.perf_counter() - started
    finally:
        connection.close()
    latency = np.array([row['latency_ms'] for row in rows])
    summary = {'mode': 'paced prerecorded JPEG replay, open-loop observations',
               'frames': len(frames), 'ticks': args.ticks, 'warmup_ticks_excluded': args.warmup,
               'requested_fps': args.fps, 'period_ms': period * 1000,
               'elapsed_seconds': elapsed,
               'latency_ms': {'mean': float(latency.mean()), 'p50': float(np.percentile(latency, 50)),
                              'p95': float(np.percentile(latency, 95)), 'p99': float(np.percentile(latency, 99)), 'max': float(latency.max())},
               'compute_over_period_count': int(np.sum(latency > period * 1000)),
               'scheduled_deadline_misses': sum(row['deadline_missed'] for row in rows),
               'skipped_slots': sum(row['skipped_slots'] for row in rows),
               'max_start_lateness_ms': max(row['start_lateness_ms'] for row in rows),
               'latency_scope': 'file read, JPEG decode/resize, adapter, JSON request, local HTTP inference, response validation and action decode; excludes initial loading, warmup, pacing sleep and final report writes',
               'limitations': 'OS file cache may be warm; no live camera capture, network camera, actuation, closed-loop dynamics or hard real-time guarantee'}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'ticks.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
