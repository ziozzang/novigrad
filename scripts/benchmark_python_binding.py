#!/usr/bin/env python3
"""Measure persistent native versus HTTP Rust inference and validate mean updates."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
from novigrad import Engine
from train_driving_game import server


def benchmark(function, repeats):
    for _ in range(10):
        function()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append((time.perf_counter() - started) * 1000)
    return {'mean_ms': float(np.mean(times)), 'p95_ms': float(np.percentile(times, 95)),
            'max_ms': float(np.max(times)), 'repeats': repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('results/driving-game/model.safetensors'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    engine = Engine.load(args.model)
    rng = np.random.default_rng(17)
    rows = rng.uniform(0, 1, (128, len(engine.input_ids))).astype(np.float32).tolist()
    result = {'platform': platform.platform(), 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'scope': 'Python list input through validated native or persistent local HTTP to returned probabilities; excludes model loading and sensor adapter',
              'native': {}, 'http': {}}
    with tempfile.TemporaryDirectory(prefix='novi-binding-benchmark-') as temp:
        with server(args.model.resolve(), Path(temp)) as client:
            for batch in (1, 32, 128):
                subset = rows[:batch]
                expected = np.asarray(engine.infer_batch(subset))
                def http_call():
                    return client.request('POST', '/v1/models/policy/infer', {'inputs': subset, 'input_ids': client.ids})['probabilities']
                actual = np.asarray(http_call())
                np.testing.assert_allclose(expected, actual, atol=1e-7, rtol=0)
                result['native'][batch] = benchmark(lambda: engine.infer_batch(subset), 100)
                result['http'][batch] = benchmark(http_call, 100)
            # The same sampled feedback must produce exactly the same stored weights.
            actions = [i % engine.config['actions'] for i in range(32)]
            rewards = [.3 if i % 2 else -.2 for i in range(32)]
            samples = [{'inputs': row, 'action': action, 'reward': reward}
                       for row, action, reward in zip(rows[:32], actions, rewards)]
            engine.learn(rows[:32], actions, rewards)
            client.learn(samples)
            client.request('POST', '/v1/models/policy/checkpoint', {})
            other = Engine.load(Path(temp)/'policy.safetensors')
            assert engine.weights == other.weights
            result['native_http_update_weights_exact'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
