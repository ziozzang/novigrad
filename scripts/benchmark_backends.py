#!/usr/bin/env python3
"""Public native/Metal API parity and host-to-host timings for identical inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from safetensors import safe_open
from novigrad import Engine
from novigrad.mlx import MlxEngine


def timed(fn, arrays, repeats):
    for array in arrays:
        fn(array)
    times = []
    for i in range(repeats):
        started = time.perf_counter()
        fn(arrays[i % len(arrays)])
        times.append((time.perf_counter() - started) * 1000)
    return {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
            'mean_ms': float(np.mean(times)), 'repeats': repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('features', type=Path)
    parser.add_argument('--input-key', default='inputs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    with safe_open(str(args.features), framework='numpy') as data:
        ids = data.get_tensor('input_ids')
        rows = np.asarray(data.get_tensor(args.input_key), dtype=np.float32)
    start = time.perf_counter()
    cpu = Engine.load(args.model)
    cpu_load = time.perf_counter() - start
    start = time.perf_counter()
    gpu = MlxEngine.load(args.model)
    gpu_load = time.perf_counter() - start
    np.testing.assert_array_equal(ids, cpu.input_ids)
    if len(rows) == 0:
        raise ValueError('empty feature input')
    checked = rows[:1024].tolist()
    expected = np.asarray(cpu.infer_batch(checked))
    start = time.perf_counter()
    actual = np.asarray(gpu.infer_batch(checked))
    first_call = time.perf_counter() - start
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
    measurements = {}
    for batch in (1, 8, 32, 128, 256):
        inputs = [rows[(np.arange(batch)+offset) % len(rows)].tolist() for offset in range(4)]
        measurements[batch] = {'native': timed(cpu.infer_batch, inputs, 50),
                               'metal': timed(gpu.infer_batch, inputs, 50)}
    # CPU training cost is measured separately; Metal backend is inference-only.
    trainer = Engine.load(args.model)
    training = {}
    for batch in (1, 32, 128):
        inputs = rows[np.arange(batch) % len(rows)].tolist()
        actions = [i % cpu.config['actions'] for i in range(batch)]
        rewards = [.1 if i % 2 else -.1 for i in range(batch)]
        training[batch] = timed(lambda x: trainer.learn(x, actions, rewards), [inputs], 20)
    import mlx.core as mx
    result = {'platform': platform.platform(), 'mlx_version': importlib.metadata.version('mlx'),
              'device': mx.device_info(), 'model': str(args.model),
              'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'features_sha256': hashlib.sha256(args.features.read_bytes()).hexdigest(),
              'parity_rows': len(checked), 'max_probability_error': float(np.max(np.abs(actual-expected))),
              'action_disagreements': 0, 'cpu_load_seconds': cpu_load, 'metal_load_seconds': gpu_load,
              'first_metal_call_seconds': first_call, 'metal_setup': gpu.setup,
              'inference': measurements, 'native_training': training,
              'scope': 'Same prepared Python-list inputs to returned Python-list probabilities, including marshalling, GPU input construction, synchronized compute and readback; excludes initial loading and feature encoding.',
              'limitations': ['Metal inference only, not GPU training or MLX-C integration',
                              'Warm shader caches possible; four input arrays alternate to avoid timing an already materialized output',
                              'Native training timings update a throwaway model each call; weights are not reset per sample',
                              'Parity is checked on these inputs, not a proof for every checkpoint/device']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'parity_error': result['max_probability_error'], 'inference': measurements}, indent=2))


if __name__ == '__main__':
    main()
