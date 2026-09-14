#!/usr/bin/env python3
"""Experimental Python MLX Metal inference; not an MLX-C integration or trainer.

Uses stable descending argsort for Engine's lower-index top-k ties. Dense
reductions can differ numerically from Rust's sequential sparse reductions;
parity is measured, never assumed. No labels or accuracy are consumed.

The ordered sparse alternative groups edges by target with stable sorting and
accumulates sequentially in a custom Metal kernel with FP contraction disabled.
It retains the checkpoint edge order, signs, gains, and deterministic top-k ties.

Official API references:
https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html
https://ml-explore.github.io/mlx/build/html/usage/compile.html
https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
https://github.com/ml-explore/mlx-c

Requires the local novigrad Python binding for the in-process Rust comparison.
First-call time includes compilation, but persistent Metal shader caches may
already be warm. Timed MLX inference excludes host input/output conversion.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from novigrad import Engine


from novigrad._mlx_ops import build


def measure(fn, inputs, repeats):
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        result = fn(inputs[i % len(inputs)])
        mx.eval(result)
        times.append(time.perf_counter() - start)
    return {'median_ms': float(np.median(times) * 1000), 'p95_ms': float(np.percentile(times, 95) * 1000)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('features', type=Path)
    p.add_argument('--input-key', default='inputs')
    p.add_argument('--out', type=Path, default=Path('/tmp/novi-mlx-probe.json'))
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--selection', choices=['argsort', 'partition'], default='argsort')
    args = p.parse_args()
    if not mx.metal.is_available():
        raise RuntimeError('this probe requires actual Metal GPU availability')
    mx.set_default_device(mx.gpu)
    forward, ids, setup = build(args.model, args.selection)
    with safe_open(str(args.features), framework='numpy') as f:
        np.testing.assert_array_equal(f.get_tensor('input_ids'), ids)
        rows = f.get_tensor(args.input_key)
    if rows.ndim != 2 or rows.shape[1] != len(ids) or not np.isfinite(rows).all() or len(rows) == 0:
        raise ValueError('invalid feature matrix')
    rows = np.asarray(rows, np.float32)
    # Include zero (all ties), equal-rate, and input normalization branch probes.
    parity_rows = np.concatenate([rows[:256], np.zeros((1, len(ids)), np.float32),
        np.ones((1, len(ids)), np.float32), np.full((1, len(ids)), 32, np.float32)])
    engine = Engine.load(args.model)
    rust = np.asarray(engine.infer_batch(parity_rows.tolist()))
    compiled = mx.compile(forward)
    output = {}
    for name, fn in [('eager', forward), ('compiled', compiled), ('ordered_sparse_compiled', mx.compile(lambda x: forward(x, sparse=True)))]:
        start = time.perf_counter()
        actual = fn(mx.array(parity_rows))
        mx.eval(actual)
        first_call = time.perf_counter() - start
        actual = np.asarray(actual)
        output[name] = {'first_call_seconds': first_call,
            'max_absolute_probability_error': float(np.max(np.abs(actual - rust))),
            'action_disagreements': int(np.sum(actual.argmax(1) != rust.argmax(1))),
            'probability_allclose_atol_1e_5_rtol_1e_5': bool(np.allclose(actual, rust, atol=1e-5, rtol=1e-5)),
            'batches': {}}
        for batch in (1, 32, 128):
            # Four distinct input arrays and a fresh function call every iteration;
            # never time re-evaluating an already-materialized output array.
            arrays = [mx.array(rows[(np.arange(batch) + offset) % len(rows)]) for offset in range(4)]
            mx.eval(*arrays)
            start = time.perf_counter()
            for x in arrays:
                mx.eval(fn(x))
            warmup = time.perf_counter() - start
            output[name]['batches'][str(batch)] = {'warmup_seconds': warmup, **measure(fn, arrays, args.repeats)}
            host_arrays = [rows[(np.arange(batch) + offset) % len(rows)].copy() for offset in range(4)]
            transfer_times = []
            for i in range(args.repeats):
                start = time.perf_counter()
                y = fn(mx.array(host_arrays[i % 4]))
                mx.eval(y)
                np.array(y, copy=True)
                transfer_times.append(time.perf_counter() - start)
            output[name]['batches'][str(batch)]['host_to_host_median_ms'] = float(np.median(transfer_times) * 1000)
            output[name]['batches'][str(batch)]['host_to_host_p95_ms'] = float(np.percentile(transfer_times, 95) * 1000)
    native = {}
    for batch in (1, 32, 128):
        inputs = [rows[(np.arange(batch) + offset) % len(rows)].tolist() for offset in range(4)]
        for x in inputs:
            engine.infer_batch(x)
        times = []
        for i in range(args.repeats):
            start = time.perf_counter()
            engine.infer_batch(inputs[i % 4])
            times.append(time.perf_counter() - start)
        native[str(batch)] = {'median_ms': float(np.median(times) * 1000), 'p95_ms': float(np.percentile(times, 95) * 1000)}
    import importlib.metadata
    report = {'mlx_version': importlib.metadata.version('mlx'), 'device': str(mx.default_device()),
        'device_info': mx.device_info(), 'model': str(args.model), 'features': str(args.features),
        'selection': args.selection, 'parity_rows': len(parity_rows), 'repeats': args.repeats, 'setup': setup, 'mlx': output,
        'native_pyo3': native, 'limitations': ['Python MLX prototype, not MLX-C; inference only',
        'Dense reductions and duplicate-edge consolidation can change sparse sequential rounding',
        'Device-resident timing excludes transfers; host_to_host includes NumPy upload and output copy; native includes Python list marshalling',
        'First-call timing may reuse persistent Metal shader cache',
        'No labels or driving performance assessed']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
