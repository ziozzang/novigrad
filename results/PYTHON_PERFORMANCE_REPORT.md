# Native Python and Metal performance experiments

Measured on Apple M2 Ultra, macOS 26.5 arm64, MLX 0.32.2. The Python extension directly calls the existing Rust learning engine through PyO3. The optional Metal backend uses MLX Python and custom Metal kernels; it is not an MLX-C integration.

## Public API comparison

The optimized Fashion checkpoint was evaluated on 1,024 feature rows: maximum absolute probability error **2.3842e-7**, **zero argmax disagreements** versus Rust. This is sampled numerical validation, not proof for all inputs or devices. Additional tests cover duplicate edges, inhibitory signs, top-k ties, output group remapping and input scaling.

Median warm latency in milliseconds, 50 repetitions with four rotating input batches:

| Rows | Native Rust via Python | MLX Metal via Python | CPU/Metal ratio |
|---:|---:|---:|---:|
| 1 | 0.111 | 1.235 | 0.09× |
| 8 | 0.847 | 1.282 | 0.66× |
| 32 | 3.373 | 1.455 | 2.32× |
| 128 | 13.474 | 2.191 | 6.15× |
| 256 | 27.527 | 3.192 | 8.62× |

Both paths receive prepared Python lists and return Python lists. Timings include boundary copies, GPU upload, synchronized computation and readback, but exclude feature encoding/model loading. First Metal inference took 142 ms; shader caches may already be warm. Use CPU for low-latency individual observations and benchmark Metal for independent batches, including multiple simulation environments. Batching sequential observations can introduce waiting and alter online learning; throughput is not a frame deadline guarantee. No energy or power measurements were taken.

CPU mean-gradient training remains available through the native binding: median batch-update latency was 0.237 / 5.322 / 21.121 ms for 1 / 32 / 128 rows. These 20-repeat measurements update a throwaway model, not reset weights before every call. They establish execution cost, not learning quality. GPU training is not implemented.

A separate five-action driving-model transport benchmark measured mean native/HTTP latency of 0.103/0.423 ms for one observation, 3.441/7.356 ms for 32, and 13.664/28.967 ms for 128 (100 repetitions). Native and HTTP mean updates produced exactly equal weights. Model and aggregation statistics differ from the table above, so do not combine their ratios.

## Mechanisms and rejected alternatives

1. The native extension removes HTTP JSON/transport from Python simulations, releases the GIL during Rust computation and supports whole minibatches. Inputs/results still copy; this is not a zero-copy API.
2. A dense MLX formulation was fast but failed numerical parity: maximum probability differences reached about 0.0122 on the steering probe and 0.00895 on another feature probe. Floating-point sum reordering can change near-tied hidden winners. It is retained only as an experimental comparator.
3. The adopted sparse Metal kernels group edges by target while preserving their original accumulation order, disable floating-point contraction, use safe math and stable top-k ordering. Production allocates no dense connectivity matrices. MLX compilation reduces dispatch overhead and batches amortize the GPU latency.
4. Threshold/partition selection with stable tie handling passed sampled parity but was slower than stable sorting: NumPy host-to-host probe latencies at batch 1/32/128 were approximately 1.304/1.388/1.569 ms versus 1.286/1.361/1.474 ms. Stable sorting remains the default. These probes have different marshalling/model scope from the public API table.

MLX-C exposes the MLX runtime to C callers, but changing the bridge alone does not eliminate GPU launch cost or fix sparse numerical semantics. Direct C integration, fused GPU learning, device-resident simulation buffers and automatic dispatch are future experiments, not delivered capabilities.

## Reproduce and inspect

```sh
.venv/bin/python -m pip install -e '.[mlx]'
.venv/bin/python -m unittest discover -s tests/python -p 'test_*.py' -v
.venv/bin/python examples/python/basic.py
.venv/bin/python scripts/benchmark_backends.py examples/vision/models/fashion-optimized/model.safetensors data/vision/fashion64/features.safetensors --input-key test_inputs --out results/python-binding/fashion-backends.json
.venv/bin/python scripts/benchmark_python_binding.py --help
.venv/bin/python scripts/mlx_engine_probe.py --help
```

Obtain/generate the model and features using the vision examples first; large model/data files are not committed. Raw [public API measurements](python-binding/fashion-backends.json) record checkpoint/feature SHA-256 hashes, device, setup costs, p95 values and limitations. [Transport measurements](python-binding/benchmark.json) and `novi-mlx-*.json` in the same directory retain the other experiments. Probe files may reference local temporary paths and are historical observations.

The basic binding test starts from a symmetric new circuit (0.5 probabilities), trains 30 teacher batches and reaches approximately 0.9912 probability on each correct class; save/reload and resumed updates match exactly. This verifies the learning bridge, not generic intelligence or a biological model claim.

Primary implementation references: [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html), [MLX-C](https://github.com/ml-explore/mlx-c).

[한국어](PYTHON_PERFORMANCE_REPORT.ko.md)
