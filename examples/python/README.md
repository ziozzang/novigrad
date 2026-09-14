# Native Python binding

`novigrad.Engine` calls the existing Rust engine in the Python process. It has no HTTP server, per-inference subprocess, pretrained policy, or Python implementation of the learning algorithm. The separate PyO3 extension depends on the core crate; the normal Rust build does not require Python.

## Install on a Mac

From the repository root, with Rust and CPython3.10+ installed:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python examples/python/basic.py
```

This builds a native extension with Maturin. Local Cargo configuration uses `target-cpu=native`, optimized for the build machine. Do not assume a locally built wheel supports older CPUs. These are local installation instructions; this package has not been published to PyPI.

## Load, infer, learn, save

```python
from novigrad import Engine

engine = Engine.load("model.safetensors")
print(engine.input_ids)  # Exact Python integers, including 64-bit neuron IDs.

rates = [0.0] * len(engine.input_ids)
rates[0] = 1.0
probabilities = engine.infer(rates)

# Supply the action actually chosen and its feedback. Positive teacher feedback
# can also train a specified target action, as in basic.py.
action = max(range(len(probabilities)), key=probabilities.__getitem__)
engine.learn([rates], [action], [1.0])
engine.save("updated.safetensors")
```

`infer` is read-only with respect to weights and leaves no pending trial. `infer_batch` amortizes the Python call over multiple rows. `learn(rows, actions, rewards)` validates the entire batch, recomputes its observations at the same starting weights, and applies one mean reward-gradient update. It is compatible with the HTTP `/learn` semantics. If rewards arrive after other weight updates, this recomputation is not an exact on-policy replay of the old policy; keep rollout weights fixed as the game runner does.

Rates must match `input_ids` order and be finite and nonnegative. Rewards must be finite; actions are zero-based indices. Inputs and results are copied across the Python/Rust boundary; this first binding does not claim zero-copy NumPy access. Python lists suffice, and NumPy callers can use `.tolist()`.

New circuits can be created with `Engine.from_edges(input_edges, plastic_edges, ...)`. Defaults follow `PlasticConfig`: two actions, learning rate0.02, gain6, active fraction0.1, homeostasison, positive readout. Pass explicit settings for a task. `set_output_actions` and `set_output_gains` expose the generic decoder. Checkpoints use the existing Safetensors format and load in Rust CLI/API tools.

`save` refuses an existing file unless `overwrite=True`; it writes atomically. Numeric inputs are validated before learning. Rust compute releases the GIL, so distinct engine objects may run on different Python threads. Concurrent operations on the same mutable engine object are not supported; serialize access or use one engine per worker. Loading a checkpoint restores weights/configuration, not a game environment, action RNG, or learner baseline.

The standalone example constructs a tiny symmetric circuit, learns two teacher-labeled patterns, and checks exact save/load predictions. It is a binding demonstration, not a biological or vision benchmark.


## Optional MLX / Metal batch inference

```sh
.venv/bin/python -m pip install -e '.[mlx]'
```

```python
from novigrad.mlx import MlxEngine
metal = MlxEngine.load("model.safetensors")
probabilities = metal.infer_batch([[0.0] * len(metal.input_ids)] * 128)
```

This experimental Apple silicon backend uses MLX and custom sparse Metal kernels. It preserves edge accumulation order and stable top-k tie selection; it does not allocate dense connectivity matrices. It is an inference snapshot: train/save with `Engine`, then reload `MlxEngine` to refresh weights. GPU learning and a direct MLX-C interface are not implemented.

On the measured M2 Ultra workload, CPU is faster for 1–8 observations; Metal is faster for 32–256. Batch 256 took 3.19 ms versus 27.53 ms on CPU, including Python-list conversion and synchronized GPU readback. Choose based on your actual batch size and latency budget; these are warm measurements, not hard real-time guarantees. See the [performance report](../../results/PYTHON_PERFORMANCE_REPORT.md) for parity, failed optimizations and reproduction commands.

The [driving example](driving.py) reuses the virtual driving environment through the native binding. Install `requirements-simulation.txt` and provide a trained checkpoint; see `python examples/python/driving.py --help`.

[한국어](README.ko.md)
