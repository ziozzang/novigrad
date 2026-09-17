# Novigrad — Software-defined Bionic NPU

Novigrad is a Mac-focused CPU research engine for connectome-constrained learning, fixed neuronal input/output ports, and reusable task adapters. It supports supervised and scalar-reward learning, local HTTP serving, soft real-time periodic execution, and delayed reward modulation. It uses published FlyWire connectivity as topology, a compact rate model for learning, and Safetensors checkpoints. The Rust crate and executable are named `novi`. [한국어 README](README.ko.md).

The current learning circuit has **319 ALPN inputs, 5,177 Kenyon cells (KCs), and 96 MBON outputs**. The 27,848 ALPN→KC connections are fixed; the weights of 62,261 existing KC→MBON connections are plastic. The engine does not create new connections. These are circuit counts, not a claim that the whole fly brain is trained.

## Build and reproduce

On an Apple Silicon Mac with Rust installed:

```sh
git clone https://github.com/ziozzang/novigrad.git
cd novigrad
cargo build --release --bins
cargo test --release
python3 -m venv .venv
.venv/bin/pip install -r requirements-data.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/prepare_training.py
cargo run --release --bin train_connectome -- \
  --task all --seed-start 101 --seeds 8 \
  --episodes 2000 --reversal-episodes 2000 --eval 1024 \
  --lr 0.02 --gain 12 --out results/v1-confirmation
```

The checked-in confirmation used 8 held-out seeds (101–108), three tasks, and three reward conditions: task reward, zero reward, and shuffled ±1 reward. It evaluated independent noisy inputs after training and after reversing the task rule. Across the 8 seeds, trained greedy accuracy was 100% on binary, four-way, and XOR tasks; mean probability assigned to the correct answer was 99.55%, 98.29%, and 89.98%, respectively. Zero and shuffled rewards stayed near chance. The synthetic task inputs and engineered scalar rewards do not establish real sensory understanding. See the [v1 report](results/V1_REPORT.md) for controls, reversal, per-run artifacts, and caveats.

## Generic engine

The Rust library accepts two edge tables, arbitrary external input IDs and output IDs, and a configurable number of actions. A small end-to-end example trains, saves, loads, evaluates, and infers:

```sh
cargo run --release --bin novi_engine -- train \
  examples/generic/input_edges.tsv examples/generic/plastic_edges.tsv \
  examples/generic/train.tsv results/generic.safetensors 2 100 7
cargo run --release --bin novi_engine -- eval \
  results/generic.safetensors examples/generic/eval.tsv
cargo run --release --bin novi_engine -- infer \
  results/generic.safetensors examples/generic/infer.tsv
```

Edge rows are `pre_id<TAB>post_id<TAB>syn_count<TAB>sign` without a header. Training/evaluation rows are `label<TAB>root_id:value...`; inference rows omit the label. The external caller supplies encoding, action choice, and reward. See [architecture](docs/architecture.md) for the learning rule and [model format](docs/model-format.md) for the 14-tensor Safetensors schema (metadata version 4; version 3 loads with unit readout gains).

## Image examples

The image pipeline fits an **unsupervised training-only adapter** (HOG, pooled pixels and PCA) that maps images to 319 input ports. The same circuit learns clothing classes and locally generated fixed-cell CAPTCHA digits. These image experiments use supervised labels; reward learning is evaluated separately.

After eight controlled configurations, the selected model disables per-output L1 normalization, trains for up to 75 epochs and retains 20% of hidden neurons. Individual synapse signs and weight bounds remain enforced. On new, previously unused evaluation data:

| Metric | v0.1.0 model on the same data | Improved mean, 3 seeds |
|---|---:|---:|
| Clothing classification, 8,000 images | 71.20% | **85.04%** |
| CAPTCHA characters, 4,000 cells | 83.50% | **99.49%** |
| Exact four-digit strings, 1,000 images | 49.20% | **98.00%** |

See the [full experiment report](results/IMPROVEMENT_REPORT.md), [mechanism documentation](docs/optimization.md), and [image instructions](examples/vision/README.md). The report preserves unsuccessful PCA/batch experiments, controls, source hashes and raw predictions. CAPTCHA uses known cell positions and shared fonts; these results do not establish arbitrary live CAPTCHA solving or general scene detection.

A small [Hugging Face steering imitation example](examples/steering/README.md) extends the same image-to-port interface to simulated driving commands. It tests offline command prediction, with separate driving sessions for evaluation. Accuracy is still weak (60% versus a 58% always-straight baseline). A 30 FPS replay measured 2.77 ms mean end-to-end processing latency and no deadline misses in 300 ticks on the M2 Ultra; this excludes live-camera and actuator latency.

## API, periodic execution, and delayed reward

All interfaces use the same numeric input ports and Safetensors engine:

- [Local HTTP API](docs/api.md): named models, port metadata, batch inference, opt-in learning and checkpoints. Build with `cargo build --release --features api --bins`, then run `novi_api --model NAME=CHECKPOINT`.
- [Soft real-time runtime](docs/runtime.md): `novi_rt CHECKPOINT INPUT.tsv --ticks 1000 --period-us 1000`, with measured deadlines and skipped overdue slots. macOS provides no hard real-time guarantee.
- [Delayed reward modulation](docs/neuromodulation.md): bounded eligibility replay and reward-minus-baseline modulation. The [27-run odor/action report](results/NEUROMODULATION_REPORT.md) includes frozen, unrelated-reward and reversal controls. Its foreleg commands are virtual output meanings, not reconstructed motor neurons.

Adapters own image/sensor encoding; applications own action meanings and reward sources. The core does not embed a task-specific vision model, motor controller, or HTTP dependency.

## Closed-loop driving game

The [HighwayEnv example](examples/driving-game/README.md) learns from game rewards without expert actions. Independent runs start from identical initial weights; episodes reset the simulator, and validation/test never train. On 20 held-out games, learned collisions fell from 15 to 1 and mean return rose from 22.20 to 27.95. The learned behavior is mainly slowing down: an always-slower diagnostic achieved zero collisions and a higher score. See the [full report](results/DRIVING_GAME_REPORT.md) and [game GIF](results/driving-game/driving.gif).

Run `scripts/play_driving_game.py MODEL.safetensors --human` to watch a saved policy in a local game window. Training uses structured kinematic observations and an environment-defined reward; this is not arbitrary game-goal understanding or a raw-camera driver.

## Three-policy voting

The [MAGI-inspired ensemble](examples/magi/README.md) gives three independently trained circuits the same observation and compares majority voting with probability averaging. On30 fresh games, collisions were0 for the validation-selected single member,20 for majority, and3 for probability averaging. Two weaker policies made identical votes; more voters did not mean better judgment. [Full report](results/MAGI_REPORT.md).

## Scope and performance

A separate full-graph LIF runner activates the derived FlyWire graph; it is not the rate-learning circuit. Each circuit executes on a CPU thread; independent API models use separate locks and bounded worker jobs. The local Cargo configuration uses `target-cpu=native`, so release binaries should be built on the target Mac rather than assumed portable. The measured M2 Ultra fixed-input forward plus zero-reward loop was 9,732 episodes/s; it is not actual training throughput or robot-control latency ([measurement](results/V1_REPORT.md)).

This prototype does not implement whole-brain training, actual dopaminergic signaling, biological timing, hardware NPU execution, or proven generalization to unseen XOR combinations. The topology is biological data, while the input encoding, action mapping, reward, and learning rule are engineered. Source rights and transforms are documented in [data/README.md](data/README.md).

## Release

[Download v0.1.1](https://github.com/ziozzang/novigrad/releases/tag/v0.1.1): Apple Silicon binaries, Safetensors model bundles, and SHA-256 checksums. Code is MIT; see [NOTICE](NOTICE.md) for data and third-party attribution.

Author: jioh jung <jung@jioh.net> · License: MIT


Native Python bindings and optional MLX/Metal batch inference are available: [examples](examples/python/README.md), [measured performance and rejected optimizations](results/PYTHON_PERFORMANCE_REPORT.md).


Frozen Gemma semantic bridge: [example and measured learning](examples/gemma_bridge/README.md), [anatomical interface research](examples/gemma_bridge/RESEARCH.md).


FunctionGemma typed API and LoRA routing experiments: [example](examples/function_bridge/README.md), [memory, cue conflict and goal-switch research](examples/function_bridge/CASES_RESEARCH.md).

Further research: [grounded goals, sensor reliability and reward-learning mechanisms](examples/function_bridge/DEEP_RESULTS.md), including negative results and stronger conventional baselines.

Biological mechanism diagnostics: [EmbeddingGemma sparsity, FunctionGemma abstention, context memory, and sequential delayed credit](examples/bio_bridge/README.md), with [primary-source research](examples/bio_bridge/RESEARCH.md).

Signal timing and strength: [continuous versus one-shot input, host memory, reward magnitude, and spaced/repeated exposure](examples/bio_bridge/SIGNAL_RESULTS.md).
