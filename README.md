# Novigrad — Software-defined Bionic NPU

Novigrad is a CPU research prototype for running a connectome-constrained neural circuit and learning from scalar reward. It uses published FlyWire connectivity as topology, a compact rate model for learning, and Safetensors checkpoints. The Rust crate and executable remain named `nobi`. [한국어 README](README.ko.md).

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
cargo run --release --bin nobi_engine -- train \
  examples/generic/input_edges.tsv examples/generic/plastic_edges.tsv \
  examples/generic/train.tsv results/generic.safetensors 2 100 7
cargo run --release --bin nobi_engine -- eval \
  results/generic.safetensors examples/generic/eval.tsv
cargo run --release --bin nobi_engine -- infer \
  results/generic.safetensors examples/generic/infer.tsv
```

Edge rows are `pre_id<TAB>post_id<TAB>syn_count<TAB>sign` without a header. Training/evaluation rows are `label<TAB>root_id:value...`; inference rows omit the label. The external caller supplies encoding, action choice, and reward. See [architecture](docs/architecture.md) for the learning rule and [model format](docs/model-format.md) for the 14-tensor Safetensors schema (metadata version 4; version 3 loads with unit readout gains).

## Image examples

The image pipeline fits an **unsupervised training-split adapter** (HOG and pooled pixels followed by PCA) that maps images to the circuit's 319 input ports. It then offers supervised teacher-gradient and sampled-reward modes. The published image experiments use supervised training and a shuffled-label control. Fashion-MNIST uses 10,000/2,000/2,000 train/validation/test images; a separate locally generated four-digit CAPTCHA task uses 3,000/500/500 full images with four output cells. **Held-out test means across three seeds: 72.62% clothing accuracy, 83.32% CAPTCHA character accuracy, and 48.67% exact four-digit success.** Frozen and shuffled-label controls remain near 10% character/class accuracy. These tasks cover clothing images and synthetic fixed-cell digits; they do not demonstrate arbitrary scene detection or live CAPTCHA solving. See [vision instructions](examples/vision/README.md), [vision report](results/VISION_REPORT.md), and [data provenance](data/README.md). Install their Python dependencies from `requirements-vision.txt` when running them.

## Scope and performance

A separate full-graph LIF runner activates the derived FlyWire graph; it is not the rate-learning circuit. The project currently runs on one CPU thread. The local Cargo configuration uses `target-cpu=native`, so release binaries should be built on the target Mac rather than assumed portable. The measured M2 Ultra fixed-input forward plus zero-reward loop was 9,732 episodes/s; it is not actual training throughput or robot-control latency ([measurement](results/V1_REPORT.md)).

This prototype does not implement whole-brain training, actual dopaminergic signaling, biological timing, hardware NPU execution, or proven generalization to unseen XOR combinations. The topology is biological data, while the input encoding, action mapping, reward, and learning rule are engineered. Source rights and transforms are documented in [data/README.md](data/README.md).

## Release

[Download v0.1.0](https://github.com/ziozzang/novigrad/releases/tag/v0.1.0): Apple Silicon binaries, Safetensors model bundles, and SHA-256 checksums. Code is MIT; see [NOTICE](NOTICE.md) for data and third-party attribution.
