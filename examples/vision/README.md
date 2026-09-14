# Image tasks on the Bionic NPU

[한국어](README.ko.md) · [Measured results](../../results/VISION_REPORT.md)

The same Rust engine learns ten clothing categories and digits in locally generated CAPTCHA images. An interchangeable Python adapter converts pixels into the circuit's input rates. The optimized v0.1.1 models achieve 85.04% mean clothing accuracy, 99.49% digit accuracy, and 98.00% exact four-digit success across three training seeds on newly held-out images. On the same data, the original released seed-1 models score 71.20%, 83.50%, and 49.20%, respectively. See the [mechanism report](../../results/IMPROVEMENT_REPORT.md). These are research examples with useful failures, not production OCR.

![Fashion examples](assets/fashion-samples.png)
![Synthetic CAPTCHA examples](assets/captcha-samples.png)

## Run a released model

From the repository root, install the dependencies and build the Rust binaries:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-vision.txt
cargo build --release --bins
```

Download `novigrad-v0.1.1-models.tar.gz` from the [release](https://github.com/ziozzang/novigrad/releases/tag/v0.1.1), and extract it at the repository root. The archive populates `examples/vision/models/` and `models/connectome/`. Inference needs no raw FlyWire or Fashion-MNIST download; topology and weights are self-contained in the checkpoints.

```sh
tar -xzf novigrad-v0.1.1-models.tar.gz
.venv/bin/python scripts/predict_image.py \
  examples/vision/models/fashion-optimized/bundle.json examples/vision/assets/fashion-example.png
.venv/bin/python scripts/predict_image.py \
  examples/vision/models/captcha-optimized/bundle.json examples/vision/assets/captcha-example.png
```

The first example's true class is `Shirt`, but this model predicts `T-shirt/top`. The second predicts the correct `5420`. Both images are predetermined test sample 0, not selected for success. Output includes every class probability. Fashion takes one image, converts to grayscale and resizes to 28×28. CAPTCHA requires a 112×28 image with four known 28×28 cells. Inputs should have a dark background and bright foreground; use `--invert` for the opposite polarity.

A prebuilt Apple Silicon executable can be selected with `--binary PATH/TO/classify_features`.

## Reproduce the original baseline training

```sh
.venv/bin/python scripts/run_vision_examples.py
.venv/bin/python scripts/report_vision.py
.venv/bin/python scripts/verify_safetensors.py \
  examples/vision/models/fashion-optimized/model.safetensors \
  examples/vision/models/captcha-optimized/model.safetensors \
  --report results/vision-safetensors-verification.json
```

The runner downloads and verifies source data, prepares the FlyWire subcircuit, creates deterministic image splits, fits adapters using training images only, and trains seeds 1–3 plus a shuffled-label control. It selects checkpoints on validation, then evaluates test data. `--skip-prepare` reuses local inputs; `--task fashion` or `--task captcha` runs one task. The first specified seed is packaged, never the seed with the best test score.

Dataset generation defaults to macOS `SFNSMono.ttf` and `Geneva.ttf`. Font binaries are not distributed. `scripts/prepare_vision.py --help` lists font options for other machines; changing fonts creates a different experiment. Numerical results can also vary with BLAS and CPU implementations. See [dataset hashes and font provenance](../../results/vision-data-manifest.json).

## Absorbing another modality

The adapter uses HOG and pooled pixels, training-only feature scaling and PCA, then maps signed components to nonnegative input rates. It reads no labels during fitting or inference. The engine receives ordinary rates and has no image-specific branch. The external output decoder uses fixed opponent gains; the actual circuit topology and synapse signs are preserved.

To add a task, supply a dataset Safetensors file with `input_ids: U64[D]`, `train_inputs`, `validation_inputs`, `test_inputs: F32[N,D]`, and corresponding `*_labels: I64[N]`. Rates must be finite and input IDs must match the graph's sorted input ports. Invoke `train_classifier` with two edge tables and the action count. `--phase tune` evaluates validation only. `classify_features` loads a checkpoint and accepts label-free `inputs` and `input_ids` tensors. See the [architecture](../../docs/architecture.md) and [model format](../../docs/model-format.md).

Published vision runs use supervised teacher gradients. Reward learning is validated separately by the binary/four-way/XOR experiments; successful bandit-only image training is not claimed here. Fashion-MNIST covers clothing categories, not scene detection. Synthetic CAPTCHA covers known cell positions and the generator's font/noise distribution, not arbitrary websites or unseen segmentation.

## Reproduce the improvement and call HTTP inference

The baseline runner above prepares the original data and bundles. The optimized protocol is documented in [optimization](../../docs/optimization.md). The published selected configuration can be rerun as:

```sh
.venv/bin/python scripts/experiment_vision.py active20 --homeostasis false --epochs 75 --patience 10 --active .2
.venv/bin/python scripts/prepare_vision_holdout.py
.venv/bin/python scripts/confirm_vision_improvement.py results/improvement/selection.json --out results/improvement/reproduction
.venv/bin/python scripts/report_vision_improvement.py --confirmation results/improvement/reproduction
```

The release retains original bundles under `fashion/` and `captcha/` for the same-data baseline comparison. The improved bundles are under `fashion-optimized/` and `captcha-optimized/`. Model selection happens before holdout evaluation, and seed 1 is packaged regardless of test score.

The same image adapter can send label-free port rates to the optional local API:

```sh
cargo build --release --features api --bin novi_api
./target/release/novi_api --model captcha=examples/vision/models/captcha-optimized/model.safetensors
```

In another terminal:

```sh
.venv/bin/python scripts/predict_image.py \
  examples/vision/models/captcha-optimized/bundle.json examples/vision/assets/captcha-example.png \
  --api-url http://127.0.0.1:8080 --model captcha
```

The client sends U64 port IDs as strings and validates returned dimensions and probabilities. It does not send labels. See [API semantics and limits](../../docs/api.md).
