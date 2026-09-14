# Offline steering imitation with Hugging Face data

This example asks whether Novigrad can learn another sensor-to-action mapping: a front camera image to a left, neutral, or right steering command. It reuses the same FlyWire-derived sparse core, image rate adapter, Safetensors checkpoints, and inference interface used by the image examples.

The candidate dataset is [URJC DeepRacer CARLA expert racing](https://huggingface.co/datasets/urjc-deepracer/carla-expert-racing), published with Apache-2.0 metadata. It contains simulated camera frames and expert steering commands. Dataset terms apply separately from Novigrad's MIT source license.

Measured pilot: **60% test accuracy versus 58% always-straight**, with weak left/right recall. See the [full report](../../results/STEERING_REPORT.md). This example demonstrates integration and numerical learning; it is not yet a useful driving policy. The final sample is 100/50/50 images from disjoint official splits, reduced after viewer rate limiting.

## What this experiment can establish

A held-out session test can measure offline imitation of recorded commands. It cannot establish collision avoidance, lane keeping under feedback, recovery after a mistake, or real vehicle safety. Those require a closed-loop environment where predicted actions change future observations.

The initial task uses only camera pixels. Expert steering supplies the training target; maneuver annotations, segmentation masks, session IDs, and timestamps never enter the model. Speed is excluded because the source contains substantial outliers. The adapter is fitted only to training images. Entire source experiment IDs stay in their original, disjoint splits. Frame thinning reduces redundant adjacent observations but does not make remaining frames statistically independent.

Three command bins use fixed boundaries at -0.2 and +0.2 in the source's normalized steering units. Probability-weighted training-set mean steering within each bin provides a coarse continuous decoder. Its MAE is compared with a constant training-median predictor. This is a binned classifier, not a continuous dynamics model.

## Next stages

After the offline baseline, a simulator adapter can feed camera rates into `Runtime`, decode an action, step the environment, and record progress, collisions, lane departures, and latency. Expert imitation can initialize the policy; delayed reward can then test whether progress and penalties improve it. Such rewards describe the engineered task and are not evidence of biologically faithful dopamine signaling.

## Run on a Mac

From the repository root, after preparing the FlyWire circuit as in the main README:

```sh
.venv/bin/python scripts/prepare_steering.py
cargo build --release --bin train_classifier --bin classify_features
.venv/bin/python scripts/experiment_steering.py \
  --data data/steering/carla_racing.npz --out results/steering/reproduction
```

Use a new output directory for each run. The runner compares two predefined homeostasis settings on validation data, saves the selection, trains a shuffled-label control, then evaluates the selected model and control once on the test split. The saved bundle contains a model and image adapter in Safetensors plus JSON action metadata.

To predict a camera JPEG:

```sh
.venv/bin/python scripts/predict_steering.py \
  results/steering/reproduction/bundle/bundle.json camera.jpg
```

The same bundle can use the existing optional HTTP server:

```sh
cargo build --release --features api --bin novi_api
./target/release/novi_api --model steering=results/steering/reproduction/bundle/model.safetensors
# In a second terminal:
.venv/bin/python scripts/predict_steering.py \
  results/steering/reproduction/bundle/bundle.json camera.jpg \
  --api-url http://127.0.0.1:8080 --model steering
```

The returned `prediction` is the most likely command class. `normalized_steer` is a probability-weighted average of training class means; it can be near zero when left and right are equally likely, so callers should retain the probability vector. No vehicle actuator is connected by this example.

## Paced replay at 30 FPS

Keep `novi_api` running with the steering model, then replay the downloaded test JPEGs:

```sh
.venv/bin/python scripts/replay_steering.py \
  results/steering/reproduction/bundle/bundle.json data/steering/source/test \
  --api-url http://127.0.0.1:8080 --model steering \
  --fps 30 --ticks 300 --out results/steering/replay-reproduction
```

The adapter and HTTP connection stay loaded across frames. Each tick reads a JPEG, converts it to rates, invokes the persistent Rust model, and decodes the response. A monotonic clock paces ticks; overdue schedule slots are skipped rather than accumulated. The runner writes per-tick actions, probabilities, latency, start lateness, missed deadlines, and skipped slots. It fails on an inference error; it has no actuator or fallback controller.

On the M2 Ultra, 300 ticks at 30 FPS took 9.97 seconds: **2.77 ms mean, 3.21 ms p95, 3.38 ms p99, 3.48 ms maximum** processing latency, with **zero scheduled deadline misses or skipped slots**. The frame budget was 33.33 ms. See [raw timing results](../../results/steering/realtime30/summary.json) and [tick records](../../results/steering/realtime30/ticks.csv).

This includes file read/decode, adapter, JSON/HTTP inference, and action decoding. It excludes initial model/adapter loading, ten warm-up ticks, sleep, and final report writes. The OS file cache may be warm. Frames repeat from 50 recorded images; model outputs do not affect future observations. Live-camera capture, exposure/buffering delays, camera transport, actuator latency, and closed-loop control are unmeasured. macOS scheduling provides no hard real-time guarantee.
