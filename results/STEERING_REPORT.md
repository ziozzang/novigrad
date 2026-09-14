# Small external-data steering imitation experiment

This experiment tests a deliberately modest goal: learning to imitate three steering commands from camera images with the existing Novigrad core. It does not build a complete autonomous vehicle stack.

## Measured result

The end-to-end imitation pipeline works, but this tiny sample does **not** demonstrate useful held-out steering generalization. The selected model mostly predicts straight.

| Measurement | Result |
|---|---:|
| Frozen → trained validation accuracy | 50% → 70% |
| Selected test accuracy | 60% (30/50) |
| Always training-majority class (straight) | 58% (29/50) |
| Shuffled-label control test accuracy | 58% |
| Selected balanced test accuracy | 35.90% |
| Shuffled balanced test accuracy | 33.33% |
| Decoded steering MAE | 0.20421 |
| Constant training-median steering MAE | 0.21571 |

Test recall was 0/8 left, 29/29 straight, and 1/13 right. The accuracy gain over the trivial baseline is only one image; the small MAE reduction is about 5.3%. These are weak results. Homeostasis **on** won validation at epoch 21; **off** reached 62% validation at epoch 2. Disabling normalization, which helped the larger clothing/CAPTCHA tasks, did not transfer automatically to this tiny steering sample. No further configuration was chosen after observing these test results.

The selected training process took 0.883 seconds on the M2 Ultra, excluding download, adapter fitting, controls, and inference. It changed 62,160 plastic weights. This confirms numerical learning and model persistence, not good driving behavior.

Raw records: [summary](steering/demo/summary.json), [selection](steering/demo/selection.json), [curves](steering/demo/homeostasis_on/curves.csv), [predictions](steering/demo/selected_test_predictions.tsv), [data manifest](steering/data-manifest.json), and [code/model hashes](steering/demo/provenance.json). Models are generated locally under `results/steering/demo/bundle/` and are not included in the already-published v0.1.1 release.

While the download was rate-limited, an auxiliary cached-data run used 58/20/31 images from four sessions within the source training split. Its accuracy was 61.29% versus 51.61% for its shuffled control ([records](steering/pilot/summary.json)). This was a separate convenience-sample pilot, not an independent replication or the official-split test above. The two predefined settings were unchanged for the final run.

## External data and split

The source is [URJC DeepRacer CARLA expert racing](https://huggingface.co/datasets/urjc-deepracer/carla-expert-racing), revision `4f993c8f052a9c94fcabbe4da63fc3eacf097c18`, declared Apache-2.0 by its publisher. The source contains simulated expert driving sessions. Its original train/valid/test splits have 20,697/18,623/7,776 rows and 10/10/4 disjoint experiment IDs.

The demo uses 100/50/50 rows without selecting on steering labels. A planned 300/150/150 uniformly spaced sample hit the viewer's HTTP 429 request limit after 109 training images. To keep this a small imitation example, the final training sample uses the first 100 positions of that original 300-point grid; validation and test each use 25 evenly spaced frames from two predefined 100-row blocks. This is a limited convenience sample, not a representative dataset benchmark. The exact row indices and group counts are recorded in the manifest. It downloads Hugging Face Dataset Viewer JPEG derivatives, not the original embedded Parquet images. Every image URL must contain the pinned source revision, split, and row index. The downloader records each JPEG hash, checks for exact cross-split JPEG duplicates, verifies session separation, and caps transferred data at 200 MB. It refuses an upstream revision change rather than silently using different data.

Images are converted to grayscale and resized to 28×28 with bilinear interpolation. The HOG/pooled-pixel/PCA adapter is fitted to training images only and emits 319 nonnegative rates. The fixed PN→KC and plastic KC→MBON circuit is the same one used in the clothing and CAPTCHA examples. No pretrained driving model is used.

Only pixels enter inference. Steering is the target: left below -0.2, straight from -0.2 through +0.2, and right above +0.2. Maneuver annotations and segmentation masks are excluded. Speed, timestamps, experiment IDs, and other control columns are also excluded from model input.

## Fixed protocol

Two candidates use the existing opponent readout, three actions, 64 PCA components, whitening 0.5, active fraction 0.2, gain 12, learning rate 0.0001, batch size one, seed one, and at most 25 epochs with patience five. They differ only in per-output L1 homeostasis. Validation accuracy selects a candidate, with cross-entropy breaking ties; `selection.json` is written before test inference. A shuffled-training-label control uses the selected configuration. No further settings are chosen from the test result.

The report includes class accuracy, mean per-class recall (balanced accuracy), confusion counts, and session-level accuracy. A coarse steering decoder weights each training class's mean steering by the model's predicted probability. Its absolute error uses normalized source steering units, not degrees, and is compared with a constant training-median predictor. A training-majority class predictor provides another simple test baseline.

## Interpretation limits

This is one small, thinned sample and one training seed. Frames within a session remain correlated; disjoint session IDs do not imply disjoint tracks, environments, or weather. Any result applies to these recorded images. It does not measure the consequences of model actions, compounding control errors, collision rates, recovery, or closed-loop reward learning. The task demonstrates the reusable sensor→rates→sparse core→action path to the extent supported by the measured result.

The three-bin decoder loses fine steering information, and opposing left/right probabilities can average to an apparently neutral command. Applications should retain the probabilities. A future simulator experiment could reuse the existing API or periodic runtime and supply delayed task rewards, but that is outside this small imitation example.

See [reproduction and local/API inference commands](../examples/steering/README.md).

## Real-time feasibility measurement

A persistent `novi_api` model and persistent Python adapter/HTTP connection processed prerecorded camera JPEGs at 30 FPS for 300 ticks on the M2 Ultra. Each measured tick includes JPEG file read/decode/resize, feature encoding, JSON/HTTP inference, response validation, and steering decoding. Mean latency was **2.765 ms**, p95 **3.211 ms**, p99 **3.380 ms**, maximum **3.481 ms**. No compute duration exceeded the 33.333 ms period; there were zero scheduled deadline misses and zero skipped slots. Maximum start lateness was 5.038 ms, illustrating that processing latency and OS wake-up timing differ.

The loop lasted 9.974 seconds, from the first tick's start through the final response; it does not wait for an extra full period after the last tick. Ten warm-up requests and model/adapter loading are excluded. The OS file cache may be warm, and the same 50 test images repeat. Actions do not modify replayed observations. This demonstrates headroom for a 30 FPS **software replay path** in this run, not measured camera-to-actuator latency or a hard real-time guarantee. Given the weak steering accuracy, inference throughput is currently less concerning than policy quality.

Reproduction: [replay command](../examples/steering/README.md#paced-replay-at-30-fps). Evidence: [summary](steering/realtime30/summary.json), [all ticks](steering/realtime30/ticks.csv), [platform and code/model hashes](steering/realtime30/provenance.json). The single-frame local and HTTP paths also agreed within 1.36e-10 probability difference ([verification](steering/demo/inference-verification.json)). Python tests: ten passed, including group leakage and steering metric checks; no Rust core changes were needed for this example.
