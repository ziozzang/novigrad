# Vision validation — v0.1.0

[한국어 요약](VISION_REPORT.ko.md)

These are held-out test results from the actual Rust connectome-constrained engine, using supervised teacher gradients. The image adapter is label-free and fitted only on training images. No pretrained CNN or OCR model is used.

| Task | Training mode | Seed | Test accuracy | Exact 4-digit success | Frozen accuracy | Best epoch |
|---|---|---:|---:|---:|---:|---:|
| fashion | supervised | 1 | 72.60% | — | 10.80% | 12 |
| fashion | supervised | 2 | 72.45% | — | 10.80% | 9 |
| fashion | supervised | 3 | 72.80% | — | 10.80% | 14 |
| fashion | shuffled | 1 | 10.80% | — | 10.80% | 0 |
| captcha | supervised | 1 | 83.35% | 48.80% | 7.95% | 6 |
| captcha | supervised | 2 | 83.55% | 49.20% | 7.95% | 8 |
| captcha | supervised | 3 | 83.05% | 48.00% | 7.95% | 10 |
| captcha | shuffled | 1 | 11.10% | 0.00% | 7.95% | 5 |

Across three seeds, Fashion-MNIST accuracy averages **72.62%**; CAPTCHA character accuracy averages **83.32%**, and whole-string accuracy averages **48.67%**. These seeds share the same data split, so the range measures training variability, not uncertainty across datasets.

## Protocol

- Fashion-MNIST: balanced 10,000 training, 2,000 validation, and 2,000 official-test images; 10 clothing categories. Training and validation source indices are disjoint.
- Synthetic CAPTCHA: 3,000 training, 500 validation, 500 test four-digit images; full strings are unique across splits. Each 112×28 image is split into four known 28×28 cells, giving 12,000/2,000/2,000 character samples. All four predictions must match for exact-string success.
- Exact train/test image duplicate count: zero for both datasets. Fonts and noise distribution are shared across CAPTCHA splits; this does not test unseen fonts or unknown segmentation.
- Adapter: 637 HOG/pooled-pixel features, training-only standardization and 64-component PCA, fractional whitening 0.5, nonnegative signed-component encoding into 319 actual ALPN IDs.
- Circuit: 27,848 fixed ALPN→KC edges and 62,261 plastic KC→MBON edges, 10% hidden activity, gain 12, learning rate 0.0001. A fixed external opponent decoder assigns +1/−1 gains to output ports; it does not change biological synapse signs.
- Up to 25 epochs; patience 5; validation selects the checkpoint. Seeds 1, 2, 3 were fixed before test evaluation. Packaged models are seed 1, chosen before seeing test results.
- Shuffled control permutes training labels once and uses the same validation selection. The frozen baseline uses initial weights. The Fashion shuffled checkpoint selects epoch 0 because later updates do not improve validation.
- Tuning used validation only: 159 fully whitened components, then lower learning rate, 64 partially whitened components, and opponent decoding. Test was evaluated only after this configuration was fixed.

## Evidence and limits

Per-run `metrics.json`, `curves.csv`, and `predictions.tsv` are in [vision/](vision/). [Machine-readable summary](vision-summary.json), [dataset provenance](vision-data-manifest.json), and [independent NumPy/Rust checkpoint verification](vision-safetensors-verification.json) accompany this report. Run `python scripts/report_vision.py` to regenerate the summary from predictions.

Each supervised run changes 62,260–62,261 plastic weights. Accuracy is well above frozen and shuffled-label controls, demonstrating learning in this circuit. Probabilities remain relatively soft; argmax accuracy is not a claim of calibrated confidence. This is a clothing-image classification example and a locally generated fixed-cell digit-recognition example, not general object detection or a solver for arbitrary live CAPTCHA systems.

The bundled examples are predetermined test sample 0: Fashion truth `Shirt`, predicted `T-shirt/top` (incorrect); CAPTCHA truth `5420`, predicted `5420` (correct). They are retained to show both success and failure.

Measured on an Apple M2 Ultra, CPU only. Two task pipelines ran concurrently; elapsed times in metrics are descriptive, not isolated throughput benchmarks. See [reproduction instructions](../examples/vision/README.md).
