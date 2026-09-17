# Actuator calibration under abstract count noise

[한국어](README.ko.md) · [Feedback study](../causal-feedback/README.md)

The compatible actuator remained stable on its four calibration prototypes: intended-action agreement was **99.12% at 32 expected total counts and 100% at 128 or more** in this experiment. This does not establish robustness on arbitrary language inputs or real fly recordings. It localizes one tested failure source: the incompatible native601 mapping stays wrong even when observations become precise.

| Expected total counts across 319 PN inputs | Native601 intended agreement | Native701 intended agreement |
|---:|---:|---:|
| 32 | 47.04% | 99.12% |
| 128 | 49.80% | 100% |
| 512 | 49.97% | 100% |
| 2,048 | 50% | 100% |
| 8,192 | 50% | 100% |

These are means over three fixed noise seeds with 256 observations per class, or 3,072 noisy inputs per count setting. Both policies receive exactly the same inputs. Across all settings there are 15,360 distinct noisy inputs and 30,720 native inference rows, plus eight unperturbed baseline rows. They are repeated observations of only four old-training class-mean prototypes, not independent semantic examples, animals, sessions, or training seeds.

## Method

The old training class means pass through the unchanged SiteCodec601 PCA/PN encoder. For each nonnegative 319-D PN vector x, independent counts are drawn with per-port expectation `N * x_i / sum(x)`, where N is one of the five prespecified total-count values. Each observed row is normalized by its observed total and rescaled to the original PN sum. This isolates counting fluctuations from overall input magnitude. An all-zero observation would remain all zero and be counted, never silently resampled; none occurred in this run.

The original and compatible native policies process the perturbed rates through their actual Rust inference path in batches of 256. No weights, gains, thresholds, or encoders are fitted. Count settings 32/128/512/2048/8192, seeds1901/1902/1903, and all repeats were fixed before measurement. Raw integer counts, normalized float32 rates, float64 returned probabilities, and prototype inputs are saved in `observations.safetensors`.

The unperturbed native601 map is `[0,3,3,3]`; native701 is `[0,1,2,3]`. Thus less noise makes native601 reliably execute the wrong command for two classes. At N=32, native701 had 27 errors among 3,072 rows: 26 warmth errors and one water error. All 12,288 rows at the four larger count values were correct. Zero observed errors on those prototypes is not proof of zero population error.

The compatible policy's unperturbed intended-versus-best-other probability margin ranges from 0.01950 to 0.03799. Per-class intended agreement, agreement with the unperturbed action, and min/5th/median/95th/max margin distributions are retained in [report.json](report.json). These softmax margins are not calibrated biological confidence or an actuator safety guarantee.

## Reproduce

```sh
.venv/bin/python examples/bio_bridge/actuator_noise.py verify
```

All 80 saved tensors and all numerical report summaries replayed exactly. [Verification](verification.json). The source lock pins the original installed native shared-library hash as well as code, model, NumPy, and Python identities. Strict replay therefore expects that native binary fingerprint; the portable source bundle deliberately excludes platform-specific binaries. A rebuilt wheel can have a different binary hash even if its numerical behavior matches. This noise study's strict verification is distinct from the portable frozen-LM feedback replay.

The model is an abstract Poisson observation diagnostic. No count value is mapped to milliseconds, measured fly firing rates, a biological sampling window, or electrode statistics. Correlated noise, drift, dropped channels, altered encoding, unfamiliar inputs, and other action prototypes remain untested. The result does not justify choosing a universal observation budget or claiming a performance optimum.
