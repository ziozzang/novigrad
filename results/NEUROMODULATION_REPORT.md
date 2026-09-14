# Neuromodulation experiment — synthetic odor to virtual action

[한국어 요약](NEUROMODULATION_REPORT.ko.md)

This is a delayed-reward learning test of the actual topology-constrained Rust engine on a narrow synthetic two-odor task. Output labels are **virtual actions**, not identified foreleg motor neurons.

| Seed | Reward-trained | Frozen | Independent shuffled target | Reversal pre / post |
|---:|---:|---:|---:|---:|
| 1 | 100.00% | 33.00% | 9.00% | 100.00% / 100.00% |
| 2 | 100.00% | 33.00% | 39.00% | 100.00% / 100.00% |
| 3 | 100.00% | 33.00% | 65.25% | 100.00% / 100.00% |
| 4 | 100.00% | 33.00% | 82.25% | — |
| 5 | 100.00% | 33.00% | 28.75% | — |
| 6 | 100.00% | 33.00% | 44.00% | — |
| 7 | 100.00% | 33.00% | 29.75% | — |
| 8 | 100.00% | 33.00% | 70.75% | — |

Across eight fixed seeds: reward-trained **100.00%**, frozen **33.00%**, and independent-target control **46.09%** (range 9.00%–82.25%). The frozen 33% is an empirical policy result and **not** an assumed 50% chance baseline. Reversal was run for seeds 1–3 only.

## Protocol and interpretation

- Data: 32 sorted real ALPN input IDs from the FlyWire-derived PN→KC graph, divided into two synthetic odor groups. Each trial sets up to 12 of 16 in-group ports to rate 1 and four opposite-group nuisance draws to rate 0.15. Sampling is with replacement. The 400 evaluation patterns per seed use a separate PRNG stream; uniqueness against training is not asserted.
- Circuit: 27,848 fixed ALPN→KC edges, 62,261 sign-constrained plastic KC→MBON edges, 10% hidden activity, learning rate 0.02, logit gain 6, default per-MBON L1 homeostasis. The decoder maps 96 MBON ports into two engineered virtual actions.
- Delayed feedback: capacity 3 observations/episode, `lambda=0.8`, exponential baseline `alpha=0.02`, reward +1 for the designated action and −1 otherwise. Weights stay fixed for all three observations. On signal, `error = reward − baseline`; old records are scaled by `lambda^age`; gradients are averaged and applied once; baseline updates after application.
- Controls: `frozen` performs the same observations/actions but discards each trace, causing no weight changes. `shuffled` samples the reward target independently of odor for every episode; it is not a permuted fixed dataset. The wide range across seeds on this tiny task shows why the control should be inspected per seed.
- Reversal: 600 episodes, correct odor-action mapping flipped after episode 300. Accuracy on the original mapping is measured before reversal and on the flipped mapping afterward. Each accuracy uses 400 separately generated patterns.
- Checkpoints are Safetensors schema 4. Per-run JSON and exact CLI commands are in [neuromodulation/](neuromodulation/); checkpoints are locally generated and excluded from Git. Regenerate with `python3 scripts/run_odor_motor_experiments.py`.

## Biological boundary

DANs provide compartmental reinforcement signals to mushroom-body KC→MBON synapses ([adult MB connectome](https://elifesciences.org/articles/62576)); reward-prediction circuit interpretations also exist ([eLife model](https://elifesciences.org/articles/75611)). This implementation uses a **global scalar baseline and eligibility replay**, so it is neither an anatomical DAN simulation nor a measured dopamine rule. FlyWire here is a [brain connectome](https://www.nature.com/articles/s41586-024-07558-y); linking its descending outputs to VNC leg motor circuits requires separate data and matching ([brain/VNC comparative study](https://www.nature.com/articles/s41586-025-08925-z)).

The local `data/neurons_783.tsv` (139,248 rows) has annotation-field counts DAN 331, MBON 96, descending 1303, and `motor` class 0. These counts do not establish an MBON→foreleg motor path.
