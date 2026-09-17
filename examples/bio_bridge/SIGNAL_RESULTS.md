# Signal schedule and strength: measured results

## Scope

These diagnostics ask what happens when a feedforward `novi` policy receives a cue continuously, once, or periodically, and when input or reward magnitude changes. The frozen temporal and inference-strength studies reuse the same 32 English/Korean confirmation texts and 128-dimensional, 2%-active checkpoints selected in the previous bio-bridge campaign. The training-strength and acquisition studies instead create new engines from the inherited Gemma-bridge dataset: 32 train, 24 test, and 12 Korean examples. Neither source is a fresh holdout for this round.

Three sampling seeds share the dataset and anatomy; they are not independent biological replicates. Logical ticks are software steps, not seconds. All accuracy values are greedy top-1 classification. Correct-class probabilities are generally near 0.25 and are not calibrated confidence or sampled-action success.

The biological motivation and primary-source limits are in [SIGNAL_RESEARCH.md](SIGNAL_RESEARCH.md). No result here identifies a fly mechanism.

![Measured signal schedules, strength, and acquisition diagnostics](../../results/signal-bridge/signals.png)

## Four meanings of “remembering”

| Meaning | What this experiment implements | What it does not show |
|---|---|---|
| Maintained input | The same cue vector is supplied again at each cue-on tick. | Memory; the current input is sufficient. |
| Cue-off host memory | `hold`, `ttl8`, or leaky preprocessing stores or mixes vectors outside `novi`. | Learned recurrence, working-memory neurons, or autonomous engine state. |
| Associative weights | Previously trained checkpoint weights map current features to actions. | A temporal trace between calls. Zero input produces no class-specific recall. |
| One-shot acquisition | A separate study applies one rewarded sampled action for each of four class exemplars. | Reliable one-example learning; the measured primary result is negative. |

The requested direct comparison is therefore precise: after a single clean cue, **direct = 25.00%** while **hold = 52.083%**. The extra 27.083 percentage points come from an external host copying the last nonzero vector. They are not learned recall by the feedforward engine.

## Frozen inference schedules

Every episode lasts 24 ticks and switches to a deterministically paired different class at tick 12. `continuous` supplies the current cue every tick; `one_shot` supplies it at ticks 0 and 12; `periodic4` supplies it every fourth tick. The full factorial has **324 temporal runs**: 3 seeds × 3 schedules × 2 dose definitions × 3 backgrounds × 6 host modes. The primary mask excludes the two initial-cue ticks. The cue-off mask is derived from the actual pulse schedule; continuous has no cue-off observations.

### Clean input, per-pulse dose

Cells show mean post-initial-cue accuracy across three frozen checkpoints. Parentheses give cue-off-only accuracy when defined.

| Host mode | Continuous | One shot | Periodic-4 |
|---|---:|---:|---:|
| Direct | 52.08% | **25.00% (25.00%)** | 29.92% (25.00%) |
| Hold | 52.08% | **52.08% (52.08%)** | 52.08% (52.08%) |
| TTL 8 | 52.08% | 42.23% (42.23%) | 52.08% (52.08%) |
| Leaky τ=4 | 47.25% | 52.08% (52.08%) | 49.05% (48.61%) |
| Leaky τ=16 | 40.06% | 43.75% (43.75%) | 40.86% (40.62%) |
| Leaky τ=16 + cutoff | 40.06% | 43.75% (43.75%) | 40.86% (40.62%) |

Continuous direct reproduces the checkpoint’s 52.08% confirmation accuracy. Direct one-shot cue-off rows are zero, yielding a uniform policy; greedy ties choose class 0, which is 25% on balanced labels. Hold simply repeats the host-stored vector and therefore reproduces cue-on predictions.

TTL 8 makes the policy return to zero after the declared lifetime. Its 42.23% one-shot aggregate is the arithmetic consequence of some cue-off ticks receiving a held vector and later ones receiving zero. It is not learned decay.

Positive leaky traces do **not** implement forgetting in this engine. Native hidden activity is divided by its maximum ([implementation](../../src/plastic.rs#L315-L320)), so multiplying a nonzero trace by any positive scalar leaves the effective direction nearly unchanged. In the separate pure single-cue silence sweep, leaky τ=16 remained at 52.083% through tick 256 even though its unit-gain norm had decayed by `exp(-16)` to about 1.13×10⁻⁷. The cutoff condition returned to 25% by tick 64, and TTL 8 was already at 25% at the sampled tick 16. Forgetting appeared only with an explicit TTL or amplitude cutoff. The cutoff is a host rule at emitted-state norm 0.05, not a learned or biological threshold.

### Background 0.25

The background is a nonzero distractor vector during nominal cue-off ticks. It exposes a key failure: hold and TTL detect “any nonzero input,” so background overwrites or refreshes memory rather than being rejected as a distractor. The table reports `post_initial_cue_ticks`; for periodic schedules this mask includes later periodic cue pulses as well as cue-off ticks. Cue-off-only values remain separate in `temporal.json`.

| Host mode | Continuous | One shot | Periodic-4 |
|---|---:|---:|---:|
| Direct | 50.52% | 17.19% | 23.25% |
| Hold | 50.52% | 17.19% | 23.25% |
| TTL 8 | 50.52% | 17.19% | 23.25% |
| Leaky τ=4 | 44.98% | 25.09% | 36.51% |
| Leaky τ=16 | 39.30% | 27.60% | 34.85% |
| Leaky τ=16 + cutoff | 39.30% | 27.60% | 34.85% |

This is not robust cue detection. A practical memory gate would need a trusted event marker or a separately evaluated novelty/reliability detector; using the target label would leak ground truth.

### Dose and post-switch interference

Per-pulse dose uses amplitude 1 for each pulse. Equal-total dose makes the sum of cue amplitude equal to 1 within each 12-tick phase. All **clean** schedule/mode accuracies were identical across these two dose definitions. That invariance did not hold with a fixed 0.25 background, because lowering cue amplitude changed the cue/background direction: continuous-direct all-tick accuracy fell from **50.52% to 26.56%**, and periodic-direct fell from **25.52% to 22.40%**. Native normalization removes a common global scale; it does not remove a change in relative mixture.

At the tick-12 target switch, persistent positive traces mix old and new feature directions. On clean continuous input, first-phase accuracy was 52.08% for every leaky mode, but second-phase accuracy fell to 40.10% for τ=4 and 26.82% for τ=16; direct remained 52.08%. Long persistence therefore created interference rather than stronger recall. The cutoff matched uncapped τ=16 over this short horizon because the trace stayed above 0.05.

## Inference-time signal strength

Global input gain was swept from zero through 10⁻⁶, 0.01, 0.1, 1, 10, 100, and 10⁶. Every positive gain produced unchanged predictions; maximum probability change relative to gain 1 was at most **5.96×10⁻⁸**. Zero gain yielded the uniform 25% tie baseline. This is expected from maximum normalization inside the engine. Global scalar gain is therefore not represented as signal strength.

Relative direction does matter. A distractor feature was added at ratios 0, 0.1, 0.25, 1, 4, and 10. Mean correct-class probability fell toward/below 0.25 and predictions changed as the distractor rotated the input direction. This tests feature competition, not odor concentration. Frozen EmbeddingGemma direction, vector norm, host reliability, and biological stimulus intensity must remain distinct concepts.

## Training-time strength and one-shot acquisition

Training uses a new engine per condition, batch-1 online updates, learning rate 0.009375, active fraction 2%, and an identity class-action map. There are 18 strength runs (six conditions × three seeds), nine one-shot/replay runs, and three paired massed/spacing controls.

### Reward and input strength

| Condition | Test accuracy / correct probability | Korean accuracy / correct probability | Weight Δ L2 |
|---|---:|---:|---:|
| Input gain 0.01, reward 1 | 47.22% / 0.2535 | 36.11% / 0.2550 | 0.7324 |
| Input gain 100, reward 1 | 47.22% / 0.2535 | 36.11% / 0.2550 | 0.7324 |
| Reward 0 | 25.00% / 0.2499 | 25.00% / 0.2498 | 0 |
| Reward 0.1 | 58.33% / 0.2504 | 61.11% / 0.2505 | 0.0858 |
| Reward 1 | 47.22% / 0.2535 | 36.11% / 0.2550 | 0.7324 |
| Reward 4 | 47.22% / 0.2638 | 38.89% / 0.2697 | 2.9542 |

Input gains 0.01 and 100 produced exactly the same sampled actions as gain 1, with only roundoff-scale weight/probability differences. Reward magnitude instead enters the policy-gradient coefficient linearly ([implementation](../../src/plastic.rs#L376-L389)): larger reward increased weight displacement and mean correct probability. Greedy accuracy was non-monotonic because the distributions remained near uniform and argmax was tie-sensitive. The 58.33% at reward 0.1 is therefore not evidence that weak reward is optimal.

### One-shot acquisition result

One fixed training exemplar per class was presented once, for four total rewarded events. This produced **25.00% test and 25.00% Korean accuracy**, with correct probabilities about 0.2499 and no held-out improvement. That is the primary one-shot finding.

| Schedule | Test accuracy | Korean accuracy | Weight Δ L2 |
|---|---:|---:|---:|
| One shot: 1 event/class | **25.00%** | **25.00%** | 0.00425 |
| Replay 32× | 26.39% | 30.56% | 0.04877 |
| Replay 32×, reward divided by 32 | 25.00% | 25.00% | 0.00174 |

Thirty-two ordered replays yielded only a small greedy gain. Dividing replay reward by 32 matched total absolute reward exposure and returned near one-shot behavior. It is not an exact causal match because later sampled actions come from changing policies.

Massed and spaced conditions received the exact same 128 ordered observations, actions, and rewards; spacing inserted seven blank inference calls between rewarded events. For every seed the final weights were exactly equal. The engine has no clock, autonomous activity, consolidation, or decay, so blank inference cannot create a spacing effect. This negative control is expected and prevents a biological spaced-learning interpretation.

## Provenance and concerns

- The temporal/inference studies' 32 cases and three frozen checkpoints were selected in the preceding campaign; training studies reuse the inherited 32/24/12 split. Reuse is suitable for a mechanism audit but not confirmatory generalization.
- The temporal grid was declared before the runs, but it explores many correlated cells on the same examples. No confidence interval or multiple-comparison claim is made.
- Greedy accuracy can move sharply while correct probabilities remain near 0.25. Raw probability and entropy fields in the JSON are required for interpretation.
- The direct 25% versus hold 52.083% comparison measures explicit host storage. It cannot support claims of working memory, continuous neural recall, or a silent associative trace.
- Decay of vector norm is canceled by positive-scale normalization. A TTL or cutoff supplies forgetting by construction.
- Reward magnitude is distinct from sensory signal strength. Relative feature direction is distinct from both.

Numerical sources: [`baseline.json`](../../results/signal-bridge/baseline.json), [`temporal.json`](../../results/signal-bridge/temporal.json), [`strength.json`](../../results/signal-bridge/strength.json), [`training-strength.json`](../../results/signal-bridge/training-strength.json), and the descriptive [`summary.json`](../../results/signal-bridge/summary.json). A vector version of the figure is available as [`signals.svg`](../../results/signal-bridge/signals.svg).

## Reproduction

From the repository root, using existing frozen embeddings and checkpoints:

```bash
.venv/bin/python examples/bio_bridge/signal_recall.py baseline
.venv/bin/python examples/bio_bridge/signal_recall.py temporal
.venv/bin/python examples/bio_bridge/signal_recall.py strength
.venv/bin/python examples/bio_bridge/training_strength.py
.venv/bin/python examples/bio_bridge/summarize_signals.py
.venv/bin/python -m unittest discover -s examples/bio_bridge -p 'test_*.py' -q
```

These commands overwrite the corresponding result artifacts. They do not run FunctionGemma or EmbeddingGemma generation; they reuse saved features and weights.
