# Neural recording stability: bounded engineering analogue

Neuralink's 2019 platform paper describes flexible threads, many electrode sites, robotic insertion, onboard amplification and digitization, and simultaneous high-channel-count recording. It establishes acquisition hardware context; this repository does not implement or evaluate that hardware ([Musk & Neuralink, 2019](https://pubmed.ncbi.nlm.nih.gov/31642810/)).

Degenhart et al. aligned low-dimensional neural manifolds across recording blocks using stable electrodes and evaluated stabilization during macaque cursor control under baseline shifts, unit dropout, and tuning changes ([Degenhart et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7822646/)). NoMAD later used nonlinear latent dynamics to align later recordings to a reference session without later-session behavior labels ([Karpowicz et al., 2025](https://www.nature.com/articles/s41467-025-59652-y)).

`neural_link_stability.py` is intentionally simpler. It treats frozen engineered KC rate vectors as recordings, applies synthetic gain/offset and channel-dropout drift, and fits a paired affine ridge map from a small set of old-training anchors. It has no temporal dynamics, factor analysis, stable-electrode identification, implanted data, spike processing, or unsupervised later-session learning. Therefore it is neither a reproduction of Degenhart/NoMAD nor a Neuralink integration.

The importer accepts NPZ or Safetensors with `sample_ids` (unique integer trial IDs), `time` (finite seconds), `neural_features`, `channel_ids` (unique integer identities in feature order), and optional integer `targets` whose only meaning is class ID. Targets never enter calibration. Safetensors additionally require the `novi.recording` version-2 metadata contract. The importer rejects floating-point IDs, duplicate IDs or channels, nonfinite values, inconsistent shapes, unknown fields, and channel identity/order mismatch. Calibration artifacts contain the augmented anchor basis, dual coefficients, actual ridge value, and channel identities and must reproduce inference exactly after reload.

The external workflow aligns paired rows by `sample_ids`, verifies paired timestamps exactly by default (or against an explicit tolerance), and requires identical channel identities and order. It never assumes row order or infers channel correspondence from width. The CLI is:

```bash
python neural_link_stability.py fit --observed train-observed.safetensors --reference train-reference.safetensors --calibration affine.safetensors
python neural_link_stability.py apply --calibration affine.safetensors --input later-session.safetensors --output corrected.safetensors
python neural_link_stability.py verify --calibration affine.safetensors --input later-session.safetensors --output corrected.safetensors --manifest corrected.safetensors.manifest.json
```

Each operation writes hashes and dimensions to a JSON manifest. The repository's `import-demo.json` records the same path on simulated KC gain/offset recordings: 32 shuffled training anchors are paired by ID, the old validation recording is corrected, and reload output is compared with direct in-memory inference. Its timestamps are synthetic seconds for the software exercise; its channel IDs are the actual `hidden_ids` stored in the frozen graph checkpoint, in checkpoint feature order.

All drift parameters, ridge strength, anchor sizes, and sham pairing are frozen before validation. The old training split supplies anchors; the old validation split is evaluated. Results are descriptive software tests on authored embedding inputs, not neural or clinical evidence.

At 32 anchors, the complete [stability report](../../results/neural-link-bridge/stability.json) gives:

| drift | cosine: uncorrected | corrected | shuffled sham | accuracy: uncorrected | corrected | shuffled sham |
|---|---:|---:|---:|---:|---:|---:|
| clean | 1.000 | 0.815 | 0.674 | 0.833 | 0.833 | 0.333 |
| gain + offset | 0.693 | 0.810 | 0.654 | 0.833 | 0.833 | 0.333 |
| dropout 25% | 0.840 | 0.811 | 0.664 | 0.833 | 0.833 | 0.333 |
| dropout 50% | 0.681 | 0.802 | 0.650 | 0.833 | 0.750 | 0.417 |

Calibration improves cosine similarity for gain/offset and 50% dropout, slightly reduces it for 25% dropout, and degrades even clean vectors because ridge calibration is not an identity map. It produces no accuracy improvement and reduces accuracy under 50% dropout. The shuffled sham performs worse, supporting the value of correct pairing within this simulation, while the negative accuracy results rule out a general stabilization claim.

Reproduce the frozen study and the full import demo from the repository root:

```bash
.venv/bin/python examples/bio_bridge/neural_link_stability.py study --output results/neural-link-bridge/stability.json
.venv/bin/python examples/bio_bridge/demo_neural_recordings.py --out results/neural-link-bridge/import-demo.json
```

The generated [import demo report](../../results/neural-link-bridge/import-demo.json) includes hashes, manifests, a rejected channel-order mismatch, and exact serialized-versus-direct inference comparison.
