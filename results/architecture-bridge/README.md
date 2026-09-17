# Architecture falsification study — 2026-09-17

**No architectural quantum jump was demonstrated.** Context-dependent routing solves the constructed rule-binding task, but a direct embedding baseline matches it on ordinary cases and exceeds the KC path on adversarial language. More probe sites and larger adapter ranks are not sufficient explanations for better robustness.

Open the [English HTML supplement](../../reports/neural-link/architecture.html) or [Korean supplement](../../reports/neural-link/architecture.ko.html) for all conditions, failures, hashes and sources. Design research: [architecture](../../examples/bio_bridge/ARCHITECTURE_RESEARCH.md), [probe locations, learned tokens and mHC](../../examples/bio_bridge/PROBE_RESEARCH.md).

## Design

- Frozen local FP32 EmbeddingGemma-300M; exact encoder weight hashes match the original training-feature provenance. Current model/config hashes are locked too. The older training provenance did not record every config hash, so historical config identity cannot be proven retroactively.
- Old train32 and reused validation12; no hyperparameter selection in this round. Two independently authored evaluation sets have 64 texts each, representing 32 bilingual pairs in eight families. They are not external benchmarks, animal recordings, or 64 independent observations.
- Context targets cross every base case with four balanced permutations. The cyclic table and an alternative Latin table were specified before evaluation. Expanded rows are repeated measurements of the base cases.
- Offline ridge probes use regularization 0.1. PCA is fitted on training data only, with effective rank31. KC activity uses the frozen graph and 2% competition. This is not native reward training.
- Full interaction, additive, blind, coefficient-count-matched interaction, direct embedding and PCA controls; context shuffle/missing/stale/reset controls. The externally supplied host latch is an exact plumbing invariant, not learned memory.
- Site probes read PN, post-ReLU raw KC, sparse KC and previously trained MBON. Each readout has 32 features and 132 coefficients. Equal coefficient counts do not equal equal information, effective rank or biological history. Multi-site and repeated-KC views are controls for fusion. Landmark4/16/64 are fixed random group sums, not learned tokens or anatomical landmarks.
- Direct bridge adapters: frozen/head-only, rank4/16/64 and unrestricted 768×128 delta; three seeds11/23/47, full-batch Adam, 300 epochs, learning rate0.03. MPS FP32 training and CPU saved-weight evaluation. This is not transformer LoRA or full EmbeddingGemma fine-tuning.

## Results

All percentages below are descriptive; site/adapter values average three software seeds. No significance or independent-biological-replication claim is made.

| Fixed comparison | Ordinary | Adversarial |
|---|---:|---:|
| Direct embedding ridge | 98.44% | 76.56% |
| Context KC interaction | 98.44% | 67.19% |
| Coefficient-count-matched KC interaction | 98.44% | 70.31% |
| Context direct embedding interaction | 98.44% | 76.56% |
| Context PCA interaction | 98.44% | 75.00% |
| Single sparse-KC site, 32 features | 92.71% | 65.63% |
| Multi-site, 32 total features | 84.38% | 68.23% |
| Fixed landmark4 | 52.60% | 48.44% |
| Fixed landmark16 | 81.25% | 59.90% |
| Fixed landmark64 | 83.85% | 66.15% |

KC interaction drops to 26.95%/24.22% when evaluation contexts are shuffled. Blind and additive context controls stay near the 25% chance level. This supports the need for the engineered interaction on this constructed task; the direct embedding interaction performs the same operation without the circuit.

Adversarial KC accuracy is only 25% on hypothetical needs, 50% on quoted distractors and 50% on other people's needs. Direct embedding is also imperfect: 50%, 50% and 75%, respectively. Simple negation cases reach 100% for both; this does not establish general negation understanding. Do not average away these counterexamples. Ridge-softmax values are uncalibrated scores.

| Direct adapter | Trainable parameters | Ordinary | Adversarial |
|---|---:|---:|---:|
| Head-only | 516 | 95.31% | 71.88% |
| Rank4 | 4,100 | 92.19% | 64.06% |
| Rank16 | 14,852 | 98.44% | 71.88% |
| Rank64 | 57,860 | 97.92% | 73.44% |
| Unrestricted bridge | 98,820 | 98.44% | 73.96% |

The full path is linear: `(x−mean)(W0+AB)Hᵀ+b`. With four outputs, rank4 already permits the maximum task-update rank. Larger ranks vary optimization/regularization, not the expressive class of four-logit linear functions. A shared learning rate/step budget is controlled but not proof of optimizer fairness. These results cannot reject the usefulness of larger nonlinear adapters or actual transformer fine-tuning.

The mHC-inspired check is **only algebraic**. A doubly stochastic matrix preserves stable mixing yet a uniform matrix has rank1 and erases stream differences; a permutation can preserve norms while changing identities. This is a counterexample to “stable mixing proves semantic alignment,” not a reproduction or rejection of mHC training.

## What to build next

Use a typed site/time interface before adding more parameters: site identity, sampling time, channel mask, normalization and context availability must be explicit. Test a small learned query resampler against fixed queries and an equally sized MLP; keep semantic transport and action-policy paths separately ablatable. Compare learned context state with a direct recurrent baseline and an analytic host-state baseline. mHC-style constrained fusion is a candidate stability control, not the default architecture.

For rank scaling, use a higher-dimensional objective (held-out semantic reconstruction, predictive neural dynamics or paired recording alignment) and substantially more independent training groups. Actual encoder LoRA/full tuning should have its own training corpus, frozen external evaluation, memory/throughput measurements and unrelated-language forgetting checks. Current evidence does not justify replacing the default engine or treating a direct bypass as neural decoding.

## Reproduction and integrity

The checked-in JSON retains per-case predictions, seed results, source/input hashes, exact inventory and encoder provenance. Safetensors remain local generated artifacts under the repository's existing ignore policy. Replaying this exact run requires those local files; JSON alone is insufficient. Start with the existing Gemma/Bio bridge preparation workflows to regenerate their training features and seed601 checkpoint.

```bash
.venv/bin/python examples/bio_bridge/architecture_study.py verify
.venv/bin/python -m unittest discover -s examples/bio_bridge -p 'test_architecture*.py' -q
.venv/bin/python reports/neural-link/build_architecture_report.py
```

For a new run, use an isolated checkout/output location and execute `develop`, `freeze`, `encode`, `final`, `verify` in that order. The Python API allows `architecture_study.OUT` to be set to a new directory inside the repository before calling these functions. Do not delete the published run to repeat it. The final operation refuses a second write, while `verify` only loads saved parameters and checks exact replay. One-shot access is a recorded procedure, not cryptographic blinding in a locally readable workspace.

Validation: 83 bio-bridge unit tests passed, including16 new architecture tests; both final sets replayed exactly from saved weights. This is saved-inference reproducibility on the recorded environment, not bitwise MPS retraining or biological replay. Random site projections depend on the recorded NumPy version. The general runtime was not modified.
