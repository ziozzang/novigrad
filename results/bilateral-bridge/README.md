# Bilateral embedding–circuit–language interface study

[한국어](README.ko.md) · [Offline HTML manuscript supplement](../../reports/neural-link/bilateral.html) · [Primary-source research](../../examples/bio_bridge/BILATERAL_RESEARCH.md)

A continuous soft-prefix bridge successfully conditions the original frozen BF16 FunctionGemma-270M on engineered circuit features. It does **not** establish fly-thought decoding, a general brain interface, or an architectural leap. On 64 new authored role/time-distractor texts, the strongest observed non-oracle variant is ordinary pooled MLP conditioning: 35/64 correct generated calls. Learned queries reach 32/64; fixed queries 33/64. All variants remain reported; no final-set model selection or retraining was performed.

## Questions and design

This study separates two experiments rather than treating every interface as a language model. Both begin with frozen EmbeddingGemma-300M classification embeddings (768 dimensions). The circuit path uses a training-only PCA map and a fixed sparse rate circuit: 319 PN inputs, 5,177 KC units (104 active), and 96 MBON features. Fixed seeded projections reduce each site to 32 dimensions. The direct-embedding comparator uses three fixed 768→32 projections without the circuit. It has the same downstream adapter size, but the upstream circuits are not capacity-matched and MBON weights already contain old-train supervision.

1. **Temporal binding.** Three observations at times −2, −1, 0, each with three site tokens, form nine typed tokens. A resampler emits four 32-dimensional latents and a supervised four-class head predicts the newest observation. Earlier frames are sampled independently from old training data; current labels do not choose them. This is externally timed latest-frame classification, not learned recurrent memory. Learned queries, fixed queries, and pooled MLPs are compared using three seeds each. Fixed-query mode freezes query vectors only; attention and metadata parameters still learn. Temporal training uses 300 updates, batch 32, Adam 0.003. There are 384 correlated training episodes and 48 development episodes, from only 32 and 12 base texts.
2. **Actual language-model conditioning.** Three current site tokens at time zero—not the nine-token temporal task—produce four 32-D latents, each projected to 640 dimensions. These four continuous prefixes precede a constant 112-token prompt in actual FunctionGemma. Only FP32 adapters train; the original BF16 transformer remains frozen. The prompt defines one `set_goal` function with four possible arguments. This is a closed four-way supervised task using genuine autoregressive generation, not open tool discovery or language reconstruction. The primary score requires a valid parsed generated call with the right argument. Four-candidate summed log likelihood is secondary; warmth uses 16 response tokens versus 15 for other goals, so ranking can be length-biased.

LM variants share seed 1701 and paired training streams. AdamW uses learning rate 0.003, batch four, zero weight decay, and gradient clipping at one. Learned/fixed/pooled adapters have 25,728/25,600/25,787 trainable parameters. Five initial models use 120 updates; four non-oracle models also use 480. The longer budget was introduced after observing development performance, before freezing final evaluation. It follows the same deterministic stream prefix, so these are not independent repetitions. The 120-step oracle injects labels directly and is a leakage-positive control, not a deployable method or a same-budget bound for 480-step models.

## Final generated-call results

| Input / adapter | Updates | Correct / 64 | Valid / 64 | Shuffled correct / 64 |
|---|---:|---:|---:|---:|
| Circuit / learned query | 120 | 15 | 45 | 12 |
| Circuit / fixed query | 120 | 31 | 64 | 23 |
| Circuit / pooled MLP | 120 | 29 | 64 | 24 |
| Direct embedding / learned query | 120 | 14 | 64 | 16 |
| Label oracle (leaks class) | 120 | 64 | 64 | 20 |
| Circuit / learned query | 480 | 32 | 63 | 20 |
| Circuit / fixed query | 480 | 33 | 64 | 20 |
| Circuit / pooled MLP | 480 | 35 | 64 | 22 |
| Direct embedding / learned query | 480 | 19 | 64 | 18 |

Zero prefixes generate zero valid calls in every variant. A prefix averaged over old training inputs generates a constant valid goal and scores 16/64 in every variant. Thus zero-prefix failure alone does not prove that an interface conveys task semantics: it also removes the learned carrier for the call format. The shuffled condition preserves the true class in 20/64 rows (31.25%), with no exact row fixed points; it is not complete semantic erasure. Identical-prefix generation is cached, so logical control rows are not independent physical generation trials.

The final set contains 64 locally authored texts in 32 English–Korean pairs across eight scenario families. It includes other-speaker, historical, quoted, and hypothetical distractors. It is neither an external benchmark nor cryptographically blinded. Accuracy declined sharply from reused 12-case development data: the 480-step fixed query had 12/12 development calls correct but only 33/64 here. This is an important counterexample to extrapolating development fit into robust semantic transmission.

Post-hoc paired bootstrap analysis (10,000 draws over 32 translation pairs) gives learned480 minus fixed480 −1.56 percentage points, 95% interval [−18.75, 14.06]; minus pooled480 −4.69 points [−17.19, 7.81]; minus direct-embedding480 +20.31 points [6.25, 34.38]. These intervals describe this authored set under this resampling scheme, not training-seed stability, connectome causality, or population generalization. Eight-family resampling is also retained, with the limitation of only eight clusters. No multiplicity-adjusted superiority claim is made. See [post-hoc analysis](posthoc-analysis.json).

Increasing 120→480 updates helped/hurt 23/6 learned-query cases, 18/16 fixed-query cases, 17/11 pooled cases, and 19/14 direct-embedding cases. Aggregate gains do not mean every example improved.

## Temporal task results

Mean final accuracy over three seeds and four past contexts per base case:

| Source | Learned query | Fixed query | Pooled MLP |
|---|---:|---:|---:|
| Circuit | 52.47% | 57.81% | 33.33% |
| Embedding | 47.92% | 43.88% | 29.43% |

These episode counts are correlated. The earlier analytic latest-only ridge baseline scored 100% on development data; it was not evaluated as a predeclared final comparator. Do not compare that development number directly with the final table. Timestamp reversal, current-frame deletion/zeroing, row shuffle, and metadata-only results are preserved per seed in [temporal-final.json](temporal-final.json). Timestamp reversal can select old training examples, so success on the reversed target is not independent semantic generalization.

## Two-tick prototype writeback and checkpoint compatibility

Each of eight predeclared final cases starts independently. Tick 1 generates a goal; strict parsing either rejects it or selects that class's old-training mean embedding. The host converts this prototype to site features for tick 2. Native Rust circuit actions are recorded alongside shadow predictions. **Native actions do not feed the next LM input.** This is `LM goal → host prototype → site features → LM goal`, not environment/action feedback or learned recurrence.

Under checkpoint 601, fixed-query goals remain stable in 8/8 cases but five are wrong fixed points; learned-query goals remain stable in 8/8 with two wrong fixed points. Neither rescues a wrong first answer. Native actions match the generated goal in only 1/8 and 3/8 respectively.

A second diagnostic swaps only compatible plastic weights to checkpoint 701 after checking graph tensors and ordered port identities. It simultaneously changes MBON features and native readout, while the LM adapter stays unchanged. Native actions then match the generated goal in 8/8 for both adapters. Fixed-query language correctness remains 3/8; learned-query correctness falls from 6/8 to 4/8. Better command execution does not imply better intent inference. This is a joint feature/readout distribution shift, not an isolated causal effect of action feedback. Native/shadow maximum probability error is 1.49e-8, below the declared 2e-6 tolerance. The eight cases are a descriptive audit, not a representative closed-loop benchmark.

## Freeze, replay, and artifacts

`protocol-lock.json` freezes all numerical sources, training features, codec, adapters, temporal heads, both native checkpoints, final text, and full model/config/tokenizer identities before encoding the final set. `final-started.json` refuses silent repeated final execution. `evaluation-lock.json` pins outputs. Every adapter and codec is saved in Safetensors. The four 480-step manifests additionally prove identical full base-memory hashes before/after training and absent base gradients; the initial 120-step runs retain weaker frozen-parameter/disk identity evidence, with their exact historical source archived.

```sh
.venv/bin/python examples/bio_bridge/bilateral_study.py verify
.venv/bin/python -m unittest discover -s examples/bio_bridge -p 'test_*.py' -q
.venv/bin/python reports/neural-link/build_bilateral_report.py
```

Verification replays all temporal predictions, every candidate likelihood/rank for all 64 rows and four controls in nine LM variants, actual free generation on eight predeclared indices per condition (32 logical rows per variant), and all original/swapped two-tick cases. It excludes timings and does not claim bitwise retraining or full-64 generation replay. Same-batch candidate likelihood tolerance is 1e-5; BF16 batch-shape changes can alter likelihoods, so batch size is part of the protocol. The runtime uses Torch MPS for actual frozen-LM inference/backprop through the adapter and CPU for tiny temporal models; this round is not an MLX-C speed benchmark or a 30 TPS serving claim.

Weights are local generated artifacts under the repository's ignore policy. JSON alone cannot replay this run. See the accompanying bundle manifest and relocation launcher for exact-artifact packaging; original Gemma model files must be supplied separately with matching hashes. Install the native binding with `.venv/bin/python -m pip install -e .` and the recorded dependencies from `requirements-functiongemma.txt`. The saved numerical sources retain their original local model paths; relocation must preserve content hashes. Reproduce a new training experiment in an isolated checkout/output location rather than deleting these published locks.

## Biological evidence and next falsifiable work

All reported interface signals are software-derived rates, not recorded fly neural activity. A real public ORN spike/calcium import was attempted: metadata was retrieved, but the workbook endpoints returned 403/401; zero paired observations and zero fitted biological predictions were produced. [Import evidence](../orn-data-pilot/import-report.json), [paired-data review](../../examples/bio_bridge/PAIRED_DATA_RESEARCH.md).

The next useful experiments are external paired neural/behavior data with animal/session separation; genuine action→environment observation feedback with no-write and shuffled-action controls; graph- and supervision-matched topology nulls; matched-gain site ablations; and multiple adapter-training seeds on new independently locked text families. Current failures favor testing causal channels and calibration before increasing query count or adapter rank. No result here establishes biological cognition, a fly equivalent of a human prefrontal cortex, or mind reading.

## Portable replay bundle

The supplemental [replay archive](https://github.com/ziozzang/novigrad/releases/download/v0.1.1/novigrad-bilateral-20260917.tar.gz) is attached to the existing v0.1.1 release as a dated research artifact; it does not move the release tag. It contains small Safetensors, frozen sources, native build sources and result locks, but excludes the base Gemma weights and prebuilt native binaries. [Checksum](https://github.com/ziozzang/novigrad/releases/download/v0.1.1/novigrad-bilateral-20260917.sha256). The unpacker verifies the full inventory and hashes before writing a new directory. The relocation launcher changes directory lookup only and requires the original model contents.

```sh
.venv/bin/python scripts/package_bilateral.py unpack \
  /path/to/novigrad-bilateral-20260917.tar.gz /new/replay-directory
cd /new/replay-directory
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-functiongemma.txt scipy==1.17.1
.venv/bin/python -m pip install .
.venv/bin/python examples/bio_bridge/replay_bilateral.py \
  --embedding-model /local/google_embeddinggemma-300m \
  --function-model /local/google_functiongemma-270m-it
```
