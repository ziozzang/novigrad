# Grounded control and adaptive mechanisms: measured research report

17 September 2026 · Apple M2 Ultra · foreground research continuation. [한국어](DEEP_RESULTS.ko.md) · [Primary-source research and experiment designs](DEEP_RESEARCH.md)

The strongest finding is not that adding a neural stage always helps. Task-specific LoRA improves valid tool calls but increases unsupported calls; heading memory helps during missing observations but residual gating fails under some biases; reward-conditioned novi learns a state-dependent choice, while an appropriately conditioned conventional policy performs better. These distinctions determine where a Software-defined Bionic NPU can add value.

![Measured mechanisms and failure modes](../../results/function-bridge-deep/mechanisms.png)

## 1. Protocol and evidence units

Three experiment suites cover language/physical grounding, non-oracle sensor fusion, and internal-state reward learning. Original FunctionGemma BF16 and EmbeddingGemma FP32 weights are unchanged; there is no additional quantization. Separate rank-8 Q/V LoRA adapters are trained on the same 76 examples for 80 optimizer steps, batch two, with seeds 731, 732 and 733. The earlier adapter 731 is retained for physical integration; it was not selected from the new evaluation to maximize performance.

A separate agent authored 80 new cases after reading only the tool schema and reported not reading training/example cases: 64 valid requests and 16 unsupported/compound/ambiguous requests, balanced English/Korean. Exact string overlap with the 76 training examples is zero. This is an independently authored software-agent set, not an independent human benchmark or a blinded population study. Interpretation of indirect requests and rejection labels remains partly subjective. Its SHA-256 is `8b54089bb9ed7832b613070e7946a992a322b4fcd5ba63c1d98920990dd1d19c`.

Each unique request gets one greedy unconstrained generation: no retries, repairs, added examples, or test-driven training-data edits. Raw tokens, parsed calls, expected calls, latency, model/adapter/config hashes and cases are retained. Repeated training seeds use the same test cases; they do not triple the independent sample size. The subsequent voting, four-times-budget and credit-assignment comparisons reuse evaluation data and are explicitly exploratory.

## 2. Language adaptation has a rejection cost

| Model | Valid exact calls /64 | English /32 | Korean /32 | Unsupported rejected /16 |
|---|---:|---:|---:|---:|
| Original BF16 | 22 | 15 | 7 | 14 |
| LoRA 731 | 49 | 26 | 23 | 5 |
| LoRA 732 | 48 | 27 | 21 | 5 |
| LoRA 733 | 41 | 22 | 19 | 6 |

The three-adapter average valid accuracy is **71.88%**, versus **34.38%** original. The range, 64.06–76.56%, shows that one successful run is insufficient. A paired bootstrap over the 64 authored cases, averaging the three adapters within each case, gives a 37.5-percentage-point gain and a descriptive 95% interval of 26.56–48.96 points. This interval describes this case set, not deployment coverage or uncertainty over all possible training seeds. Prompts also share author, language and paraphrase families; treating individual cases as exchangeable can understate that clustering uncertainty.

“Unsupported rejected” means no valid allowlisted call survived the parser. It is not evidence of a deliberate semantic refusal. Original-model conversational or malformed responses can also count as rejection. Adapted models often convert unsupported requests into a syntactically valid but wrong operation. Schema validation cannot distinguish these from intended calls. Do not enable the adapter as an unattended generic API operator based only on valid-case accuracy.

Short-call generation p50 is 0.284–0.301 seconds across adapted runs; p95 is 0.316–0.340 seconds. Timings include prefill and synchronized MPS generation, exclude loading, and are not a long-text TPS or hard real-time benchmark.

### Three-model voting does not solve correlated errors

Cached outputs from all three training seeds were compared using two fixed rules; this is post-hoc and adds no new model evidence.

| Rule | Valid correct /64 | Unsupported rejected /16 | Valid abstentions | Unsupported calls capable of changing state |
|---|---:|---:|---:|---:|
| Strict majority | 48 | 6 | 0 | 7 |
| Unanimity, otherwise abstain | 40 | 7 | 17 | 6 |

A majority fails to beat the best individual adapter; unanimity sacrifices coverage without adequately rejecting unsupported operations. These models share the same base and training data. Training-seed diversity is not sufficient task or error diversity. No ensemble is installed as the default dispatcher.

## 3. Real generated goals now drive physical target contact

The new integration reparses exact stored FunctionGemma outputs, creates a fresh typed session, invokes the actual frozen native novi policy, resolves the selected resource through a host map, and moves a simulated body toward that resource. The language model runs once per unique instruction in the evaluation script; its recorded output is replayed across environment seeds. The host invokes `choose_action` after an accepted `set_goal`. This is an in-process integration benchmark, not a new online HTTP or robot service.

Four resource locations lie on a radius-four circle. Their shared rotation changes across ten environment seeds. A host service supplies exact position and the known resource map; noisy visual heading and odometry drive the estimator. Each episode resets position, heading memory and noise streams. The policy is frozen, verified by unchanged weights. Reward is **+1 for entering the requested resource radius, −1 for another resource, 0 for timeout or an unusable instruction**. Matching a label without moving earns nothing. These reward rules deliberately prefer abstention to wrong contact.

| Controller/decoder | Original language: visible / dropout | LoRA 731 language: visible / dropout |
|---|---:|---:|
| Oracle semantic goal, same physical controller | 100% / 100% | 100% / 100% |
| Parsed enum directly to host map | 25% / 25% | 70% / 70% |
| Parsed enum → novi → heading memory | 25% / 25% | **70% / 70%** |
| Same novi, remove odometry | 25% / 24% | 70% / 66.5% |
| Rotate novi's decoded resource identity | 2.5% / 2.5% | 0% / 0% |

The oracle row replaces only the semantic goal, not noisy heading with true heading. The rotated-decoder control can occasionally correct an already wrong semantic decision; its nonzero original-model score is not evidence of useful decoding.

Correctly interpreted goals reach the intended resource with memory in these environments. The principal end-to-end bottleneck is language interpretation. **novi adds no measured benefit over the direct enum baseline here.** It reuses the previous learned semantic policy, but resolving an enum through embeddings is redundant for this task.

Each variant/scenario contains 40 requests × 10 seeds = 400 workload rows, but repeated requests with the same expected/selected goal and seed duplicate physical trajectories. The oracle has only 40 distinct goal/seed trajectories per scenario. Counts are not independent trial counts, and no binomial CI is computed over 400. Raw rows include language case and environment seed, and aggregates disclose unique physical configurations. Noise is coupled by timestep across variants; observed sensor values diverge when actions change the trajectory.

Control-only p50/p95 was approximately **9/11 microseconds** on this machine. That measures Python heading fusion and steering, excluding language, embeddings, model loading, simulated sensors and physics. It supports separating slow language decisions from frequent control updates; it is not a complete sensor-to-actuator deadline guarantee.

Limits: exact host localization, known maps, no obstacles, no moving/absent resources, no collision model, no learned steering and no anatomical CX dynamics. Goal switching, unavailable-target handling and physical online reward learning remain future work.

## 4. Non-oracle innovation gating: strong in one regime, harmful in another

Four filters receive the same generated sensor streams: fixed trust, explicitly labelled oracle cue reliability, odometry-only after initial visual calibration, and an innovation gate. Twenty seeds × six scenarios run for 120 steps. The gate uses only the wrapped visual-minus-odometry-prediction residual: full correction below 18°, decreasing weight until 35°, then no visual correction. These fixed constants were not tuned on the results.

| Scenario | Fixed trust MAE | Innovation gate MAE |
|---|---:|---:|
| Clean cues | 1.42° | 1.42° |
| Abrupt visual bias | 45.89° | **2.20°** |
| Gradual visual drift | 38.86° | 38.86° |
| Both visual and odometry biased | **46.57°** | 73.72° |

The abrupt-bias paired difference is about −43.69° (20-seed percentile bootstrap interval −43.99…−43.35°). Across the six fixed scenarios, the gate-minus-fixed mean is −2.756°; resampling the 20 per-seed means across scenarios gives −2.859…−2.651°. That narrow interval reflects repeated synthetic seeds under a fixed scenario mix, not robustness to arbitrary real-world shifts. The adverse simultaneous-bias result must not be hidden by the mean.

The gate is a **hand-selected outlier heuristic, not learned reliability**. Slow drift stays within its acceptance region; bad odometry can make a useful visual correction look implausible. Neither the experiment nor the cited biology proves that innovation residuals identify which sensor is wrong. More independent anchors, explicit uncertainty and retained alternative hypotheses are needed. The gate is retained as an experimental comparator, not substituted globally into the API or grounded controller.

## 5. Internal state changes what reward learning can represent

The real connectome-constrained native Engine receives 319 engineered nonnegative RBF/scalar ports encoding hunger, benefit and aversive cost. The task samples continuous independent values in [0,1]. Approach earns `hunger * benefit - cost`; avoid earns zero. Policies sample their actual action and receive only its reward; no optimal action label is supplied. The ports and utility are engineering choices, not measured PN coding or dopamine chemistry.

Three freshly instantiated, identically initialized connectome models vary training-context and action-sampling seeds; these are not three randomized anatomical initializations. The runs compare full-state novi, the same engine with hunger replaced by its distribution mean, a conventional linear policy, and zero-reward novi. Each uses a distinct action RNG and the same training contexts. The no-hunger comparison retrains a separate model with different observations; it is not an evaluation-time ablation of the same trained policy. Frozen greedy evaluation uses 2,048 independent contexts per seed. Short training is **2,048 interactions / 64 batch updates**, not 2,048 optimizer steps. The exploratory extension is 8,192 interactions / 256 updates with identical settings, fresh initialization and reused evaluation seeds.

| Policy | Mean utility: 2,048 interactions | Mean utility: 8,192 interactions |
|---|---:|---:|
| novi with internal state | 0.02395 | **0.03803** |
| novi without hunger | 0 | 0.00445 |
| Initial raw-feature linear policy | 0 | 0 |
| novi with reward disabled | −0.24290 | −0.24290 |
| Random-action reference | −0.12314 | −0.12314 |
| Analytic hunger-blind expected-utility rule | 0.04243 | 0.04243 |
| Full-information optimal choice | 0.05738 | 0.05738 |

The initial untrained engine favors approach, so the zero-reward control is not a random policy. Its weights are checked to remain exactly unchanged. The full-state policy's regret drops from 0.03344 to 0.01935 with the longer budget. Nevertheless, even its longer run is below the analytic hunger-blind reference: learning and optimization gaps remain larger than some benefits of the extra state. All real Engine variants are saved locally as Safetensors, with checkpoint/config/connectivity hashes in the reports.

### Same-policy causal input probe

A follow-up freezes each long-trained full-state checkpoint and changes only its hunger input at evaluation. Utility is still scored using the original environment state. Clamping hunger to 0.5 reduces mean utility from 0.03803 to 0.02008; shuffling hunger across test contexts reduces it to 0.01819. Greedy actions change on 8.12% and 9.55% of cases respectively. All hashes and weights remain unchanged. This is evidence that the trained policy uses the hunger channel beneficially on this synthetic distribution, distinct from the separately retrained no-hunger control. It remains an exploratory intervention on reused contexts, not biological validation. See `state-probe.json`.

### Diagnosing the failed conventional baseline

We did not interpret the raw linear baseline's collapse as evidence of a superior biological architecture. A fixed exploratory diagnostic compares raw features, range-based centering/scaling, a learned state-value baseline, and that baseline plus entropy regularization. All four use 8,192 sampled interactions, three seeds, common action uniforms and the same actual-action rewards. The learner receives no counterfactual reward or target action. No setting was tuned after the diagnostic results.

| Conventional policy | Mean utility | Regret |
|---|---:|---:|
| Raw-feature REINFORCE | 0 | 0.05738 |
| Centered/scaled REINFORCE | 0.04328 | 0.01410 |
| Centered actor-critic | 0.04332 | 0.01406 |
| Actor-critic + entropy 0.02 | **0.04563** | **0.01175** |

![Credit-assignment and conditioning curves](../../results/function-bridge-deep/credit-assignment.png)

Input conditioning recovers most of the improvement. The learned value baseline adds almost nothing at this budget; entropy supplies a smaller additional gain. The best conventional result exceeds novi's 0.03803, though these engineered feature representations and update rules differ. This motivates testing comparable input conditioning and exploration in novi; it does not establish a general architecture ranking. The diagnostic is exploratory because it followed inspection of the initial failure and reused test contexts.

## 6. What to retain and what remains open

Retain the typed separation between language, host grounding, state estimation, action and outcome; the resettable simulator; provenance and independent-language evaluation; and mechanism ablations. Retain LoRA as an explicit opt-in domain adapter, with unsupported-request behavior scored separately. Retain the sensor gate as a conditional comparator. Retain reward-only internal-state training and its stronger conventional reference.

Do not promote majority voting, schema validity, a correct action label, low generation latency, or a single favorable seed into general autonomy claims. None of these experiments implements FC2, hΔK, PFL populations or compartmental dopamine rules. The source-based roadmap distinguishes these candidate biological computations in [DEEP_RESEARCH.md](DEEP_RESEARCH.md).

Priority follow-ups are an explicit abstention/clarification contract with genuinely new negative examples; independent third-source anchors for cue reliability; motion-level interventions; absent and moving targets; and comparable feature-conditioning/exploration changes inside novi. A future connectome-constrained recurrent controller must beat the present conventional baselines and reproduce prospectively chosen perturbation signatures.

## Reproduce and inspect

From repository root, use the existing virtual environment, native binding and local model directories described in [README.md](README.md). The plotting helper additionally requires Matplotlib (measured version 3.11.2); install it separately with `.venv/bin/python -m pip install matplotlib==3.11.2` if needed. This run does not add an automatic model download or license acceptance.

```sh
.venv/bin/python examples/function_bridge/deep_evaluate.py --out results/function-bridge-deep/language-base.json
.venv/bin/python examples/function_bridge/deep_evaluate.py --adapter results/function-bridge/lora --out results/function-bridge-deep/language-lora731.json
.venv/bin/python examples/function_bridge/finetune.py --seed 732 --out results/function-bridge-deep/seed732/lora
.venv/bin/python examples/function_bridge/finetune.py --seed 733 --out results/function-bridge-deep/seed733/lora
.venv/bin/python examples/function_bridge/deep_evaluate.py --adapter results/function-bridge-deep/seed732/lora --out results/function-bridge-deep/language-lora732.json
.venv/bin/python examples/function_bridge/deep_evaluate.py --adapter results/function-bridge-deep/seed733/lora --out results/function-bridge-deep/language-lora733.json
.venv/bin/python examples/function_bridge/ensemble_audit.py
.venv/bin/python examples/function_bridge/grounded.py
.venv/bin/python examples/function_bridge/grounded.py --language results/function-bridge-deep/language-base.json --out results/function-bridge-deep/grounded-base.json
.venv/bin/python examples/function_bridge/reliability.py
.venv/bin/python examples/function_bridge/internal_state.py
.venv/bin/python examples/function_bridge/internal_state.py --train-interactions 8192 --output results/function-bridge-deep/internal-state-long.json
.venv/bin/python examples/function_bridge/state_probe.py
.venv/bin/python examples/function_bridge/credit_assignment.py
.venv/bin/python examples/function_bridge/summarize_deep.py
.venv/bin/python -m unittest discover -s examples/function_bridge -p 'test_*.py' -v
```

[Raw results, compact summary, figures and verification manifest](../../results/function-bridge-deep). Checkpoints remain local and Git-ignored; source code remains MIT, and model terms remain separate. The foreground continuation ends after three controlled suites, with gains, failures and unresolved questions recorded rather than a production-readiness claim.
