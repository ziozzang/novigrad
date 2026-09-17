# Causal feedback through a frozen language-model interface

[한국어](README.ko.md) · [HTML report](../../reports/neural-link/feedback.html) · [Biological and BMI research](../../examples/bio_bridge/CAUSAL_FEEDBACK_RESEARCH.md)

This experiment isolates a narrow question: can a frozen FunctionGemma consume continuous prefixes produced by an explicit host belief updater after an executed action receives feedback? It does not test learning by the language model, reward-modulated synaptic learning, fly cognition, or autonomous discovery of a game's objective.

The previous prototype-writeback study made the next input resemble the model's previous answer and preserved wrong fixed points. Here the physical executed action determines an observation, and the host uses that observation before constructing the next input. The environment is still a **four-choice hidden-goal bandit assay**: a fixed target, a decision budget, and termination. There is no continuous world model, resource depletion, locomotion, learned forward model, sensory prediction residual, or dopamine prediction-error implementation.

## Measured results

Under the compatible native701 executor, successes within four decisions were:

| Adapter | Unchanged | Gain .5 | Gain 1 | Static prior | Permuted prototypes | Same-first no-repeat host |
|---|---:|---:|---:|---:|---:|---:|
| Pooled MLP | 35/64 | 42/64 | 52/64 | 36/64 | 42/64 | 64/64 |
| Learned query | 32/64 | 45/64 | 52/64 | 36/64 | 48/64 | 63/64 |
| Fixed query | 33/64 | 44/64 | 45/64 | 35/64 | 41/64 | 64/64 |

Continuous full-gain host stimulation rescued 17, 20, and 12 initially failed episodes. Mean decision counts were 2.047, 1.984, and 2.094, but the learned adapter includes an invalid first call with an early stop; read efficiency with the conditional-success summaries and invalid counts. Self-writeback and flipped-feedback successes remained at the unchanged baseline in all three adapters.

The matched host references were designed **after observing the first two adapters' results**, are separately hashed, and are not part of the original frozen factorial. They force exactly the same first raw LM output/parsed command, including immediate failure on an invalid call, before using host logic. Same-first fixed-reliability Bayes reaches 64/62/64; same-first no-repeat reaches 64/63/64. Thus keeping the initial language interpretation identical does not remove the continuous-prefix interface's performance gap. The learned adapter's invalid first call makes its matched ceiling 63 rather than 64. [Matched-reference traces](matched-reference.json).

Full-gain minus label-permuted success differences, with an exploratory 10,000-draw bootstrap over 32 bilingual pairs, are pooled +15.63 percentage points (95% interval [1.56,29.69]), learned +6.25 [−3.13,15.63], and fixed +6.25 [0,14.06]. The learned/fixed comparisons do not establish an alignment-specific advantage across these architectures. These are unadjusted, authored-case, post-hoc intervals—not training-seed, topology, animal, or population inference. [All grouped and paired analysis](posthoc-analysis.json).

Full-gain successes with delayed/noisy feedback are respectively 44/42 (pooled), 47/47 (learned), and 45/43 (fixed), versus immediate 52/52/45. These are one fixed noise schedule per case. Under the incompatible native601 map, full-gain successes are only 27/25/18; no-repeat host search reaches the 32-case reachability ceiling. The controller does not learn an inverse actuator map.

The run comprises 84 conditions over the reused cases, with 2,926 distinct actual LM generations (903/951/1,072). All full environment/host/prefix traces replayed exactly from cached strings, and all 358 predeclared distinct generation replays (108/95/155) matched exactly. Base weights remained bit-identical in memory. This supports a working, reproducible host-guided interface, not a language-model or brain-like reasoning advantage.

## Prespecified diagnostic and data reuse

All 64 texts and all three 480-update adapters from the prior study are reused. These are 32 dependent English–Korean pairs across eight authored scenario families, already inspected in the earlier report. This is a post-hoc mechanistic diagnosis, not a new held-out generalization result. There is no new training, hyperparameter search, gain selection, or best-model selection.

The three adapters are circuit pooled MLP, learned query, and fixed query. The original BF16 FunctionGemma-270M and FP32 adapters remain frozen. The input features still originate in frozen EmbeddingGemma-300M and the unchanged SiteCodec601. Each adapter sees exactly the same initial case prefix across all language-model conditions. A single original language-model output is reused whenever its prefix is identical; repeated deterministic calls are not independent attempts.

The numerical runner, environment, tests, initial prefix tensors, prototype tensors, priors, cases, actuator checkpoints, and previous study's protocol/evaluation/verification identities are frozen before the run. The prior protocol transitively validates original model/config/tokenizer, embedding, native circuit, codec, and adapter contents. Every numerical outcome and generated string is retained.

## Four separate interfaces

1. **Controller observation.** The environment returns executed action, the action to which delivered feedback belongs, scalar reward, terminal flag, and step. It does not return the hidden target to the policy. Evaluators retain targets for scoring.
2. **Host belief update.** Initial masses are a temperature-one softmax of the prior study's summed response log likelihood divided by response token count. This removes the direct 15-versus-16-token sum comparison but does not calibrate probabilities. The host applies a Bayesian likelihood with fixed reliability 0.9. That common model is deliberately misspecified for both deterministic observations and the 25% noisy scenario; the analytic reference is not described as optimal.
3. **Belief-to-prefix mapping.** Four prefixes were obtained from the old training class-mean embeddings through the saved circuit codec and adapter. Their label ordering is supervised host calibration. Updated beliefs weight these prototype prefixes. The mixture is blended with the original prefix at gain 0.5 or 1 and rescaled to the original prefix norm. This is continuous stimulation engineered by the host, not a posterior computed or learned by the language model. Interpolated prefixes need not lie on the training manifold.
4. **Execution mapping.** Generated `set_goal` commands are converted to executed actions using native Rust inference on the four fixed old-training prototypes. These deterministic mappings are measured once and cached. Codec601 and all language input features remain unchanged while the executor varies. The experiment isolates cached actuator-map sensitivity, not full online sensorimotor dynamics.

The native601 command-to-action map is `[0,3,3,3]`: food and warmth actions are unreachable, so the balanced 64-case success ceiling is 32/64 regardless of reasoning quality. Native701 is `[0,1,2,3]`, making all goals reachable. Both checkpoints have identical ordered graph tensors and differ only in plastic weights. This removes the previous study's simultaneous MBON-feature change; it does not make the four-prototype calibration representative of arbitrary stimulation.

## Conditions and scoring

Every episode resets and allows at most four decisions. Correct executed actions physically succeed and terminate. Each executed action costs 0.05; observed reward is 0.95 for indicated success or −0.05 for indicated failure. Invalid generated calls terminate as failures without a physical action. Physical utility is success minus action costs. Invalid early stops can look artificially cheap, so raw action counts or utility alone must not define superiority. Conditional decision counts among successful episodes are reported separately and also have selection bias.

Actual language-model conditions:

- `original`: unchanged initial prefix every time; the primary deterministic no-change reference.
- `self_writeback`: the **commanded** goal's prototype, regardless of the executed action or outcome; a self-confirmation control.
- `belief_half`, `belief_full`: host posterior stimulation with gains 0.5 and 1.
- `static_prior_prefix`: full-gain prior-weighted prototypes after the initial action, with no outcome incorporated; this is distinct from unchanged input.
- `flipped_feedback`: full-gain stimulation after inverting the scalar observation delivered to the host. The host is unaware of the inversion.
- `permuted_prototypes`: full-gain posterior with a fixed cyclic label-to-prototype permutation.
- `uniform_prefix`: a fixed equal-weight prototype mixture after the initial action.

Separate host references bypass the LM entirely: `analytic_fixed_model` chooses the largest host posterior; `prior_ranked_no_repeat` tries commands once each in descending original-prior order, ignoring feedback for action selection. In a deterministic four-choice task with a compatible executor, the latter can solve every case by four attempts **without feedback**. Therefore final success alone cannot demonstrate a special feedback mechanism or a language-model advantage. Compare success by decision budget, first-error rescue, invalid calls, repeated executed actions, and action cost as well.

All LM conditions share the first decision and terminate immediately on physical success. Their cumulative success therefore cannot fall below the unchanged-prefix first-decision baseline: this assay cannot test destabilization after a correct action. Extra attempts alone can help, even with uniform or permuted prototypes. The static-prior, permutation, and no-repeat comparisons are consequently essential; a gain over `original` alone is insufficient evidence for information-specific feedback.

All ten conditions run with immediate feedback under both executors. Only `belief_full` and `analytic_fixed_model` additionally run with one-step-delayed feedback or 25% reward flips, with paired presampled noise indexed by case and decision. Delayed observations preserve the executed-action identity to avoid assigning an old outcome to the current command. These limited robustness conditions cannot rank all ten methods under noise.

Physical success still ends an episode when scalar feedback is delayed, noisy, or inverted. There is no next decision after a true successful reward: this mainly probes reactions to failure before termination, not learning from positive rewards. Terminal status is therefore informative, and flipped feedback is not a fully blinded deception experiment: a false positive after failure continues, while a false negative after real success ends immediately.

## Replay

```sh
.venv/bin/python examples/bio_bridge/causal_feedback.py verify
.venv/bin/python -m unittest discover -s examples/bio_bridge -p 'test_*.py' -q
.venv/bin/python examples/bio_bridge/analyze_feedback.py
.venv/bin/python reports/neural-link/build_feedback_report.py
```

Analysis refuses to overwrite an existing report. Do not delete published locks or run outputs to repeat a measurement. The initial `prepare`, `run`, and `verify` commands are intended in that order for an isolated new study directory.

Verification has two explicit layers: complete environment/host/prefix replay using saved model strings, and actual frozen-LM regeneration for every distinct prefix reached by eight predeclared case indices across all conditions (the museum family, four goals × two languages; not all eight families). It does not regenerate every prefix for all 64 cases. SHA identities and before/after base-memory hashes are checked; timings are not used as an accuracy criterion. This Mac MPS study is not a throughput or hard-real-time benchmark.

## Biological interpretation

Fly mushroom-body dopamine studies motivate distinguishing expected outcomes from delivered outcomes; efference-copy studies motivate distinguishing a motor command copy from an observed sensory consequence. Neither establishes that the host Bayesian updater, four labeled prototypes, or simulated Novi rates are those biological mechanisms. The accompanying research review distinguishes implemented assays from stronger proposed tests. Real neural recordings, identified compartments, measured behavior, and causal perturbations remain absent.

A successful host-guided rescue would show steering of a frozen LM through this interface. To attribute an additional capability to the LM or brain-like circuit, it would also have to outperform observation-matched analytic/search references or demonstrate information processing those references lack. Dynamic environments, controlled action–outcome decoupling, learned prediction models, and external neural/behavioral datasets remain separate future experiments.

## One-prototype follow-up

A separately locked **post-hoc** experiment replaces posterior-weighted mixtures with the single prototype selected by host posterior argmax. Compatible-executor immediate-feedback successes increase from 52/52/45 to 64/62/64. The learned adapter has 11 helped and one harmed case; a misspecified host belief can repeat an already failed action. These totals equal same-first host Bayes and do not exceed same-first no-repeat search. [Complete methods, counterexample and exact replay](../hard-feedback-routing/README.md).

## Portable bundle

The dated archive `novigrad-feedback-20260918.tar.gz` is intended for the [v0.1.1 release assets](https://github.com/ziozzang/novigrad/releases/tag/v0.1.1). It contains the previous bilateral bundle unchanged, this study, count-noise and hard-routing artifacts, Safetensors inputs, sources, reports, and a content manifest. Original Gemma weights and platform-specific native libraries are excluded.

```sh
python3 scripts/package_feedback.py unpack novigrad-feedback-20260918.tar.gz /new/path/feedback-replay
cd /new/path/feedback-replay
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-gemma.txt -r requirements-functiongemma.txt scipy==1.17.1
.venv/bin/python -m pip install .
.venv/bin/python examples/bio_bridge/replay_feedback.py --embedding-model /local/embeddinggemma --function-model /local/functiongemma --include-hard-routing --check-only
.venv/bin/python examples/bio_bridge/replay_feedback.py --embedding-model /local/embeddinggemma --function-model /local/functiongemma --include-hard-routing
```

Run the unpack command from the repository checkout; the destination must not already exist. Model-directory contents are checked against the original recorded hashes; relocation changes only path lookup. Actual regeneration uses the original Mac MPS precision/runtime assumptions. The separate count-noise verifier pins the original native shared-library hash, so rebuilding a wheel may not satisfy that strict fingerprint even when sources agree. Its original 80 tensors were replayed exactly, but it is not promised as a binary-independent portable replay. Archive unpack and relocated LM replay use the existing environment for distribution verification; that is not a clean-room dependency installation or cross-platform equivalence test.

### Runtime-template provenance correction

The first actual portable replay failed when the relocated model directories contained only files listed in the original model protocol. That inventory filtered by suffix and omitted FunctionGemma's `chat_template.jinja`; the tokenizer therefore had no chat template. The original in-place runs had that file available. We retain this failure in `portable-template-audit.json` and add an explicit supplementary runtime-dependency lock checked by the portable wrapper. Model weights and numerical artifacts are unchanged.

**The template hash is pinned after outcomes, not before them.** This correction cannot retroactively establish the template's identity at the original freeze. Subsequent exact generation replay establishes consistency with the now-pinned template. Supply the complete matching model tokenizer files, including `chat_template.jinja`; copying only the old model-lock entries is insufficient.
