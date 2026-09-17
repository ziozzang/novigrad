# One-prototype routing versus continuous posterior mixtures

[한국어](README.ko.md) · [HTML report](../../reports/neural-link/feedback.html) · [Parent assay](../causal-feedback/README.md)

This post-hoc experiment asks whether blending several labeled prototype prefixes loses command fidelity. It was designed after inspecting the completed continuous-feedback study. It reuses the same 64 authored cases and three frozen adapters; it is not a fresh holdout or a new learning result.

## Intervention and outcome

After the identical initial FunctionGemma call, the same explicit host Bayesian updater computes a posterior from the executed action and observation. Continuous routing uses a posterior-weighted mixture of four prototypes. Hard routing selects the prototype at the largest posterior coordinate. Both restore the original prefix norm. No model weights, priors, feedback likelihood, executor, or decision budget change. Only the compatible native701 executor and immediate feedback are tested.

| Adapter | Continuous successes | Hard successes | Helped / harmed | Continuous mean actions | Hard mean actions |
|---|---:|---:|---:|---:|---:|
| Pooled MLP | 52/64 | 64/64 | 12 / 0 | 2.047 | 1.828 |
| Learned query | 52/64 | 62/64 | 11 / 1 | 1.969 | 1.844 |
| Fixed query | 45/64 | 64/64 | 19 / 0 | 2.094 | 1.766 |

These hard-routing totals equal the separately measured same-first host Bayes totals (64/62/64); they do not surpass the same-first no-repeat host totals (64/63/64). Equal totals alone do not establish identical decision traces. The host explicitly selects a class before the LM receives its prototype. This is evidence for an interface choice, not reasoning learned by the LM or fly circuit. Mean physical actions include unsuccessful episodes and the learned adapter's invalid first call, so they are not an unbiased efficiency comparison.

## A retained counterexample

The learned-query adapter is harmed on `bilateral-water-night_shift-ko`: the request says that the speaker has already slept and now wants to wet a dry tongue. Continuous routing executes rest → warmth → water and succeeds. Hard routing executes rest → warmth → food → rest and fails. The host's fixed 0.9 observation reliability discounts rather than excludes failed actions; after three failures its last posterior still assigns rest 0.4277 and water 0.3995. The routed LM therefore faithfully transmits a bad host decision. This is a controller-model failure, not necessarily a decoder error. A second failed case is the unchanged invalid initial output at case 31; hard routing never gets an opportunity to repair it.

This counterexample makes the architectural trade-off concrete: a discrete typed channel can preserve an upstream decision more reliably while also preserving its mistakes. A no-repeat strategy exploits the deterministic task's known structure better here. Noisy, delayed, changing-goal, many-action, or real biological tasks could change this comparison and were not tested in this extension.

A post-hoc trace audit additionally finds that every follow-up generated command equals the recorded host-belief argmax: 53/53 pooled, 55/55 learned, and 49/49 fixed. [Audit and input identity](host-command-audit.json). This localizes the learned hard-routing failure to the host selection rather than a failure to transmit the selected class. It is a reused-trace description, not an additional evaluation.

## Frozen artifacts and replay

`protocol-lock.json` was frozen before new decoding. An earlier preparation was superseded before any outcomes: `../hard-feedback-routing-predecode-v0/` retains the original lock, matching source snapshots, and an explanation of the added full cached-replay guard. No measured result was selected between these preparations.

The run reuses exact prefix-digest cache hits from the parent assay and performs 156 new actual frozen-LM generations. Original base weights are bit-identical before and after the run. `verify` validates all evaluation hashes, recomputes complete environment/host/prefix traces for all three adapters and 64 cases using saved generated strings, and reasserts identical first raw outputs and parsed commands. It additionally regenerates 16 distinct new prefixes (5/4/7 per adapter) reached by the eight prespecified museum-family cases. That actual regeneration coverage is four goals × two languages within one family, not every case or family.

```sh
.venv/bin/python examples/bio_bridge/hard_feedback_routing.py verify
```

See `results.json` for every trace and paired comparison, `generation-cache.json` and `new-prefixes.safetensors` for new decoding inputs/outputs, and `verification.json` for exact replay outcomes. The portable parent bundle includes this extension; base Gemma weights remain external.
