# Causal action–environment feedback for the Novi–LLM bridge

## Scope

The next experiment should ask whether a generated action changes an external state and whether the resulting observation improves the next decision. This is stronger than feeding the model a prototype selected from its own previous output. It remains an engineered software experiment: Novi rates are simulated, the environment is designed by us, and no result would demonstrate fly thought, dopamine physiology, or a biological brain–LLM interface.

Keep three signals separate:

```text
action copy:       c_t = copy(a_t) available before the outcome
sensory residual:  epsilon_(t+1) = o_(t+1) - F(s_t, a_t)
value error:       delta_(t+1) = clip(r_(t+1) + gamma V(s_(t+1)) - V(s_t))
```

An action copy predicts self-generated sensory change. A sensory residual detects mismatch between that prediction and the next observation. A value error concerns reward expectation. They may interact, but they are not interchangeable labels for “feedback.”

### Implemented scope of the first assay

The first implementation is deliberately weaker than the dynamic experiment proposed below. It is a four-choice hidden-goal bandit with a fixed demand, a four-decision budget, and termination when the executed action matches that demand. Feedback reports the executed action and a success/failure reward; it does not arise from locomotion, resource dynamics, or a learned forward model. Reward is therefore target-class matching inside the assay, not a physically grounded discovery signal.

FunctionGemma and Novi remain frozen. A host Bayes rule updates a belief over four supervised class prototypes and supplies continuous prefixes; there is no model learning, dopamine prediction error, sensory residual, or recurrent world-model learning. The run can test whether outcome-informed host stimulation changes later frozen-LM choices relative to self-writeback and other controls. It cannot establish autonomous goal discovery, general closed-loop control, or a biological feedback mechanism. The dynamic environment, expectation-matched prediction-error tests, and efference-copy perturbations below remain proposed follow-up experiments.

## Primary evidence and its limits

| Primary study | What was measured or perturbed | Constraint for this software study |
|---|---|---|
| [Felsenberg et al. (2017), *Re-evaluation of learned information in Drosophila*](https://www.nature.com/articles/nature21716) | Behavioral, imaging, and circuit manipulations showed that reactivated appetitive memories could undergo extinction or reconsolidation depending on whether the expected reward was omitted or delivered. Identified MBON–dopamine-neuron recurrence supported compartment-specific re-evaluation. | This motivates expectation-versus-outcome tests. It does not license a single global “dopamine scalar,” and software reward must not be called a measured DAN signal. |
| [König et al. (2021), *Dopaminergic mechanism underlying reward-encoding of punishment omission during reversal learning*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7893153/) | Imaging, optogenetics, and behavior identified a specific mushroom-body relay in which omission of expected shock recruited reward-encoding dopamine neurons during reversal. | Omission can have learned value, so reward sign cannot be inferred from raw stimulus presence alone. The result is compartment- and task-specific. |
| [Kim, Fitzgerald, and Maimon (2015), *Cellular evidence for efference copy in Drosophila visuomotor processing*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6327952/) | Whole-cell recordings in tethered flying flies found cell-type-specific motor-related visual inputs with timing and sign appropriate to cancel expected visual consequences of saccades. | Use a pre-outcome action copy and test its timing/sign. Do not treat all motor modulation as one generic vector or suppress unexpected external motion. |
| [Shanechi et al. (2016), *Robust brain-machine interface design using optimal feedback control modeling and adaptive point-process filtering*](https://pubmed.ncbi.nlm.nih.gov/27035820/) | Non-human-primate closed-loop BMI experiments separated intention estimation from adaptive neural decoding and evaluated control across sessions. | Offline decoding accuracy is insufficient: evaluate online trajectories, recovery after error, lag, and stability while the decoder and controlled process interact. |
| [Zhang et al. (2021), *A prototype closed-loop brain–machine interface for the study and treatment of pain*](https://www.nature.com/articles/s41551-021-00736-7) | Online neural-state detection triggered intervention in freely moving animals, and behavioral consequences were measured in the closed loop. | This supports detect–act–measure discipline, not equivalence between Novi ports and stimulation or between model state and animal neural recordings. |

## Smallest runnable experiment

Use a bounded two-step resource environment with four needs. Each episode begins from a logged latent environment state and exposes only a partial observation. FunctionGemma proposes a schema-validated action; the environment, not the model or target label, executes the transition and returns the next observation and reward.

```text
(hidden environment x_t, observation o_t)
  -> frozen embedding/Novi state -> frozen FunctionGemma tool call
  -> parse and order validation -> committed action a_t
  -> environment.step(a_t, rng) -> x_(t+1), o_(t+1), r_(t+1)
  -> optional action-copy prediction F(s_t, a_t)
  -> residual/value-error update -> next decision
```

Resources must have real dynamics: consuming water can reduce thirst but deplete the water site; turning can change which site is visible; a stochastic wind or obstacle can make the same action produce a different observation. Reward is computed from preregistered state changes such as need reduction and collision cost. The class label, desired answer, and model-generated justification never enter `environment.step`.

The primary endpoint is generated-rollout episode return and goal completion. Also report transition prediction error, recovery steps after an unexpected perturbation, invalid-call rate, and action/observation latency. Per-step classification remains diagnostic.

## Falsifiable controls

1. **No-effect action:** validate the call but prevent it from changing the environment. Any claimed closed-loop gain should disappear.
2. **Yoked outcome:** replay another episode's observations and rewards with matched marginals. This breaks the current action–outcome relation without making the input obviously corrupt.
3. **Action shuffle:** execute a different balanced action after logging the proposed action. If performance is unchanged, the generated action was not causal.
4. **Open-loop replay:** score the same prerecorded sequence without allowing actions to alter later inputs. Report this separately from generated rollout.
5. **Action-copy ablations:** remove, delay, sign-flip, or substitute the previous action copy while preserving observation and parameter budget. A useful forward model should be harmed selectively when self-generated motion must be discounted.
6. **Unexpected perturbation:** add external motion after the action. Correct cancellation should remove the predicted self-generated component while leaving the unpredicted residual detectable.
7. **Expectation control:** match delivered reward while changing its trained expectation. A value-error mechanism predicts a smaller update for expected reward and a larger update for surprising reward.
8. **Omission and delay:** omit an expected outcome or delay it beyond the eligible window. Compare exact-order, wrong-order, reward-shuffled, and magnitude-matched controls.
9. **Strong baselines:** text-only state serialization, observation-only policy, action-copy-only policy, direct analytic controller with the same observation, and a label-oracle ceiling marked invalid for deployment.

Run paired interventions from identical pre-action states and environment random seeds. Test both `same state + different action` and `same action + different environment disturbance`. This identifies the transition dependence that a self-confirming prototype loop lacks.

## Guards against circular evidence

- Build the next input only after `environment.step`; future observations or rewards cannot enter the current prefix, normalizer, or target.
- An action copy may predict an outcome but cannot count as evidence that the outcome occurred.
- Keep proposed, validated, committed, and executed actions distinct. Invalid calls cause the declared no-op/error transition.
- Fit forward models, value functions, normalizers, and thresholds on training/development episodes only. Group all time steps and language variants from one episode in one split.
- Use generated history for the primary result. Oracle-history and teacher-forced rollouts are ceilings, not closed-loop evidence.
- Freeze the environment transition/reward code and final episode manifest before final evaluation.
- Reset all recurrent state, queues, RNG streams, and pending actions between episodes.

Log the raw generation, parsed call, pre/post environment hashes, proposed and executed actions, RNG seed, observation, reward, predicted observation/reward, residual, update identifier, and causal timestamps. Saved replay should reproduce transitions and decisions from those logs.

## Rejection criteria

Reject the causal-feedback interpretation if a no-effect action, yoked outcome, or action shuffle retains the gain; if reward is computed from the expected class rather than an environmental consequence; if an action copy alone is presented as observed success; if the effect exists only under oracle history; or if a wrong-sign/wrong-time copy performs as well as the aligned copy on perturbation trials.

Even a retained result would establish only that an engineered action–environment–observation loop improves this frozen-model system. Biological alignment would require paired neural recordings, measured behavior, identified compartments, and causal perturbations in animals.
