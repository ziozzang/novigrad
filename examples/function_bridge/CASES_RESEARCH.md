# Function calls, memory and adaptive behavior: an evidence-led case catalog

2026-09-17. We separate three questions: whether language maps to a valid API call, whether the called software behaves correctly, and whether a biological circuit supports the proposed computation. Success at one level is not evidence for the others.

## 1. High-level goals and low-level control

FunctionGemma is a 270M function-calling model intended for task-specific adaptation, not a general conversational controller. Google's Tiny Garden and Mobile Actions examples demonstrate application-specific command interfaces. Our equivalent is a four-tool vocabulary (`set_goal`, `observe_heading`, `choose_action`, `get_status`) with typed scalar arguments. [Official model card](https://ai.google.dev/gemma/docs/functiongemma/model_card)

This is a software control interface. The semantic/value bridge and anatomical candidate ports remain distinct. The encoder and novi can run once per changed high-level goal, while the motor controller runs frequently using current host observations. Generating text for every steering tick would add unnecessary delay and opportunities for invented observations.

The default MPS implementation uses original BF16 FunctionGemma and original FP32 EmbeddingGemma. No extra quantization is applied. A LoRA experiment trains separate adapter weights; it changes the model's behavior while leaving the base checkpoint file unchanged. It is not “identical original-model inference” when the adapter is enabled.

## 2. Goal switching

FC2, EPG and PFL3 experiments motivate separate goal and heading representations. FC2-related goals are spatial angles, not arbitrary semantic labels. The demonstrated biological comparison is more specific than a general executive function. [Mussells Pires et al., 2024](https://www.nature.com/articles/s41586-023-07006-3)

**Executed software case:** at step 60, a target bearing changes from 70° to −65°. A wrapped proportional controller acts on a persistent heading estimate. We compare it with a true-heading analytic control under the same disturbance sequence. This verifies a functional goal-switch mechanism, not FC2 neural dynamics. The target bearing is supplied by the evaluator; language-to-spatial grounding is not learned here.

## 3. Missing landmarks and working memory

Experiments support persistent heading representations and updating by self-motion, with errors when external cues are unavailable. A stateful estimate is therefore a justified computational hypothesis, but not perfect recall. [Kim et al., Science 2017](https://pubmed.ncbi.nlm.nih.gov/28473639/), [Turner-Evans et al., Neuron 2020](https://pubmed.ncbi.nlm.nih.gov/32916090/)

**Executed cases:** visual cues disappear for steps 35–74, with and without noisy odometry. Across three fixed seeds, heading RMSE in the interval is about 5.03° with odometry and 52.19° without it. A visible-cue condition provides a reference. Ground truth is available only to the evaluator and the separately simulated analytic control; the estimator gets noisy observations or an explicit missing value. The estimator is a circular state variable, not a reconstructed ring-attractor network.

## 4. Conflicting sensory cues

Fly heading studies find that cue informativeness and familiarity influence the weighting of conflicting cues. This supports testing reliability-sensitive fusion rather than assuming every observation is equally trustworthy. [Multimodal cue integration and learning, 2025](https://www.nature.com/articles/s41593-024-01823-z)

**Executed cases:** a visual cue is rotated by 115° during the conflict interval. An explicitly supplied low reliability reduces its weight relative to self-motion prediction. Interval heading error is about 27.17°, versus 114.48° with fixed trust. This result depends on the host providing the correct low reliability (0.005); discovering cue reliability is not implemented. Neither the input reliability nor true heading may be fabricated by FunctionGemma.

## 5. Hunger versus aversion

A fly food-choice study varied sweet/bittersweet options and deprivation. It identified hunger- and experience-dependent effects, with involvement of fan-shaped-body inputs and neuromodulatory networks. It supports testing internal-state-dependent choice rather than assigning one unchanging value to food. [Sareen et al., 2021](https://www.nature.com/articles/s41467-021-24423-y)

**Proposed, not yet executed:** a factorial task crosses hunger, caloric benefit and aversive cost. The environment owns these observations; a policy chooses approach or retreat and receives outcome feedback. Controls must include a simple utility model and a policy without internal-state input. A hand-coded priority rule could implement behavior but would not demonstrate learning or recover the biological circuit.

## 6. Reversal, extinction and recovery

Different dopaminergic cell types can write or update memories with distinct rules. This argues against treating all reward effects as one global dopamine scalar. [Aso & Rubin, 2016](https://elifesciences.org/articles/16135) Extinction research further warns that suppressing an old behavior can involve an opposing memory rather than erasing the original one. [Felsenberg et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5392358/)

**Already executed in the Gemma bridge:** the same novi weights continue training after actuator meanings rotate. The original setup fails; lower output gain with stronger entropy regularization and a longer training budget recovers 83–88% English accuracy in exploratory follow-ups. This is algorithmic policy adaptation, not a compartment-specific dopamine model. Context restoration and spontaneous recovery of old mappings remain untested. See [earlier report](../gemma_bridge/README.md).

## Interface boundaries and failure cases

FunctionGemma's actual outputs exposed several failure modes: undeclared enum values, copied example angles, extra calls, dialogue in place of calls, and correct schema with wrong intent. The dispatcher can reject bad syntax/ranges or forbidden tools; it cannot reliably identify a semantically wrong but valid `set_goal(water)` call. Model quality must therefore be measured independently of schema validity.

`observe_heading` records a **user-reported, unverified value**, explicitly labelled in status. It does not feed the navigation benchmark, whose observations are host-owned. Trusted outcomes use a separate HTTP capability and are never part of the model's tool declaration. Unknown/duplicate outcomes, goal changes while an action is pending, and invalid arguments are rejected before session mutation. This localhost prototype has one in-memory session; it is not a durable multi-user production service.

## Evidence hierarchy and next experiments

- Language tests: exact function and argument equality, frozen prompts, raw outputs, distinct development/exploratory/fresh sets; parser success alone is insufficient.
- Runtime tests: actual HTTP, actual models, exactly-once outcome application within a session, error atomicity and explicit trust boundaries.
- Control tests: deterministic seeded trajectories, matched disturbances and analytic references, missing observations, reliability ablation, no truth leakage.
- Biology: primary experiments and versioned candidate connectivity guide hypotheses; they do not validate our numeric constants or model dynamics.

The most useful next integration is a host-grounded target map: language selects an allowed goal ID, the environment resolves a target bearing, the memory controller maintains heading, and a calibrated action decoder steers. Compare conventional control, the functional CX-inspired mechanism, and eventually a connectome-constrained recurrent implementation. The ordinary controller remains an essential baseline, not an obstacle to obtaining a positive result.
