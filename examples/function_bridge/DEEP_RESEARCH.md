# Deep research design: grounded goals, learned cue trust, and hierarchical action

Updated 17 September 2026. This document proposes experiments; it does not report new runs. It separates findings from primary studies, engineering analogies motivated by those findings, and claims that would require new evidence.

## Scope and evidence boundary

The intended system has several levels that should not be collapsed into one “fly-like executive”:

1. A language model maps an instruction to an allowlisted semantic goal or skill.
2. A host-owned world model grounds that symbol in current objects, locations, sensor observations, and task constraints.
3. A navigation state estimator maintains heading and evaluates cue reliability.
4. A goal representation specifies a spatial bearing or vector.
5. A comparator converts heading-versus-goal error into steering commands.
6. Descending and motor layers turn commands into physical motion.
7. Environment-owned outcomes provide reward or failure evidence.

Primary fly studies constrain parts of levels 3–7. They do not show that flies represent language, that a semantic need is an FC2 state, or that a generic reward scalar reproduces mushroom-body dopamine circuits. Robotics studies motivate the language/action hierarchy, but they do not supply biological validation.

## Three biological roles that must remain distinct

### Mushroom body: learned value and memory-dependent choice

The mushroom body (MB) is a central site for associative learning. Kenyon-cell representations converge onto MB output neurons (MBONs); experiments support a relationship between MBON ensemble balance, learned valence, and memory-guided action selection. Extinction work further shows that reduced expression of an old response can arise from a parallel opposing memory rather than deletion of the original trace. [Aso et al., 2014](https://elifesciences.org/articles/04580), [Felsenberg et al., 2018](https://pubmed.ncbi.nlm.nih.gov/30245010/)

Engineering implication: MB-inspired machinery is a candidate for learning cue or outcome value. It should not be described as the storage location for a compass heading merely because both systems influence behavior. A single global reward update is an abstraction, not a reconstruction of compartment-specific KC–MBON plasticity or dopaminergic teaching signals.

### Central complex: heading, spatial goals, and context-specific memories

EPG population activity supplies a persistent heading representation with attractor-like dynamics. Sensorimotor experience can remap visual input onto this representation. More recent multimodal experiments show that cue informativeness and familiarity affect cue weighting and remapping. [Kim et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28473639/), [Fisher et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7753972/), [Basnak et al., 2025](https://www.nature.com/articles/s41593-024-01823-z)

FC2, PFL2, and PFL3 studies support comparison of allocentric heading and goal signals during particular navigation tasks. PFL3 populations contribute lateralized steering, while PFL2 activity is associated with increased steering strength at large directional errors. These experiments concern angular goals, not arbitrary words such as “water” or “rest.” [Mussells Pires et al., 2024](https://www.nature.com/articles/s41586-023-07006-3), [Westeinde et al., 2024](https://www.nature.com/articles/s41586-024-07039-2)

Two 2026 olfactory studies make “goal representation” more specific, not more universal. Kathman et al. observed odor-evidence integration and post-odor directional persistence in a VT062617-labelled fan-shaped-body local-neuron population dominated by hΔK. A targeted hΔK split line supported a more limited causal result than the broader VT062617 line. The effect therefore should not be generalized to all fan-shaped-body neurons or reassigned to FC2. [Kathman et al., 2026](https://www.nature.com/articles/s41467-026-75945-2)

Siliciano et al. found that flies tracking an odor corridor stored and updated directions associated with returning to the plume boundary. EPG perturbation disrupted directed returns, and FC2 activity was associated with a return-direction memory in that task. They found no evidence that FC2 represented every state-dependent direction in the behavior. [Siliciano et al., 2026](https://pubmed.ncbi.nlm.nih.gov/42486983/)

Engineering implication: use separate typed state for heading, task context, accumulated odor evidence, and target bearing. Treat FC2-like, hΔK-like, and other fan-shaped-body mechanisms as competing or complementary hypotheses tied to specific tasks. Do not implement a universal `goal_neuron` variable and cite all of them as support.

### Descending pathways: motor execution and arbitration

PFL output reaches lateral accessory lobe and descending pathways, but a structural route does not by itself define motor function. Experiments identify descending neuron types that predict and influence orientation during walking and show that steering commands are combined with ongoing locomotor state. [Namiki et al., 2018](https://elifesciences.org/articles/34272), [Rayshubskiy et al., 2025](https://elifesciences.org/articles/102230)

Engineering implication: the navigation comparator should produce a bounded command consumed by a separate actuator layer. Arrival, collision, velocity, and energy are properties of the environment and plant, not of the semantic goal decoder.

## Timescale separation

Language interpretation should run on instruction changes, clarification events, or deliberate replanning. Semantic grounding should run when the world model changes enough to alter the target. Heading estimation and motor control should run at sensor/control rate. Cue-reliability learning should update over enough observations to distinguish persistent noise from a transient conflict. Value learning may span trials and contexts.

This separation is consistent with hierarchical robot systems. SayCan combines language-level preferences with affordance or value estimates grounded in available skills. RT-H inserts language-described motions between high-level tasks and low-level actions and evaluates the hierarchy on robot tasks. These are engineering precedents, not fly models. [Ahn et al., 2022](https://proceedings.mlr.press/v205/ichter23a/ichter23a.pdf), [Belkhale et al., 2024](https://www.roboticsproceedings.org/rss20/p049.html)

A useful contract is:

- language layer: `goal_id`, constraints, and confidence about the parse;
- grounding layer: target object and target bearing with provenance;
- estimator: heading distribution and learned cue-quality state;
- controller: bounded turn/forward command;
- environment: timestamped sensor events and physical outcome;
- learner: update keyed to the committed action and observed outcome.

The language model must not fabricate sensor values, cue reliability, target contact, or reward.

## Experiment 1: learn cue reliability without an oracle field

### Question

Can the estimator learn which heading cue is informative from observation history, rather than receiving a host-authored reliability number?

### Design

Simulate visual and wind cues with independently varied circular noise, bias, dropout, and latency. Hide those parameters from the policy. Give the estimator only cue samples, availability flags, and odometry. It maintains an online uncertainty estimate from prediction residuals and cross-cue consistency. Conflict trials rotate one cue after a history period.

### Controls and falsification

- Fixed equal trust.
- Odometry only.
- Oracle reliability as an upper bound, never as an evaluated input.
- Reliability histories shuffled across episodes.
- A learner receiving cue identity but no history.

The hypothesis fails if inferred trust does not predict each cue’s future error, if the estimator follows a historically worse cue in conflict, or if shuffled histories perform equally well. Report circular heading error, calibration of predicted uncertainty, conflict capture, and recovery after cue restoration.

### Self-confirming-trust caveat

Residuals measured only against the estimator’s own state can make the currently favored cue appear reliable. At least one trust signal must come from information not generated by that estimate: delayed physical displacement, independent odometry, cross-validation on future observations, or intermittent externally anchored landmarks. The analysis must show results with the anchoring source removed or corrupted.

Biological evidence: cue informativeness and familiarity affect weighting and remapping in the fly heading system. Software analogy: an explicit online statistical estimator. The paper does not establish the proposed estimator or its update rule. [Basnak et al., 2025](https://www.nature.com/articles/s41593-024-01823-z)

## Experiment 2: familiarity-driven remapping with counterbalanced cue histories

### Question

Does prior coherent experience cause a cue to guide remapping of a newer cue, and can trust change when the familiar cue becomes unreliable?

### Design

Expose the agent to cue A during varied self-motion, then cue B, then both together. Introduce controlled conflicts and finally test each cue alone. Counterbalance A/B order, exposure duration, noise, and modality. In a second phase, make the initially familiar cue persistently unreliable.

### Controls and falsification

- Frozen cue mappings.
- Plastic mappings with no familiarity state.
- Reversed order and matched exposure.
- Conflict without self-motion.

The hypothesis fails if order has no effect under matched quality, if the newer cue never acquires the learned offset, or if familiarity remains dominant despite sustained evidence that it is inaccurate. Measure pre/post cue offsets, capture probability, learning rate, and retention after separation.

Biological evidence: visual-to-heading mappings change with sensorimotor experience, and familiar cues can guide remapping. Software analogy: adjustable cue-to-heading offsets and uncertainty. [Fisher et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7753972/), [Basnak et al., 2025](https://www.nature.com/articles/s41593-024-01823-z)

## Experiment 3: intermittent odor evidence and task-specific goal memory

### Question

Can an agent learn when to persist with a direction through expected odor gaps and when to abandon it for search?

### Design

Use a physically simulated turbulent plume. The agent receives intermittent odor encounters, wind, visual motion, and odometry. A separate evidence state integrates odor encounters and gates persistence of the current goal bearing. Plume statistics change between environments; no fixed “correct persistence time” is supplied.

Compare two goal-memory hypotheses:

- a context gate that controls whether an existing directional goal persists;
- a vector memory that stores or updates the return direction to a plume boundary.

### Controls and falsification

- Instantaneous odor reflex.
- Fixed persistence duration.
- Randomized odor times preserving encounter count.
- Heading estimator intact versus impaired.
- Evidence integrator reset between episodes versus intentionally carried across matched contexts.

Reward comes only from physically reaching the source or reacquiring a specified plume boundary, with time and energy costs. The hypothesis fails if integration does not improve source acquisition on held-out plume statistics, if randomized timing performs equally well, or if the memory does not adapt when blank-duration statistics change.

Kathman et al. support temporally integrated odor evidence and post-odor persistence in VT062617-labelled local neurons, with partial hΔK specificity. Siliciano et al. support vector-like boundary-return memories and task-linked FC2 activity. These are different computations and cell populations; neither supports FC2 as a universal memory for all odor goals. [Kathman et al., 2026](https://www.nature.com/articles/s41467-026-75945-2), [Siliciano et al., 2026](https://pubmed.ncbi.nlm.nih.gov/42486983/)

## Experiment 4: physically ground language-selected goals

### Question

Can language select a stable semantic objective while the host resolves it to different spatial targets as the world changes?

### Design

Place water, food, shelter, and rest locations in a simulator. FunctionGemma selects an allowlisted `goal_id` only when instructed. Host perception resolves the selected goal to a currently available object and bearing. Move objects between episodes, use multiple instances, remove requested resources, and switch goals mid-episode.

### Controls and falsification

- Oracle semantic goal with the same grounding and controller.
- Keyword parser.
- Correct language goal with shuffled target bearing.
- Flat per-tick language action generation.
- Hierarchical language goal plus conventional controller.

Reward is contact with the correct physical resource under the relevant environment state. Penalize collision, unreachable-object selection, elapsed time, and energy. Never reward agreement between an internal action label and the requested word.

The hypothesis fails if performance does not follow object relocation, if paraphrases change trajectories after controlling for the parsed goal, or if the system claims success when the target is absent. Report parse accuracy, grounding accuracy conditional on the parse, controller success conditional on grounding, and end-to-end success.

Biological evidence: heading and angular-goal signals can be separated and compared. Software analogy: semantic symbol → host-resolved target → angular controller. Fly experiments do not show semantic goals or language grounding. [Mussells Pires et al., 2024](https://www.nature.com/articles/s41586-023-07006-3), [Westeinde et al., 2024](https://www.nature.com/articles/s41586-024-07039-2)

## Experiment 5: causal comparator and descending-control tests

### Question

Do separate signed steering and large-error gain components explain recovery from angular perturbations?

### Design

Use an engineered comparator with a heading estimate, goal bearing, signed turn output, and an independently parameterized large-error gain. Apply 30°, 90°, and near-180° perturbations while matching disturbances across systems.

### Causal controls and falsification

- Swap left/right output.
- Remove signed comparison.
- Remove large-error gain.
- Clamp heading while changing goal, and clamp goal while rotating heading.
- Replace the mechanism with a conventional wrapped proportional controller.
- Add actuator delay and saturation.

Predictions must be registered before running: left/right swapping should reverse correction; removing signed comparison should abolish directional recovery; removing large-error gain should selectively impair large perturbations. Measure physical target acquisition, integrated angular error, overshoot, collision, energy, and recovery time. Failure of these intervention signatures falsifies the proposed decomposition even if average reward remains high.

Evidence: PFL2/PFL3 perturbation and recording results motivate the decomposition; descending-neuron work motivates a separate motor layer. Analogy: the software variables are not neuron populations, and matching a response curve is not evidence of anatomical implementation. [Westeinde et al., 2024](https://www.nature.com/articles/s41586-024-07039-2), [Rayshubskiy et al., 2025](https://elifesciences.org/articles/102230)

## Experiment 6: hierarchy versus flat language control

### Question

Does keeping language at the goal/skill timescale improve robustness and efficiency without hiding failures in the lower controller?

### Design

Compare: (a) typed high-level goal plus host grounding and conventional control; (b) high-level goal plus a small intermediate skill vocabulary; (c) flat language generation at every control tick; and (d) an oracle high-level goal. Match sensor access, action bounds, task distribution, and compute budget where possible.

Evaluate held-out paraphrases, distractor instructions, target relocation, cue dropout, delayed observations, and goal switches. Include malformed and semantically wrong but schema-valid calls.

### Outcomes and falsification

Use physical completion, collisions, energy, elapsed time, intervention count, latency, and recovery from wrong high-level decisions. Report outcomes conditional on correct parsing and correct grounding as well as end-to-end outcomes. The hierarchy claim fails if gains disappear after matching the low-level controller and observation privileges, or if slower language calls create unacceptable recovery latency.

SayCan motivates scoring high-level proposals with grounded affordances. RT-H motivates an intermediate language-motion layer. Neither result implies that the current four-tool interface is sufficient, nor that tool-call accuracy predicts physical success. [Ahn et al., 2022](https://proceedings.mlr.press/v205/ichter23a/ichter23a.pdf), [Belkhale et al., 2024](https://www.roboticsproceedings.org/rss20/p049.html)

## Architecture roadmap

### Stage A: hard trust boundary

- Model-callable tools accept only semantic goals and deliberate high-level skills.
- Sensor observations, timestamps, odometry, cue availability, target contact, and reward enter through host-only interfaces.
- Every action and outcome carries episode, step, observation, and goal revisions.
- Validation and authorization complete before state mutation.

### Stage B: non-oracle estimator

- Maintain circular heading uncertainty, not only a point angle.
- Learn cue quality from history and independent anchors.
- Log raw observations, residuals, inferred trust, and future prediction error.
- Keep oracle reliability as an analysis-only ceiling.

### Stage C: typed grounding

- Resolve semantic goals against current world state.
- Return explicit absent, ambiguous, and unreachable states.
- Preserve provenance from instruction through selected object and bearing.

### Stage D: bounded control

- Compare the engineered heading/goal mechanism with ordinary control.
- Keep actuator dynamics and descending-command decoding separate.
- Run preregistered causal ablations rather than relying only on task reward.

### Stage E: physical outcome learning

- Reward verified contact, acquisition, avoidance, or task completion.
- Test reversal and context changes with held-out environments.
- Preserve old-policy probes because behavioral extinction need not mean erasure.
- Compare global reward learning with compartmental or multi-trace alternatives before making MB analogies.

### Stage F: evidence discipline

- Version datasets, environments, splits, seeds, checkpoints, adapters, and evaluation code.
- Separate syntax validity, semantic correctness, grounding correctness, control quality, and physical success.
- Treat anatomical connectivity as candidate structure and perturbation/behavior as functional evidence.
- Use “source-inspired engineered mechanism” until dynamics and causal signatures are matched prospectively.

## Primary sources

1. Kim et al. (2017), ring-attractor dynamics in the Drosophila central brain.
2. Fisher et al. (2019), sensorimotor remapping of visual input to heading.
3. Basnak et al. (2025), multimodal cue integration and learning.
4. Mussells Pires et al. (2024), allocentric goal-to-egocentric steering transformation.
5. Westeinde et al. (2024), PFL2/PFL3 transformation of heading into goal-oriented commands.
6. Kathman et al. (2026), odor evidence integration and working memory in VT062617/hΔK-related FB neurons.
7. Siliciano et al. (2026), vector-based olfactory edge tracking and task-linked FC2 goal memory.
8. Aso et al. (2014), MBON valence and memory-guided action selection.
9. Felsenberg et al. (2018), parallel opposing memories in extinction.
10. Namiki et al. (2018), organization of descending sensorimotor pathways.
11. Rayshubskiy et al. (2025), descending steering control during walking.
12. Ahn et al. (2022), SayCan language planning grounded by affordances.
13. Belkhale et al. (2024), RT-H language action hierarchies.
