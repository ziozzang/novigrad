# A bounded robotics bridge for Novi

This note proposes experiments; it reports no new run, trained language model, or FlyGym result. Novi currently implements a simplified PN→KC→MBON rate circuit derived from real connectivity. It does **not** implement the central complex (CX), descending-neuron (DN) pathways, the ventral nerve cord, muscles, or a biological sensorimotor loop. Those missing systems must not be implied by naming software variables after fly functions.

## What biology measures—and what it does not grant

### Heading and goal heading

In tethered flies, [Seelig and Jayaraman (2015)](https://www.nature.com/articles/nature14446) measured a rotating activity bump in central-complex compass neurons that tracked the animal's heading and updated with visual and self-motion cues. This supports a dynamically maintained heading variable under the tested conditions. It does not make an arbitrary embedding coordinate a compass neuron.

[Mussells Pires et al. (2024)](https://www.nature.com/articles/s41586-023-07006-3) separated a goal-heading representation from current heading and linked FC2/PFL3 circuitry to orientation behavior. The result motivates keeping **desired heading** and **observed heading** as distinct typed variables. The goal was a navigational direction, not unconstrained natural-language intent.

### Learned value, internal state, and action context

[Aso et al. (2014)](https://elifesciences.org/articles/04580) related MBON population activity and causal manipulations to learned positive and negative valence and behavioral choice. This supports a value-sensitive mushroom-body output stage. It does not show that MBONs contain general language semantics or directly specify joint torques.

[Krashes et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2780032/) showed that hunger state gates retrieval/expression of appetitive memory through identified neuromodulatory mechanisms. A robotics analogue may expose internal state as an explicit input and test memory-dependent policy changes. A scalar software variable called hunger is still an engineering abstraction, not a measured hormonal circuit.

[Kim et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6327952/) found cellular evidence for efference copy in fly visuomotor processing: identified visual neurons received motor-related input with timing and sign suited to cancel expected visual motion during saccades. This motivates keeping the expected sensory consequence of a command separate from the residual external visual signal and from the motion actually observed. A command is not an observation: slip, collision, and actuator failure can separate them.

### Proprioception and descending control

[Mamiya et al. (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/) characterized femoral chordotonal organ (FeCO) proprioceptor subclasses selective for tibia position, movement direction and velocity, or vibration. The engineering consequence is to preserve distinct proprioceptive quantities rather than collapse them into a generic body-state scalar, and to supply them separately from motor command with timestamps and uncertainty.

[Namiki et al. (2018)](https://elifesciences.org/articles/34272) mapped descending neurons linking brain regions to the ventral nerve cord, while [Cande et al. (2018)](https://elifesciences.org/articles/34275) used descending-neuron activation to relate individual and combined pathways to behavior. These studies support a many-to-many descending control interface rather than one word per motor. They do not provide a complete inverse map from high-level goal to locomotor command.

Together, these results motivate typed state, memory/value modulation, heading comparison, proprioceptive feedback, and a descending action interface. They do not validate Novi's present PN→KC→MBON circuit as a CX or DN model.

## A demonstrated bidirectional interface precedent

[Flesher et al. (Science, 2021)](https://pubmed.ncbi.nlm.nih.gov/34016775/?dopt=Abstract) combined motor-cortex decoding for a robotic arm with tactile feedback delivered through somatosensory-cortex stimulation. In the studied participant, tactile feedback improved task completion time. The relevant engineering principle is that the readout and feedback-writing channels are calibrated for different roles and evaluated through physical task outcomes. It does not establish a fly–LLM connection or make software prefixes equivalent to neural stimulation. Our proposed software loop must likewise test both action grounding and the information carried back from the environment.

## Three token interfaces

The interface choice determines what must be trained.

1. **Typed text.** Serialize a small schema—goal, observed state, uncertainty, permitted tools—into ordinary text tokens. This works with an unchanged language-model tokenizer and is auditable, but it is verbose and may lose metric precision. Text should describe measurements; it must not silently insert the correct action or future outcome.
2. **Continuous soft prefix, `m × d_model`.** A learned bridge maps a fixed state window to `m` floating-point vectors placed in the model's embedding stream. These vectors are not token IDs, words, or neurons. Perceiver-style latent queries, [Flamingo](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf), and [BLIP-2](https://openreview.net/pdf?id=KU9UojoX7U) demonstrate ways to compress non-text inputs into a frozen or partly frozen language model, but paired objectives are still needed to make the vectors useful.
3. **Learned discrete tokens.** Quantize state or action chunks and extend/reuse token IDs. [RT-2](https://arxiv.org/abs/2307.15818) and [OpenVLA](https://openvla.github.io/) train vision-language-action models on robot trajectories; [FAST](https://huggingface.co/physical-intelligence/fast/blob/main/README.md) learns a tokenizer for continuous action sequences. Assigning IDs alone gives them no grounded meaning. The model must be trained or adapted to consume and/or emit those codes, and tokenizer collisions must be excluded.

A shared dimension or embedding table is only a transport convention. It is not alignment. An embedding encoder is a non-generative semantic feature extractor: its output can summarize text similarity, but arbitrary physical measurements do not become semantic merely by projection into that width.

## Proposed two-clock architecture

Use a **fast motor loop** for stabilization and a **slow semantic loop** for goals, mode changes, and recovery decisions.

At roughly 20–100 Hz, a specialist controller consumes pose, contact, local sensory features, and the current bounded goal; it emits joint targets or a compact motor primitive. At roughly 0.5–2 Hz, or on an event, the language path reads an aggregated state window and selects a typed command such as `set_goal_heading`, `approach_odor`, `stop`, or `request_recovery`. A safety layer validates range, expiry, and permitted tools before committing the command. The precise rates are experiment parameters, not biological claims.

The read and write bridges are independent maps:

```text
observations -> state encoder R -> text / soft prefix / token codes -> LM
LM tool decision -> parser and guard -> action-grounding map W -> specialist controller
                                      ^                              |
                                      |------ observed outcome ------|
```

`W` should not be defined as `Rᵀ` or assumed to invert `R`. Reading asks which latent state predicts a semantic decision and therefore needs calibration or paired alignment evidence. Writing asks which grounded command changes the body safely; its grounding may come from a measured analytic controller or a fitted model. Both paths need independently verified mappings, but neither necessarily requires learned dynamics. Suitable objectives and checks include task success, valid-command loss, representation alignment on paired observations, and prediction of a future observed state. Cycle consistency may be auxiliary, but two expressive networks can hide information and reconstruct each other without learning physical meaning. Causal tests must intervene on goal, observation, action availability, or feedback while holding matched alternatives fixed.

A typed message should distinguish actual observation from intention:

```json
{
  "seq": 1842,
  "frame": "arena/world",
  "time_s": 27.440,
  "valid_until_s": 27.940,
  "goal": {"type": "heading", "rad": 1.20},
  "observed": {"heading_rad": 0.83, "sigma_rad": 0.12, "odor": 0.41},
  "commanded": {"turn_rate_rad_s": 0.30, "issued_at_s": 27.420},
  "ack": {"command_seq": 1841, "status": "accepted"}
}
```

Sequence numbers prevent stale acknowledgements; frame and units prevent silent coordinate errors; expiry bounds slow-model latency. Preserve distributions, intervals, or top-k hypotheses through the bridge where possible. Taking an early argmax destroys ambiguity and can make a confident downstream decision look better calibrated than its evidence.

## Two distinct write destinations

A tool call that controls a robot and a signal that perturbs a neural population are different write targets. Both can be supported, but they require separate calibration.

**Semantic-to-circuit input.** An instruction embedding can be mapped to a nonnegative pattern over verified Novi input ports, with explicit gain, duration and onset. The existing PortBridge already provides an engineered embedding-to-PN representation; its use of text embeddings does not establish a natural fly sensory code. A future goal-heading population could be encoded as a circular activity profile, but the current PN→KC→MBON model has no FC2/PFL3 dynamics in which to install such a profile. Implement and validate that module separately rather than renaming an arbitrary projection as FC2 stimulation.

**Circuit-output or model-intent to movement.** A readout from simulated populations, or a schema-valid LM intent, can drive an experimentally calibrated motor basis. For a proposed neural write port, define `u = B a`, where columns of `B` are allowed spatial patterns and `a` is a low-dimensional amplitude vector; a temporal envelope adds duration. Constrain amplitudes, signs and timing to the actual simulator/API contract. These are software interventions, not animal stimulation prescriptions. Calibrate the forward effect `(state, u) -> next observation/behavior` first. Only then fit a write policy for desired task changes. Reading a state correctly does not imply that injecting its reconstruction creates that state.

Locally perturb each basis coefficient while keeping initial state and randomness paired. Estimate an effect matrix from changes in measured task variables, inspect its rank and conditioning, and identify unreachable directions or ambiguous controls. A large embedding cannot recover actions absent from the actuator's reachable set—the earlier Novi actuator-map result illustrates that engineering problem. Test off-target effects, nonlinear dose responses and recovery after removing input; a local effect matrix is not a global controllability proof.

Read calibration and write calibration must use independently evaluated outcomes. A read/write cycle that reproduces its own latent can hide an arbitrary code without changing the body correctly. Include shuffled stimulation maps, equal-amplitude ineffective inputs, actuator-disabled trials, and a specialist controller receiving identical observations. Keep environment reward separate from a language-model reward prediction: a model-generated statement of success must not become the training reward or the measured outcome.

## Bounded experiments and predictions

All scenarios should first use a rule-based or learned specialist controller that works without an LM. The language interface changes only a declared goal or mode.

* **Odor–wind anemotaxis:** cross odor presence, wind direction, turbulence, and plume loss. Predict that temporal state beats latest-frame input after plume loss; compare oracle plume vector, hand-coded cast/surge, linear/MLP state policy, structured text, and soft prefix.
* **Goal heading in darkness:** remove visual landmarks after establishing orientation, then introduce controlled self-motion bias. Predict increasing uncertainty and graceful performance loss rather than a fabricated exact heading. Compare oracle heading, path integration, frozen last heading, and shuffled goal.
* **Internal-state memory gating:** cross remembered odor value with hunger-like state supplied independently. Predict a state-dependent choice only when memory and state jointly warrant it. Include memory-shuffled, state-shuffled, state-only, and label-oracle controls.
* **Proprioceptive perturbation:** offset or delay observed leg state while keeping issued commands unchanged, and separately perturb the actuator. Predict different signatures for sensing versus execution faults. Compare command-only, observation-only, matched-delay, and no-perturbation models.
* **Reversal and reward omission:** reverse one goal–outcome relation after acquisition and sometimes omit expected reward. Score trial-wise adaptation, perseveration, and recovery. A host Bayesian update, supervised retraining, and fixed policy are separate baselines; none should be called dopamine learning without a measured mechanism.
* **Tool–body combinatorial generalization:** hold out combinations of semantic tool, body side, terrain, and language paraphrase. Predict that typed composition helps only if action grounding is independently learned. Compare flat class IDs, compositional schema, text-only state, and an oracle action.

Use fixed episode seeds and splits by entire scene/trajectory, not overlapping windows. Report success, time, falls/collisions, energy or distance, invalid/stale commands, intervention count, deadline misses, stop latency, recovery after perturbation, and calibration. Report slow-loop semantic accuracy separately from fast-loop control quality. Essential causal controls are zero/random/shuffled state, permuted state-to-case pairing, swapped goals, unavailable-action masks, delayed feedback, replayed identical observations, and equal-capacity non-connectome encoders. A valid tool call is not evidence of motor competence; closed-loop improvement alone does not identify whether the LM, host algorithm, or specialist controller caused it.

## Relation to current robot foundation models

RT-2 and OpenVLA discretize robot actions into language-model-style outputs and train on robot data. Their success is evidence for co-training a grounded interface, not for zero-shot motor meaning in a general LM. FAST improves sequence compression for such discrete autoregressive action generation. In contrast, [π0](https://www.physicalintelligence.company/download/pi0.pdf) uses a separate continuous action expert conditioned by a vision-language backbone. This is especially relevant to Novi: semantic reasoning and fast continuous control can share context without sharing an output clock.

Neural-data models such as [NDT2](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html) and [POYO-1](https://poyo-brain.github.io/) provide useful precedents for timestamped events, heterogeneous sessions, and contextual adaptation. They do not show that neural activity is aligned to an LLM, nor that a decoder trained across recording sessions supplies causal motor commands.

## FlyGym feasibility and release discipline

[FlyGym 2.x](https://neuromechfly.org/installation/) provides an Apache-2.0 simulation environment around NeuroMechFly and supports CPU execution. Its optional NVIDIA Warp path is not the Mac route. The current repository environment uses Python 3.11, whereas the current FlyGym 2.x installation guidance targets Python 3.12; use an isolated pinned environment rather than changing the repository interpreter. FlyGym 2.x also has API changes relative to paper-era 1.x releases, so pin the exact package/version, MuJoCo version, timestep, arena, controller parameters, assets, and reset seeds.

The first readiness check should be a CPU smoke test with a native rule controller: deterministic reset, several hundred stable simulation steps, valid observations/actions, and identical replay under the pinned environment. A heading-token or soft-prefix toy test may follow, but it is still an interface test. Do not claim LM training, embodied success, or a biological motor circuit until separate locked experiments measure them.
