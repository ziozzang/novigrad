# Continuous recall, one-shot signals, and signal strength

## The distinctions the software experiment must preserve

Four different operations can produce behavior after a cue, and they must not be called by one name.

1. **Maintained sensory input:** the cue is still present. A response can persist because input continues; no memory is required.
2. **Working memory after cue offset:** an internal state persists or is updated after the cue disappears. This requires recurrent state, integration, or an external state store.
3. **Associative recall:** a cue presented now reads out a value learned earlier. The sensory response may be transient even though the stored association is durable.
4. **One-shot training:** one cue–outcome episode changes later behavior. This describes acquisition count, not the duration of sensory input or the mechanism of recall.

The current frozen-embedding → feedforward `novi` evaluation has no autonomous temporal state between calls. Repeating the same vector merely recomputes an output. Any continuity comes from repeated host input, plastic weights changed by an update, or explicit host state. It is not working memory.

## Primary evidence and bounded software predictions

| Primary study | What was directly shown | Limit and falsifiable software prediction |
|---|---|---|
| [Seelig & Jayaraman, 2015](https://www.nature.com/articles/nature14446) | In tethered walking flies, an ellipsoid-body activity bump tracked a visual landmark and continued to track heading using self-motion in darkness. | This is heading-state dynamics, not mushroom-body associative recall or language memory. A true persistence test must remove the cue, provide controlled odometry, and compare a recurrent/stateful estimator with a reset feedforward control. Without post-offset state, prediction should return to prior/chance. |
| [Neuser et al., 2008](https://www.nature.com/articles/nature07003) | Flies retained a briefly viewed visual orientation; ellipsoid-body ring-neuron function and plasticity were causally implicated. | The behavioral assay establishes short spatial orientation memory under its protocol, not a general-purpose working memory. Software should vary cue-off delay and include reset, distractor, and motor-only controls; accuracy must be plotted against delay rather than described as simply “persistent.” |
| [Honegger et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869/) | KC population responses stayed sparse over tested odor concentrations, but stimulus order mattered: first presentations were broader and prior strong stimulation affected later responses. | Signal strength and presentation history are confounded if intensity is swept in a fixed order. Cross intensity with randomized order, exact repetition, and interstimulus interval. A static embedding pipeline predicts no history effect for identical vectors; any effect requires an explicit adaptation state. |
| [Lüdke et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5960692/) | Calcium imaging found odor-specific post-offset activity; among measured locations, KC-soma calcium patterns retained enough information to decode the preceding odor at 15–16 s. The authors proposed this as a possible substrate for short sensory memory. | Decodability is correlational and does not establish that KC-soma calcium causes behavioral recall; unmeasured signals may contribute. Software should separately log state at the input, hidden, and output stages after cue removal and causally reset each. A host-copied vector is an engineered trace, not evidence for a KC analogue. |
| [Masek & Heisenberg, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2556361/) | Behavioral tests supported separable memory for odor quality and intensity; intensity memory was shorter-lived and generalization depended on concentration range. | Text-vector norm must not be assumed to equal biological odor intensity, especially after normalization. Encode semantic identity and an independently controlled reliability/amplitude channel, then test a full identity × amplitude matrix. Reject factorization if changing amplitude systematically changes inferred identity. |
| [Hattori et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5806120/) | Repeated odor presentations suppressed a specific α′3 MBON response through dopamine-dependent plasticity; KC activity decreased much less and other MB compartments did not show the same strong suppression. | Repetition suppression is neither generic sensory fatigue nor a global memory-strength variable. Compare an early adaptation state, an odor/request-specific familiarity state, both, and neither. Recovery after a gap and stimulus specificity distinguish the hypotheses. A stateless engine should show identical output on exact repeats before learning. |
| [Hige et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/) | Pairing odor with activation of a defined dopamine input could induce odor-specific synaptic depression and learned behavior with a narrow temporal-order dependence. | This supports temporally specific associative plasticity in defined compartments, not a universal “one prompt is enough” rule. Run exactly one validated action–outcome update, then test held-out paraphrases before any second update; compare unpaired, reversed-order, and shuffled-label controls. |
| [Tully et al., 1994](https://pubmed.ncbi.nlm.nih.gov/7923375/) | Extended aversive training produced genetically and functionally distinguishable consolidated components; spaced and massed protocols did not create the same memory form. | Equal update counts do not make schedules equivalent. Compare one-shot, massed, and spaced updates with the same total examples and reward magnitude, followed by no-update retention probes. A feedforward policy with static weights cannot exhibit time-dependent consolidation unless such dynamics are explicitly implemented. |
| [Felsenberg et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5392358/) | Re-presenting a learned appetitive cue without the expected reward could engage extinction or reconsolidation-related circuitry depending on prediction conditions. One or repeated retrieval presentations could alter later expression. | Repeated evaluation is not neutral if evaluation itself supplies omissions or learning updates. Separate read-only probes from cue-with-outcome episodes. Compare no probe, read-only probe, expected outcome, and omitted outcome; freeze weights during pure measurement. |
| [Yang et al., 2026](https://www.nature.com/articles/s41593-026-02381-2) | After single-trial aversive training had become behaviorally undetectable, repeated reminders in matching contexts recovered avoidance and reinstated an active MBON trace; altered reminders could produce avoidance of the originally unpaired odor. Texture/light changes blocked recovery in tested conditions, while temperature change did not. | This shows reconstructive, context-dependent recovery in this fly protocol, not that every low behavioral score hides a memory. Software must preserve an acquisition checkpoint, verify that ordinary readout has fallen, and compare paired, unpaired, novel, and changed-context reminders. A reminder that updates weights is retraining, not passive recall; false-memory language requires a demonstrable original episode and misattribution control. |

## Proposed software experiments

### 1. Cue-on versus cue-off persistence

Use sequences with a cue present for one step and absent for 1, 2, 4, 8, and 16 steps. Compare: stateless feedforward, host-held last observation, a bounded recurrent estimator, and an oracle state. During the gap, include no motion, known odometry, and misleading odometry. Report post-offset accuracy and state decay separately from cue-on accuracy. The expected stateless result is no genuine retention; repeating the cue tests sustained input, not recall.

### 2. One update versus continuous reinforcement

Hold total evaluation examples fixed. Train with one outcome, repeated massed outcomes, or the same number spaced by unrelated trials. Use disjoint paraphrase families for acquisition and recall. Evaluate immediately and after intervening no-update traffic. Match reward sum and optimizer exposure, and include shuffled outcome and outcome-before-action controls. A one-shot claim requires improvement after exactly one causal update relative to all controls, not merely one minibatch containing repeated copies.

### 3. Retrieval without accidental retraining

Create immutable checkpoints after acquisition. Branch each into no re-exposure, read-only cue probes, cue plus expected outcome, and cue plus omitted outcome. Record weights before and after every probe. If read-only inference changes behavior or weights, the implementation is contaminated. If omission changes later behavior, call it software updating; do not infer extinction or erasure without context renewal/reinstatement tests.

### 4. Strength, reliability, and identity factorial

Keep the normalized frozen semantic vector fixed while independently varying a host-supplied amplitude/reliability field. Also create semantic competitors at fixed amplitude. Cross four identities, several amplitudes, cue duration, and randomized presentation order. Controls should include raw-vector norm, normalized vector, amplitude-only input, and shuffled amplitude. The desired invariant is stable identity over a declared amplitude range with confidence or abstention responding to reliability. Do not choose thresholds on the test matrix.

### 5. Adaptation versus persistent recall

Present exact repeats, semantic paraphrases, and unrelated distractors at matched intervals. An early adaptation model should depend on recent physical/input-channel history; a familiarity model should be stimulus-specific; associative recall should depend on prior outcome; working memory should be measurable after cue removal. Cross these factors rather than treating any falling response as forgetting. Reset state between sequences to detect leakage.

### 6. Silent-trace reminder challenge

After one update, retain an immutable checkpoint and weaken only the readout or allow a predeclared decay until behavior reaches the control range. Apply repeated reminders that match the trained cue/context, use the originally unpaired cue, use a novel cue, or change context. Compare inference-only reminders with reminders that permit updates. Recovery from inference-only probes requires stored state that the current stateless feedforward engine does not have; recovery after updates is relearning or reconstruction in the engineered system. Report both to prevent a retraining effect from being mislabeled as continuous recall.

## Confounds and acceptance rules

- Randomize intensity order. Honegger et al. show that a fixed high-to-low or low-to-high sequence can change the apparent concentration effect.
- Keep inference probes read-only. Outcome omission is an event only when the environment explicitly delivers it and the learner updates from it.
- Treat repeated text as repeated input, not elapsed biological stimulation. Define logical time and duration explicitly.
- Separate vector direction, vector norm, host reliability, reward magnitude, and number of updates. Normalization erases norm as a strength variable.
- Split paraphrase families and translations together. Many trials from one checkpoint and the same semantic template are not independent replicates.
- Predeclare primary contrasts and summarize seed-level effects. A post hoc best interval, strength, or sparsity is exploratory.
- Evidence for working memory requires above-control performance after cue offset and no truth leakage. Evidence for one-shot learning requires a measurable change caused by one update. Evidence for associative recall requires a stored update plus a later cue. Maintained performance while a cue remains visible establishes none of these by itself.
