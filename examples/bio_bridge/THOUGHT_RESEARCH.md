# Testing latent state in a fly-inspired embedding bridge

## Scope: what “thinking” can mean here

This project cannot extract a fly's private experience, inner language, or subjective thought. A defensible target is an **operational latent variable**: a quantity that is not identical to the current stimulus but predicts behavior, can be decoded from neural activity on held-out trials, and changes behavior when the relevant circuit is perturbed.

Four quantities should remain distinct:

1. **Sensory evidence**: information available in the current observation.
2. **Internal or motivational state**: hunger, satiety, arousal, or another state that changes how the same cue is used.
3. **Decision variable or expected value**: a learned quantity that predicts a coming choice beyond current stimulus identity.
4. **Heading or motor plan**: an allocentric direction estimate or action-related signal.

A decoder can recover any of these from activity without proving that the circuit uses the decoded variable. It can also recover the experimenter's input through a passive encoding path. Cross-validated decoding is therefore a measurement step; causal intervention and behavioral specificity are separate tests.

## Primary evidence and its limits

| Primary study | What was measured or manipulated | What it supports | What it does not support |
|---|---|---|---|
| [Aimon et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6395010/) | Fast near-whole-brain calcium/voltage imaging in behaving adult flies; walking produced broad activity changes, while spatial sources also tracked sensory and behavioral variables. | Neural recordings must be aligned with stimulus and behavior, and global movement state is a major covariate. | A low-dimensional component is not automatically a thought or decision variable. Calcium imaging is slower than spikes and does not by itself establish causality. |
| [DasGupta et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4206523/) | Flies chose between visual patterns of varying difficulty; reaction time and accuracy changed with evidence quality, and FoxP manipulation altered the speed–accuracy relation. | A behavioral signature can motivate an evidence-accumulation model and yield falsifiable psychometric and chronometric predictions. | The behavioral signature does not uniquely identify a neural accumulator, and a language classifier's logits are not evidence that flies use language-like deliberation. |
| [Seelig and Jayaraman (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4704792/) | Two-photon imaging during virtual navigation showed an ellipsoid-body population bump tracking visual heading and updating with self-motion in darkness. | Heading is a concrete latent variable with circular geometry that can be compared with behavior and sensory conditions. | A decodable bump is not a general-purpose belief or semantic intention. Head fixation and virtual reality constrain generalization. |
| [Kim et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28473639/) | Optogenetic perturbation displaced the heading bump; the circuit maintained and naturally evolved the imposed representation. | A perturb-and-recovery experiment can test whether a decoded state participates in circuit dynamics rather than merely correlating with input. | This result is specific to the heading system and does not license a general “thought attractor” claim. |
| [Krashes et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2780032/) | Hunger/satiety and targeted dNPF/dopamine manipulations changed expression of appetitive odor memory. | Internal state can gate retrieval and action from a learned cue; identical cue identity need not imply identical behavior. | Hunger was not decoded as an abstract language vector, and the result does not establish a fly prefrontal cortex or a universal value channel. |
| [Sterne et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10523640/) | Sequential operant choices, reward schedules, MB perturbations, and plasticity models tested reward expectation in matching behavior. | Expected reward should be inferred from choice histories and falsified through contingency changes and circuit intervention. | Model fit alone cannot prove the exact biological learning rule; expected value is task- and history-dependent rather than a semantic label. |
| [Shiu et al. / FlyWire Consortium (2024)](https://www.nature.com/articles/s41586-024-07558-y) | Synapse-resolution reconstruction of an adult female fly brain. | Cell types, candidate paths, directionality, and counted chemical synapses can constrain an intervention target. | A static specimen does not provide activity, membrane dynamics, individual motivational state, learned weights, or subjective content. Connectivity cannot substitute for paired neural and behavioral recordings. |

## A measurement ladder

### 1. Define variables before inspecting activity

For a hunger–benefit–cost task, predeclare:

- observation `o_t`: the currently shown food cue and cost/threat cue;
- state `s_t`: experimentally assigned hunger level or a host-provided state value;
- offer values: benefit `b_t` and cost `c_t` varied independently;
- choice `a_t`: approach, avoid, or another fixed action set;
- outcome `r_t`: physically grounded reward/cost where available;
- nuisance variables: text family, sentence length, movement, elapsed time, trial history, and action frequency.

Do not call a class label “thought.” The scientific targets are conditional predictions such as whether hidden activity represents hunger after controlling for the current cue, or whether an intervention changes cost sensitivity without erasing cue decoding.

### 2. Establish behavioral computation

Fit a held-out choice model before interpreting hidden activity:

```text
P(approach) = sigmoid(theta_0 + theta_b b_t - theta_c c_t
                      + theta_s s_t + theta_bs b_t*s_t
                      + history terms)
```

The interaction term tests whether internal state changes the use of benefit. Independently manipulate benefit, cost, and hunger; otherwise hunger can be decoded simply because it is always paired with food words or one action. Test contingency reversal and state switches to separate current evidence from learned value and perseveration.

### 3. Decode with strict cross-validation

Given hidden activity `h_t`, train simple probes on training trials only:

```text
hunger probe:       s_hat = f_s(h_t)
value probe:        v_hat = f_v(h_t), where v_t is defined from outcomes/choice history
choice probe:       a_hat = f_a(h_t)
heading probe:      (sin(phi_hat), cos(phi_hat)) = f_phi(h_t)
```

Use regularized linear probes first and report held-out balanced accuracy or circular error. Hold out entire flies for biological data and entire semantic paraphrase families for the embedding experiment. Randomly splitting near-duplicate paraphrases permits lexical memorization and is not evidence of an intent representation. Fit every scaler, prototype, ridge coefficient, and threshold inside each training fold.

Required controls are:

- an **input-only probe** on frozen embeddings;
- a **label-shuffled probe** with the same split and fitting procedure;
- matched nuisance probes for length, lexical family, action, and trial time;
- behavior-history and current-stimulus baselines;
- a held-out state switch and a held-out benefit–cost combination;
- multiple seeds reported as repeated optimization/sampling runs, not independent animals.

If the hidden probe performs no better than the input-only probe, the graph has not added evidence for a latent computation. If both decode hunger when hunger words are explicitly present, they only demonstrate encoding of provided input.

### 4. Intervene and test specificity

A candidate state representation becomes mechanistically interesting only if an intervention produces the predeclared behavioral change while preserving relevant controls. For example:

- clamp or swap the candidate hunger component while holding the sensory embedding fixed;
- silence candidate ports or hidden units selected on training data only;
- compare a norm-matched random-unit intervention and an input-only manipulation;
- test whether benefit sensitivity changes while raw cue identity and unrelated choices remain decodable;
- wash out or reverse the intervention and test recovery.

An intervention on units selected by a decoder is circular unless selection, effect-size estimation, and final evaluation use disjoint data. Broad output loss may reflect damage to the classifier rather than removal of a state variable.

## Frozen EmbeddingGemma experiment

The proposed repository experiment is a **semantic embedding bridge simulation**, not a measurement of fly thought.

### Data and conditions

Create language observations that independently cross:

- need: hungry versus satiated;
- benefit: low versus high food value;
- cost: low versus high threat/effort;
- semantic family: disjoint paraphrase templates reserved for train, validation, and test;
- state presentation: explicit state in the sentence versus a host-maintained state omitted from later observations.

Use frozen EmbeddingGemma features, the existing signed port encoding, and frozen Novi input-to-hidden connectivity. Do not fine-tune the embedding model. Existing confirmation cases may be used for a descriptive compatibility check, but they are reused cases and cannot serve as a new generalization result.

### Three readouts with different claims

1. **Embedding prototype matching** compares each frozen input vector with prototypes built from training texts. This is the encoding-only baseline and tests whether the language model already separates the labels.
2. **Hidden ridge probe** predicts state, net value, or intended action from frozen native hidden activity. It tests linear accessibility after the graph transform.
3. **Native policy readout** measures the action selected by the existing engine. It tests whether the current policy uses the information, subject to its prior training and readout compatibility.

Prototype success is semantic similarity, not cognition. Hidden-probe success beyond chance is decodability, not causal use. Native action changes can still arise from direct wording. Only the factorial controls and intervention distinguish these possibilities.

### Falsifiable tests

| Test | Prediction if the intended computation is present | Failure interpretation |
|---|---|---|
| Semantic-family holdout | State/value decoding transfers to unseen paraphrase families | A random-row split may have measured lexical template reuse |
| Explicit versus host-maintained state | Later cue processing remains state-dependent when the state word is absent only if a persistent host state exists | The stateless feedforward engine should lose the omitted variable; this is expected, not biological forgetting |
| Benefit × cost grid | Choice and value probe vary monotonically with both independently manipulated factors | A need-word classifier is not value integration |
| State switch | Choice changes promptly when hunger changes while identical cues and costs are held fixed | Persistent old choice suggests history/perseveration or a faulty host state |
| Decoder-selected intervention | Targeted intervention changes state-dependent benefit weighting more than norm-matched random intervention | Equal disruption indicates generic damage or decoder selection bias |
| Shuffled labels | Probe returns to its permutation null | Above-null shuffled performance signals leakage or an invalid split |

### Identity decoding versus causal use

Run a read-only identity decoder for cue class and semantic goal alongside the intervention experiment. High identity accuracy can coexist with no behavioral use. Conversely, a native action can change even when the separately trained identity probe is stable. Report both:

```text
encoding:      Can input or hidden activity reconstruct the provided label?
behavior:      Does the frozen policy choose differently?
intervention:  Does manipulating a candidate component selectively change behavior?
```

The strongest allowed conclusion from current repository-only work would be: “A frozen language embedding and graph transform made a predeclared state/value label decodable, and a held-out intervention changed a specified policy dependence.” It would still be a software result. Aligning language-vector axes with fly neural state requires simultaneous fly neural recordings, behavior, stimulus and internal-state labels, cross-animal registration, and causal perturbation.

## Confounds that can mimic a latent state

- **Input reconstruction:** the decoder reads the supplied state word rather than a transformed or maintained variable.
- **Movement leakage:** broad activity predicts walking or turning, which correlates with choice; Aimon et al. show why movement regressors are mandatory.
- **Outcome leakage:** post-choice or post-reward activity is used to decode a pre-choice value.
- **Semantic-family leakage:** close paraphrases occur on both sides of a random split.
- **Action imbalance:** always approaching when hungry allows an action decoder to masquerade as a hunger decoder.
- **Trial-history leakage:** adjacent samples from one episode cross folds.
- **Probe flexibility:** a high-capacity decoder extracts incidental information unavailable to the native readout.
- **Selection circularity:** the same trials choose units, tune the probe, and estimate the intervention effect.
- **Connectome overreach:** a path or synapse count is described as a dynamic belief, value, or thought.

## Acceptance criteria for future work

A credible software diagnostic should require all of the following: predeclared labels and contrasts; semantic-family held-out evaluation; input-only and shuffled-label baselines; time-causal features; a hidden probe that adds information or a policy intervention with selective effect; state-switch and benefit–cost tests; and full per-seed results. A biological claim additionally requires paired neural and behavioral data and a causal perturbation in flies. Without those data, use “state decoding in a fly-inspired embedding bridge,” never “extracting a fly's thoughts.”
