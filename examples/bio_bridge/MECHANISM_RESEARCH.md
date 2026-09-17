# Extracting testable mechanisms from the fly olfactory circuit

## What can be extracted

The repository's edge tables can identify which recorded presynaptic and postsynaptic root IDs are connected and can count their synapses. They do not contain receptor kinetics, membrane state, transmitter concentration, adaptation time constants, inhibitory gain, dopamine release, or an eligibility variable. Even a correctly annotated ORN/PN, APL, KC, DAN, or MBON path is structural evidence only.

The equations below are therefore minimal **mechanism-inspired hypotheses**. Their state variables and constants must be declared and tested; they cannot be inferred from static connectome edges. Frozen EmbeddingGemma vectors are engineered language features, not odors. The `novi` graph is a feedforward rate policy unless an experiment explicitly adds host state or synaptic state.

## 1. ORN/PN adaptation and antennal-lobe gain control

### Primary evidence

[Gorur-Shandilya et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5524537/) recorded identified Drosophila ORNs under naturalistic and Gaussian odor streams. ORN gain adapted to stimulus mean and variance; mean-dependent gain was mainly at transduction, while variance-dependent gain appeared at transduction and spike generation. Complementary kinetics helped preserve encounter timing across intensity. This supports dynamic, history-dependent gain rather than a static rescaling rule.

[Olsen et al. (2010)](https://doi.org/10.1016/j.neuron.2010.04.009) independently manipulated feedforward and lateral input in the adult antennal lobe. PN normalization scaled with total ORN population activity, increased the drive required for saturation, and made responses more transient. Their compact model used a channel's feedforward drive and total ORN activity. This supports divisive population normalization, but does not make every form of attenuation divisive.

[Cafaro (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4831330/) measured ORNs and their connected PNs under background and pulse stimulation. PN spike-rate adaptation exceeded ORN adaptation, and different sites contributed to background versus pulse adaptation. A single “sensory decay” state would collapse mechanisms the experiment separated.

### Minimal competing models

Let `x_i(t) ≥ 0` be input in channel `i`, and let `m_i` be a causal running background estimate:

```text
m_i(t+1) = (1-α) m_i(t) + α x_i(t)
```

Three hypotheses should be kept distinct:

```text
Subtractive/high-pass: y_i(t) = relu(x_i(t) - β m_i(t))
Divisive adaptation:   y_i(t) = f(x_i(t)) / (σ + β m_i(t))
Population gain:       p_i(t) = f_i(x_i(t)) / (σ + γ Σ_j f_j(x_j(t)))
```

`α`, `β`, `γ`, and `σ` are engineered constants with no connectome-derived value. `f` may be a fixed saturating front end, but adding it creates another hypothesis.

### Causal assumptions and falsification

- A subtractive model predicts an offset/threshold and can silence a steady weak input. A divisive model predicts ratio-like scaling while preserving sign/order above zero. Fit both on adaptation sequences and test them on held-out means, variances, and pulse amplitudes.
- A population denominator predicts cross-channel suppression: increasing an unrelated channel should reduce the target channel even when its own history is fixed. Channel-local adaptation does not require that effect.
- Randomize high-to-low and low-to-high intensity order. Otherwise history can masquerade as an intensity curve.
- Include step-on, steady background, brief pulse-on-background, offset, and recovery gaps. A pure static normalization must fail the recovery/time-course test if adaptation state is necessary.
- Compare matched-output controls: a static gain chosen to match average response, a shuffled-history state, and a reset-every-tick state. Better average scaling alone is not evidence for feedback.
- Record raw input, state, denominator/subtraction, and output. An improvement after adding state is evidence for that software state under this task, not an ORN or PN mechanism.

The current engine's hidden activity is normalized by its maximum after top-k selection. That produces near invariance to a positive global input scalar, but it is instantaneous and stateless. It does not implement Weber–Fechner adaptation, high-pass filtering, background recovery, or cross-time feedback.

## 2. APL inhibition and sparse KC pattern separation

### Primary evidence

[Lin et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/) activated and blocked both legs of the KC–APL feedback circuit. KCs activated the GABAergic APL neuron; APL inhibited KCs. Acute APL output blockade broadened and correlated KC odor representations and selectively impaired learned discrimination of similar, rather than dissimilar, odor pairs. The authors noted that the manipulation did not formally prove sparsity itself was the sole mediator.

[Amin et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7541083/) used local stimulation and volumetric calcium imaging to test propagation within APL. Their results supported spatially localized APL activity and inhibition, making a single uniform global inhibitory scalar an incomplete biological description.

### Minimal competing models

Let PN-like features drive pre-inhibition KC activity:

```text
u(t) = relu(W_PN→KC x(t))
```

A global feedback proxy is:

```text
a(t) = mean(u(t))                 or Σ_k u_k(t)
z_k(t) = relu(u_k(t) - β a(t))
```

A local proxy replaces one scalar with a fixed neighborhood or compartment matrix:

```text
a_j(t) = Σ_k L_jk u_k(t)
z_j(t) = relu(u_j(t) - β a_j(t))
```

Top-k selection is a third mechanism:

```text
z(t) = keep_k_largest(u(t))
```

Top-k enforces an active count but contains no APL cell, feedback latency, GABA dynamics, or local reciprocity. A shadow-forward APL calculation that does not feed back into policy output is an audit probe, not functional inhibition.

### Causal assumptions and falsification

- Predeclare similar and dissimilar cue pairs using a measure fixed before labels/results. Measure active fraction, pairwise representation correlation/cosine, margin, and held-out discrimination.
- Compare no inhibition, global subtractive/divisive inhibition, local inhibition, and top-k. Match mean norm or active count where possible so an accuracy difference is not merely scale or capacity.
- A pattern-separation prediction is an interaction: inhibition should reduce representation overlap and improve discrimination more for similar than dissimilar pairs. Uniform improvement is insufficient.
- Perturb locality. Shuffle `L` while preserving row sums, exchange local groups, or replace it with a rank-matched global operator. If the local model is superior only because it has more parameters, the anatomical analogy fails.
- Test mixtures and weak diagnostic features beside strong distractors. Excess inhibition may cause detection failures; lower activity is not automatically better separation.
- Hold the plastic readout fixed when measuring representation geometry, then train matched readouts. Otherwise changed learning and changed encoding are confounded.

Actual PN→KC and KC/MB connectivity can constrain which edges exist in `W`. It cannot determine `β`, the state equation, locality of functional voltage/calcium spread, or whether an observed software improvement was caused by an APL-like process.

## 3. Dopamine timing and eligibility

### Primary evidence

[Hige et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/) paired odor with activation of a defined dopamine input and measured odor-specific depression in a mushroom-body compartment. Plasticity depended strongly on relative timing, and dopamine effects were spatially compartmentalized. The result supports heterosynaptic, order-sensitive plasticity in the tested preparation; it does not specify a universal trace kernel.

[Handler et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012144/) showed trial-by-trial behavioral reversal and bidirectional KC–MBON plasticity as odor/reinforcement order changed. DopR1 and DopR2 pathways contributed through distinct second-messenger routes. Receptor expression outside the MB and the defined odor assay limit a purely local, general-purpose interpretation.

### Minimal eligibility hypotheses

For feature/action activity `φ_t`, maintain a causal trace at each eligible plastic parameter:

```text
e_t = λ e_(t-1) + φ(x_t, a_t)
Δw_t = η r_t e_t
```

An order-sensitive signed kernel can instead update from an event pair:

```text
Δw = η r K(Δt) φ,   Δt = t_reward - t_cue
```

where `K(Δt)` may differ in magnitude or sign for reward after versus before cue. `λ`, `η`, and `K` are hypotheses. A per-action ID queue that stores the exact old row and action performs delayed replay. It can solve credit identity, but it is not a decaying synaptic eligibility trace because it preserves a discrete record without local decay or superposition.

### Causal assumptions and falsification

- Sweep signed lag, including outcome-before-cue, simultaneous, and cue-before-outcome conditions. Report the complete preregistered curve; do not select the best lag afterward.
- Compare exact ID replay, exponential eligibility, rectangular eligibility, zero trace, wrong-current credit, shuffled timing, and shuffled action identity.
- Match reward amplitude. With a fixed lag, multiplying by `exp(-lag/τ)` is just a constant smaller reward; an immediate amplitude-matched control is required.
- Use truly sequential sampling and batch-1 updates when claiming an online delay test. If all actions are sampled from one frozen rollout, short delays are hidden inside the sampling phase.
- Store only local features needed by the proposed trace and bound memory. If the system retains full observations indefinitely, success may come from replay memory rather than eligibility.
- Test superposition with two or more pending cues and crossed outcomes. A scalar trace should exhibit predictable interference; an exact ID map can keep them perfectly separate.
- Route updates to declared compartments/heads and compare a shuffled routing control. A name such as “compartment” is not evidence of correspondence to an MB compartment.
- Verify read-only inference cannot update weights and that each reward is consumed once. Measure weight change as well as greedy accuracy, since near-uniform probabilities make argmax unstable.

A convincing software result would identify which mathematical rule is necessary under controlled timing, amplitude, and storage. It would not show dopamine, a molecular eligibility trace, or fly memory without physiological measurement.

## Integrated test matrix

| Mechanism | State required | Primary intervention | Essential negative control | Failure criterion |
|---|---|---|---|---|
| ORN local adaptation | per-channel running mean/variance | change stimulus history with current pulse fixed | reset or shuffled history | no held-out history/recovery effect |
| PN population normalization | instantaneous population drive, optionally slow depression | change other-channel population activity | matched static gain | no cross-channel suppression or better static fit |
| APL global inhibition | population activity scalar | remove/scale feedback | norm/active-count matched transform | no similar-pair-specific decorrelation benefit |
| APL local inhibition | localized activity vector | preserve versus shuffle neighborhoods | row-sum/parameter-matched global operator | locality adds no held-out benefit |
| Eligibility trace | decaying local feature/action state | signed cue–outcome lag | exact replay and amplitude controls | effects explained by replay or reward scale |

The mechanisms should first be tested separately. A combined model is justified only after each state passes its own intervention and negative controls. Otherwise multiple flexible states can fit the same behavior without identifying why it improved.
