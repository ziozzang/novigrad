# Architectural candidates for an embedding-to-circuit bridge

## Scope and claim boundary

The current bridge preserves useful task categories but loses much of the exact description identity and produces weakly separated action probabilities. This motivates a new architecture comparison; it does not show that the connectome has acquired language semantics. The three candidates below are software hypotheses constrained by selected circuit findings.

No new measurement is reported here. A direct path from the input embedding to an output is **semantic transport**, not neural decoding. A recurrent software state is not a fly memory trace. Contrastive alignment to a fly becomes a biological alignment only when it is fitted to paired stimulus, neural activity, behavior, and state data from real animals.

## Primary evidence and limits

| Primary study | Result relevant to the design | Limit on the software analogy |
|---|---|---|
| [Aso et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4273436/) | Mushroom-body output neurons form parallel, compartmental channels; manipulating selected MBONs biased learned approach or avoidance. | This supports separating task/value channels. It does not support a language-embedding residual or identify a universal semantic bottleneck. |
| [Dolan et al. (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226615/) | Identified lateral-horn neurons receive both hardwired PN input and learned MBON input and are required for parts of innate attraction and memory retrieval. | Converging pathways motivate a dual-path comparison, but a direct embedding bypass is not equivalent to either measured fly pathway. |
| [Kim et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28473639/) | Perturbing the central-complex heading bump displaced an internal heading estimate that the circuit then maintained and updated. | This is evidence for a recurrent state in a specific navigation circuit, not for a general semantic belief state. |
| [Green et al. (2019)](https://www.nature.com/articles/s41593-019-0444-x) | Heading activity and an internal goal predicted sustained oriented walking; perturbation and behavioral analyses connected the heading estimate to navigation. | Goal-relative navigation does not establish arbitrary language-conditioned routing. |
| [Westeinde et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10881393/) | FC2/PFL pathways transform an allocentric goal and heading signals into steering-related activity using measured nonlinear interactions. | This supports testing goal-dependent signal use. It does not imply that every task should use FC2/PFL identities or a generic multiplicative gate. |
| [Aimon et al. (2019)](https://doi.org/10.1371/journal.pbio.2006732) | Near-whole-brain recordings in behaving flies exposed broad movement-related activity alongside localized sensory and behavioral correlates. | Any real alignment needs simultaneous covariates and behavior; a static connectome simulation cannot substitute for recordings. |
| [Schneider, Lee, and Mathis (2023)](https://www.nature.com/articles/s41586-023-06031-6) | CEBRA learned contrastive neural embeddings using paired time and/or behavioral variables and tested consistency across recordings. | It supplies an alignment method, not evidence that frozen language vectors already match fly neural activity. Pairing and split design determine the claim. |

Static connectome edges constrain possible paths and synapse counts. They do not provide neuronal dynamics, learned weights, context, or the observation-to-neural correspondence needed to select among these architectures.

## A separate adaptation axis: bridge rank versus model fine-tuning

Two uses of “low rank” must not be conflated.

**Bridge adaptation** keeps EmbeddingGemma frozen and learns a matrix at the embedding-to-port boundary. For a base map `W0`, a controlled family is

```text
W = W0 + A B,    A in R^(319 x r), B in R^(r x 768)
r in {4, 16, 64}, plus an explicitly named unrestricted-W condition
```

The unrestricted condition is “full-rank bridge fitting,” not full-model fine-tuning. With only 32 training descriptions, a centered empirical design matrix has rank at most 31. Rank 64 and unrestricted bridge parameters therefore cannot be supported by 64 independent directions from those samples; they mainly increase parameterization and interpolation freedom. Effective update rank, singular values, and generalization by held-out semantic family must be reported. The higher-rank conditions are stress tests, not presumed improvements.

**Embedding-model adaptation** changes parameters inside the 308M-parameter EmbeddingGemma encoder. [Hu et al. (2021)](https://arxiv.org/abs/2106.09685) introduced LoRA, which freezes a pretrained weight and learns a low-rank additive update. [Liu et al. (2024)](https://openreview.net/forum?id=3d5CIRG1n2) proposed DoRA, separating weight magnitude from direction and applying a low-rank update to direction. Full fine-tuning updates all selected model weights. Results from those papers do not guarantee that LoRA, DoRA, or full tuning will help this small multilingual bridge.

[Google's official EmbeddingGemma guide](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers) documents Sentence Transformers fine-tuning and says the notebook can run on CPU or GPU. The [official overview](https://ai.google.dev/gemma/docs/embeddinggemma) describes 308M parameters, 768-to-128 MRL output sizes, and sub-200 MB **quantized inference** memory. That inference figure is not a training-memory estimate. The official example reports CUDA use and does not provide an Apple-Silicon/macOS peak-memory or throughput guarantee. Mac feasibility must therefore be established by a small MPS/CPU compatibility and peak-memory probe for the exact dependency and model revisions before scheduling training.

The present 32 texts are inadequate evidence for adapting a 308M encoder. They contain too few independent semantic families, and repeated paraphrases or translations can make apparent sample count much larger than information count. If model adaptation is explored, keep it a separate experiment with many additional train-only pairs, untouched families and languages, and a frozen-base comparison. Start with LoRA rank 4; increase rank only by development results. DoRA and full tuning are resource and overfitting stress tests, not required rungs. No such model fine-tuning has been performed in this architecture round.

## Candidate 1: task bottleneck plus residual semantic transport

Use two explicitly named paths:

```text
e                 = frozen sentence embedding
p                 = fitted 319-port input
z_task             = circuit_task_summary(native_circuit(p))
z_residual         = R(e)                    # direct transport
semantic_output    = D([z_task, z_residual])
action_output      = policy(z_task)          # residual excluded by default
```

The task path asks whether the fixed topology supports a compact action-relevant representation. The residual path preserves description details that a task bottleneck is expected to discard. This is a reasonable engineering decomposition, analogous only at a broad level to interacting learned and hardwired pathways. If `R(e)` bypasses the native graph, reconstruction attributable to it must never be called neural decoding.

Required ablations are task-only, residual-only, fused, shuffled-residual, frozen random projection, and a parameter-matched direct MLP/ridge from `e`. Report exact-description retrieval and task accuracy separately. Measure how much a trained fusion decoder relies on each path by zeroing one path at evaluation. Do not let the residual reach the action policy unless that is a separately declared experiment.

This candidate is the fastest way to retain identity. Its central failure mode is trivial copying: high identity with no added circuit computation. It should therefore be treated as an engineering upper bound and strong control rather than the main neuroscience result.

## Candidate 2: goal-conditioned routing with bounded recurrent state

Test computation that cannot be solved from the current sentence alone. One minimal state model is:

```text
g_t = sigmoid(W_g [context_t, h_(t-1)] + b_g)
x_t = g_t * circuit_features(port_transform(e_t))
u_t = sigmoid(W_u [x_t, h_(t-1)] + b_u)
h_t = clip((1-u_t) * h_(t-1) + u_t * tanh(W_x x_t + W_h h_(t-1)), -1, 1)
action_t = softmax(W_a [x_t, h_t, goal_t])
```

The equations are an engineered gated state, not extracted FC2, PFL, or ring-attractor dynamics. State is reset between episodes, has a fixed dimension and finite numeric range, and receives only observations available at that time. A host-provided goal is labeled as provided context; an inferred belief is evaluated only when the target variable is hidden and must be estimated from observation history.

The decisive dataset must contain identical current observations whose correct action differs with goal or history. Suggested episodes include cue dropout with odometry, delayed goal switches, conflicting cues with reliability observable from past errors, and distractor intervals. Feedback must arise from environment outcomes, not from hidden truth passed into the state update.

Required controls are no context, shuffled context, additive conditioning, multiplicative conditioning, state reset every step, frozen/random recurrence, feed-forward history windows with matched capacity, and an analytic task-state baseline. Audit tensors to verify that future observations, correct actions, and simulator truth never enter before prediction. Compare performance immediately after switches, during dropout, and after distractors, not only aggregate accuracy.

This is the strongest candidate that can be tested now for a substantive architectural change. It makes a falsifiable claim: bounded state and goal routing should help only where history or goal changes the action, and their advantage should disappear on matched stateless trials. Failure on the temporal controls rejects the mechanism even if semantic reconstruction improves.

## Candidate 3: predictive or contrastive cross-modal alignment

With real paired data, learn separate language/stimulus and neural encoders:

```text
q_t = f_language(description_t, measured_stimulus_t)
k_t = f_neural(recording_t)
L_pair = -log exp(sim(q_t, k_t)/temperature)
               / sum_j exp(sim(q_t, k_j)/temperature)
L_pred = distance(P(k_(<=t)), k_(t+1))
L = L_pair + lambda * L_pred
```

Positive pairs must share a recorded trial and aligned time window. Negatives should be balanced across stimulus, behavior, session, animal, and movement so the model cannot identify camera, fly, or batch. Hold out animals, sessions, stimulus families, and time blocks. Compare against ridge, CCA/Procrustes, time-only embeddings, trial-ID baselines, time-shifted pairs, and shuffled pairs. Fit all preprocessing within the training split.

The project currently lacks paired fly recordings, so synthetic circuit rates cannot validate this claim. A simulator can test file formats and leakage guards only. Until real data exist, candidate 3 should remain a data contract and preregistered protocol rather than a scored biological result.

## Recommended experiment order

1. **Primary test: candidate 2.** Build authored sequential episodes where observation, goal, and history are independently varied. Freeze the language encoder and native topology. Select state size, gate form, and regularization on development families only.
2. **Engineering ceiling: candidate 1.** Run it on the same splits to measure recoverable identity. Label every residual-only and fusion result as semantic transport. It is a useful product option if identity matters, but it cannot establish circuit decoding.
3. **Deferred biological alignment: candidate 3.** Define a versioned paired-data schema now; train it only after acquiring real stimulus–neural–behavior recordings with animal/session identifiers.

For candidate 2, use a full factorial evaluation: seen versus held-out semantic family, same versus switched goal, current cue versus dropout, and clean versus distractor. Predeclare a primary temporal metric and report family-level results. Sampling seeds are repeated software runs, not independent biological replicates.

## Strong baselines and acceptance rules

Every architecture should face these baselines on identical splits and parameter accounting:

- raw frozen-embedding nearest prototype and regularized linear models;
- the locked PCA bridge and native policy;
- a parameter-matched direct recurrent model with no connectome layer;
- fixed history concatenation and an analytic state estimator where available;
- shuffled labels, shuffled episode order, shuffled context, and time-shifted feedback;
- residual-only and circuit-only paths;
- calibration metrics and action margins in addition to greedy accuracy;
- exact identity retrieval, semantic-family accuracy, and task reward reported separately.

A candidate is kept only if its preregistered temporal or interaction benefit repeats across seeds and held-out families, beats the capacity-matched direct baseline or reveals a reproducible tradeoff, and passes serialization/replay plus no-truth-leak tests. A higher aggregate score caused solely by direct residual copying is an engineering result, not a mechanism result. Candidate 3 additionally requires held-out animals or sessions and paired biological data.

### Adversarial falsification set

Splits must group every paraphrase, translation, template variant, and counterfactual of one underlying scenario. Fit PCA, port maps, gates, adapters, thresholds, prototypes, and calibrators on training data only. Search logs and hashes must show that final examples were not used for prompt writing, vocabulary selection, hard-negative mining, early stopping, or architecture choice.

The final set should include minimal pairs whose surface similarity conflicts with the required action:

- negation: “I need water” versus “I do not need water”;
- scope: “not warm but thirsty” versus “warm, not thirsty”;
- quoted or hypothetical need: “she said ‘I am hungry’” versus the agent being hungry;
- stale state: a previously true goal followed by an explicit switch;
- lexical decoy: many food words while the operative constraint is threat or warmth;
- Korean/English pairs with the same meaning and pairs with one altered relation;
- identical current observation with different hidden history, and different wording with the same history.

Reject a semantic architecture if it fails the negation/counterfactual pairs while passing ordinary paraphrases, if performance collapses when lexical families are held out, or if a raw-input baseline matches it within the preregistered tolerance. Reject the recurrent mechanism if shuffled history or future-leaking history performs equally well, state survives an episode reset, or benefit appears on stateless trials but not history-dependent trials. Reject multiplicative routing if additive conditioning or a parameter-matched direct model matches it across held-out families. Reject contrastive alignment if time-shifted or animal/session-ID controls retain the reported advantage.

Biological inspiration is also falsifiable. A circuit analogy loses support when the proposed computation does not require the defining variable or causal structure from the cited experiment—for example, “goal routing” that works after goal shuffling, “persistent state” that works after every-step reset, or “cross-modal alignment” learned without paired modalities. Such failure does not invalidate the software as an engineering tool; it invalidates the named biological interpretation.

### Adaptation comparison

If sufficient additional training data are later collected, compare frozen base, bridge updates at ranks 4/16/64/unrestricted, encoder LoRA at preregistered ranks, DoRA, and full encoder tuning under the same family-level splits. Match optimizer steps and report trainable parameters, peak memory, elapsed time, effective update rank, and frozen-general-domain regression tests. A larger method is rejected when its held-out gain does not exceed seed variation, calibration or multilingual robustness worsens materially, or exact retrieval rises through training-family memorization without temporal-task improvement. Full fine-tuning requires a data-volume and compute justification before execution; it is not a default baseline for 32 examples.

## Implementation contract for the next round

New work should be isolated to `examples/bio_bridge/architecture*.py`, matching tests, and `results/architecture-bridge/`. Store split manifests, input hashes, model revision, topology hash, fitted parameters, state-reset rules, and exact prediction arrays. Development chooses the architecture; the final authored-family set is opened once. Replay means deterministic reconstruction of saved computations, not biological experience replay.
