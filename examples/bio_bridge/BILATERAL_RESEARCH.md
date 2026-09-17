# A bilateral Novi-state ↔ frozen-LLM interface

## Scope

This project can test whether an engineered Novi state helps a frozen FunctionGemma issue valid tool calls and whether validated calls can change the next Novi input. It cannot read a fly's thoughts, establish a neural language, or demonstrate a biological brain–LLM interface. The local Novi rates are simulated states derived from language embeddings and a connectome-inspired graph; they are not recordings from an animal.

The two directions must remain distinct:

```text
read:  current Novi tensors -> resampler -> FunctionGemma conditioning -> tool call
write: validated tool call -> declared signed port transform -> next Novi input
loop:  environment outcome -> trusted feedback -> next state; never model-authored reward
```

A successful read does not prove the LLM used the state causally. A successful write does not prove that the selected PN root IDs have the assigned semantics.

## What the primary studies establish

| Primary study | Relevant result | Limit here |
|---|---|---|
| [Li and Liang (2021)](https://aclanthology.org/2021.acl-long.353/) | Prefix tuning conditioned a frozen language model with learned continuous vectors. | Their prefixes were task parameters, not trial-varying neural measurements. Success would support a conditioning mechanism, not neural decoding. |
| [Lester et al. (2021)](https://aclanthology.org/2021.emnlp-main.243/) | Soft prompt vectors adapted frozen language models, with effects depending strongly on model scale. | FunctionGemma is much smaller and specialized for function calling; transfer is an empirical question. |
| [Alayrac et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) | Flamingo used a trainable Perceiver Resampler and gated cross-attention to present variable visual features to a frozen language model. | Novi neuron rows are not visual tokens, and modifying transformer blocks would no longer be the smallest frozen-model experiment. |
| [Li et al. (2023)](https://proceedings.mlr.press/v202/li23q.html) | BLIP-2 used a learned querying transformer between frozen image and language models. | Q-Former gains relied on large paired image–text data. Small synthetic goal labels cannot justify the same cross-modal claim. |
| [Tang et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304553/) | Subject-specific fMRI encoding models plus a language model reconstructed aspects of perceived and imagined continuous language; cooperation and extensive paired data mattered. | This was human fMRI with subject-specific paired recordings, not direct soft-prefix injection and not evidence that simulated fly states contain language. |
| [Ye et al. (2025), BrainLLM](https://www.nature.com/articles/s42003-025-07731-7) | A participant-specific adapter mapped fMRI features to vectors matching an LLM's text-embedding width, concatenated them with preceding-text embeddings, and trained the adapter with a generative loss while keeping the LLM fixed. The study compared aligned brain input with permuted brain input and a standard LLM. | The recordings were responses to the continuation being reconstructed, with hundreds to thousands of paired samples per participant. This is a direct continuous-input precedent, not evidence that simulated Novi features correspond to language. |
| [Willett et al. (2023)](https://www.nature.com/articles/s41586-023-06377-x) | An intracortical speech BCI decoded temporal neural features into phoneme probabilities and combined them with a language model; held-out sentences and raw-decoder measures separated neural and language-model contributions. | A language model can repair decoder errors. End-to-end text quality alone cannot show how much information came from the neural channel. |
| [Kim et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28473639/) | Optogenetic displacement of a fly heading representation changed a maintained circuit state. | This supports the logic of read–perturb–read causal tests in one heading circuit, not writing language concepts into PNs. |
| [Zhang et al. (2021)](https://www.nature.com/articles/s41551-021-00736-7) | A closed-loop animal BMI linked online neural-state decoding to stimulation and measured behavioral consequences. | It demonstrates timing and causal-loop discipline, not correspondence between the present software ports and biological stimulation. |

Two recent preprints are useful as proposals and cautions, not settled primary evidence. [Tang et al. (2026), NOBEL](https://arxiv.org/abs/2602.21522) proposes a unified token space and LLM backbone for EEG, MEG, and dual-path fMRI processing, including direct stimulus inputs. Because it is an arXiv v1 preprint, the present design borrows only the explicit multi-source alignment question; it does not assume the claimed cross-modal generality or causal interpretation. [Dhiman (2026)](https://arxiv.org/abs/2604.04033) reports that apparent advantages of a connectome-constrained model were reduced by shared random initialization and degree-preserving rewired controls. This single-author arXiv v1 study motivates stricter topology controls here; it is not independent validation of Novi.

## Existing local interface

The current `function_bridge` uses a strict four-tool grammar: `set_goal`, `observe_heading`, `choose_action`, and `get_status`. `Session` validates a complete call before mutation, allows only one pending action, and accepts reward through a trusted environment-only hook. `NoviBackend` converts four fixed goal sentences into rates and applies native inference. FunctionGemma generation is presently text/schema conditioned; no continuous Novi tensor enters the model.

Those properties should be preserved. The new experiment adds a conditioning channel without making reward model-callable and without weakening the parser.

## Smallest runnable experiment

Freeze the local BF16 FunctionGemma 270M weights. Train only a small resampler from a declared Novi state snapshot to `K` vectors in the model input-embedding width:

```text
S_t in R^(N x F)                   # selected rates, not labels
Q in R^(K x d_r)                   # learned query parameters
H_t = cross_attention(Q, project(S_t))
P_t = layer_norm(H_t W_out)        # K soft input tokens
LLM input = [P_t ; embedded chat-template tokens]
```

For the first bounded run, `S_t` should be a small, fixed set of already available tensors such as PN ports, pooled KC rates, MBON rates, current heading encoding, and a pending-action flag. Record each source separately. Do not include target action, class label, expected tool name, future reward, or post-action state.

Use FunctionGemma's actual `inputs_embeds`, attention mask, and position handling, followed by greedy generation and the unchanged strict parser. Token-level prefixes are the primary condition. A pooled control maps the entire snapshot to one repeated or one single vector with matched trainable-parameter budget. The comparison asks whether multiple prefix tokens retain useful source/temporal distinctions; it does not call individual tokens neurons.

### Data task

Construct sequential episodes in which identical user text requires different calls or arguments because the observable Novi state/history differs. Examples:

- ambiguous “continue” after different valid goals;
- `choose_action` allowed only after a goal exists and no action is pending;
- a goal switch after environment feedback;
- heading report followed by a status request;
- cue dropout where a bounded state, rather than target labels, carries the previous observation.

Split by scenario family before creating paraphrases or Korean translations. The old FunctionGemma train/holdout examples may be reported as inherited diagnostics, but a new authored family holdout is needed for the bilateral claim.

## Mandatory read-channel controls

Run every condition through actual frozen FunctionGemma generation and strict parsing:

1. **Text-only:** current schema/chat path, no prefix.
2. **Zero prefix:** same shapes and masks with all-zero vectors.
3. **Row-shuffled state:** another episode's state, permuted within balanced label strata where appropriate.
4. **Time-shifted state:** previous/next episode state with current text, excluding legitimate history conditions.
5. **Label-oracle prefix:** directly encode the expected call/argument. This is a leakage ceiling, clearly marked invalid for deployment.
6. **Pooled prefix:** one pooled state vector with matched trainable parameters.
7. **Token prefix:** `K` resampled vectors.
8. **Structured-tool text:** serialize only causally available state into ordinary text. This is a strong, interpretable baseline.

If text-only or structured state text matches the soft prefix, the continuous interface has not shown added value. If shuffled state performs like aligned state, the model is ignoring the Novi channel. If only the label oracle works, the interface pipeline is functional but the measured state is insufficient.

### Continuous injection versus reranking

The comparison must distinguish two ways a language prior can dominate:

```text
continuous injection: state adapter -> soft tokens -> frozen LLM generation
reranking:             text-only LLM candidates -> state score selects candidate
```

BrainLLM is directly relevant because it tested the first pattern against permuted-brain and standard-LLM controls, while Tang et al. (2023) used neural evidence in candidate search/scoring. In this project, run both with the same candidate budget and the same state features. Reranking is restricted by its candidate set; continuous injection can still emit fluent prior-driven text that ignores its prefix. Neither should be declared superior from task accuracy alone.

Add a **prior-only candidate coverage** measure: whether the correct serialized call already appears in text-only top-`k` generation. Report improvement separately when the target is already covered and when it is absent. Compare aligned state with state permuted across rows, time-shifted within a scenario, and replaced by a trainable constant prefix. Stratify by text-only target surprisal. A larger aligned-minus-permuted advantage on high-surprisal targets is evidence that aligned input changes generation, but it remains a software dependence result.

Guard against a learned constant or class prior by balancing targets within every split and testing a constant user prompt. Also evaluate contradictory text: text requests one goal while the state corresponds to another, with the intended authority rule declared in advance. If the model always follows text, the prefix is unused; if it always follows state, it may have learned a shortcut rather than context-sensitive arbitration. The label-oracle condition must never be mixed into adapter selection.

For any claimed connectome contribution, add topology nulls outside the locked biological graph: a shared-initialization degree-preserving rewire ensemble, a port permutation preserving input-degree strata, and a random projection with matched feature width and norm. Sparse random graphs matched only by total edge count are insufficient. These are computational falsification controls; they do not determine whether the biological circuit implements language.

Measure exact valid-call accuracy, schema rejection, tool-name accuracy, argument accuracy, negative-case refusal, and sequence success under generated-history rollout. Also report the LLM log-probability margin for the expected first call tokens where technically reliable; do not call softmax a calibrated confidence without calibration evaluation.

## Write channel and causal loop

Only a fully parsed and schema-validated call may produce a write command. Keep proposed and committed writes separate:

```text
generated bytes
  -> strict parse + schema/order checks
  -> proposed command (no mutation)
  -> host safety/state-machine check
  -> signed port vector with declared gain and TTL
  -> engine tick
  -> environment observation/reward
  -> next read snapshot
```

`set_goal` may write an engineered goal vector to declared input ports for a bounded TTL. `observe_heading` may write a circular host observation only if the environment supplied it; FunctionGemma must not invent it. `choose_action` requests native policy inference but does not itself provide reward. `get_status` is read-only. Every write log should include source call ID, port IDs, signed values, gain, TTL, pre/post state hashes, and environment outcome.

The minimal causal ablations are committed write versus no write, sign-flipped write, port-permuted write, delayed write, and gain-matched random write. A changed native action after a write demonstrates software sensitivity to the engineered input. It does not identify a biological PN semantic role.

## Teacher forcing and leakage

Train-time teacher forcing may supply the correct prior tool call or action. Deployment receives the model's own parsed output. Mixing these regimes can create a false closed-loop result: one early error changes later state, while an oracle history hides that error during evaluation.

Report two separate modes:

- **oracle-history diagnostic:** correct prior calls and rewards are supplied; this measures conditional prediction only;
- **generated rollout:** only validated generated calls and real environment outcomes enter the next step; invalid calls cause the declared no-op/error transition.

Primary sequence success must use generated rollout. Never train or evaluate the read resampler on a state produced using the same row's target label. Fit normalizers, pooling, resampler parameters, early stopping, and thresholds using training/development episodes only. Group all time steps from an episode and all paraphrase/translation variants into one split.

## Acceptance and rejection

Keep the soft-prefix mechanism only if aligned token prefixes outperform text-only, zero, shuffled, time-shifted, and parameter-matched pooled controls on held-out scenario families, and if the gain persists in generated rollout. Require exact saved-weight replay and unchanged FunctionGemma hashes.

Reject the bilateral interpretation if any of these occurs:

- final state contains target/tool/label fields or post-outcome features;
- shuffled or time-shifted states retain the claimed benefit;
- oracle-history success disappears under generated rollout;
- only direct label-oracle prefixes succeed;
- PN writes alter reported state but not the preregistered downstream action/reward;
- a direct structured-text baseline matches the prefix;
- final examples influence resampler size, token count, prompt, checkpoint choice, or stopping.

Even a retained system is an engineered cybernetic loop between frozen FunctionGemma and Novi. Biological validation would require real paired neural recordings, controlled stimulation, behavior, held-out animals/sessions, and causal perturbations.

## Implementation boundary

Place new code in dedicated bilateral experiment modules and results, without changing the existing public protocol. Save a versioned episode manifest, split groups, model and tokenizer hashes, exact chat template, state-source schema, causal timestamps, resampler weights, optimizer history, generated bytes, parse results, write logs, and environment outcomes. Freeze development choices before encoding the final set. Replay loads saved weights and regenerates predictions; it is not biological replay.
