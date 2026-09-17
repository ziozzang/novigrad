# Bio bridge: measured diagnostic results

This directory tests whether two local Gemma interfaces can support small, auditable learning diagnostics. Frozen EmbeddingGemma features serve as engineered sensory inputs. FunctionGemma emits typed executive proposals. A CPU `novi` policy receives sampled-action reward on top of fixed connectome-derived edges. These roles are software roles: language vectors are not odor codes, top-k activity is not APL inhibition, a tagged queue is not a synaptic trace, and neither model is a fly prefrontal cortex homologue.

Nothing here changes the library API or demonstrates a biological mechanism. The studies are diagnostic defaults run on hand-authored language cases. They do not use a physical environment, and greedy classification accuracy is not physical reward.

## Current campaign status

The primary metric remains **53.90625% balanced accuracy** from the frozen FunctionGemma base condition: the mean of 45.3125% accuracy on 64 valid calls and 62.5% rejection on 16 negative cases. Four core experiment suites produced three retained diagnostic additions and one discarded prompt intervention. A later 18-run streaming-credit follow-up is reported separately and does not change the primary metric.

The newly authored material totals **172 texts**:

- 80 FunctionGemma cases: 64 valid operations and 16 requests that should produce no call, balanced across English and Korean.
- 60 EmbeddingGemma cases: 48 known needs and 12 unknown/ambiguous cases, balanced across English and Korean.
- 32 later sparsity-confirmation texts: four known needs, English/Korean, evaluated without retraining the selected checkpoints.

“Newly authored” describes files created for this round. It does not mean an independent external benchmark or independent biological replication. The function and 60-item embedding sets are hand-authored protocol tests. The 32-item set was authored after the 2%-versus-20% contrast had been selected and remains a same-author, same-domain confirmation.

## 1. FunctionGemma: typed intent proposals

All four conditions use greedy decoding and the same 80 cases. The adapter is the existing LoRA from `results/function-bridge/lora`; no training occurred in this experiment. “Policy” is an added developer instruction about latest-request priority and abstention. A correct negative case requires no parsed call. `get_status` or any other call is still incorrect when the expected result is no call.

| Condition | Valid accuracy (64) | Negative rejection (16) | Balanced accuracy | Unsupported mutating/state calls on negatives |
|---|---:|---:|---:|---:|
| Base | 45.3125% | 62.5% | **53.90625%** | 6 |
| Base + policy | 39.0625% | 43.75% | 41.40625% | 9 |
| Adapter | 78.125% | 0% | 39.0625% | 13 |
| Adapter + policy | 56.25% | 0% | 28.125% | 10 |

The adapter improved positive-call matching but failed all 16 abstention cases. The prompt policy reduced balanced accuracy for both base and adapter, so it is the discarded intervention. These data do not support a claim that instruction text repairs semantic refusal. Parser nonexecution may also be malformed generation rather than an intentional refusal. Multi-turn histories contain supplied canonical assistant calls, not self-generated rollouts; the separate history files are an ablation, not evidence of stable deployed memory.

The 16-case history ablation scored 10/16 with full history versus 9/16 with only the latest user message for base, and 12/16 versus 12/16 for the adapter. A post hoc semantic-agreement gate raised adapter balanced accuracy to 52.34% by rejecting seven negative cases, but reduced valid matches from 50 to 39. It still trailed the ungated base (53.91%); no gate became a production default. The gate reads only the latest message and cannot resolve history-dependent intent.

## 2. EmbeddingGemma: geometry, margin rejection, and a policy proxy

The prototype classifier uses class means from the inherited 32-item training split. Its rejection margin was selected on 12 inherited validation examples plus eight separately written unknown/ambiguous calibration examples. The table evaluates the 60 new texts. “Forced” always predicts one of four needs and therefore rejects none; its balanced score is mechanically halved by 0% unknown rejection.

| Dimensions | Forced known accuracy | Forced unknown rejection | Forced balanced | Calibrated known accuracy | Calibrated unknown rejection | Calibrated balanced |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 77.08% | 0% | 38.54% | 47.92% | 75.00% | 61.46% |
| 128 | 81.25% | 0% | 40.63% | 50.00% | 83.33% | **66.67%** |
| 768 | 79.17% | 0% | 39.58% | 56.25% | 75.00% | 65.63% |

The calibrated result shows the expected tradeoff: rejection improves while known-class accuracy drops. The margin is exploratory because only 20 calibration items selected it and only 12 negative test items assess rejection.

Google’s [official EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) lists 768, 512, 256, and **128** as MRL output sizes and instructs users to truncate and re-normalize. Thus 128 is the smallest officially documented MRL size. The 64-dimensional condition is an intentionally below-card truncation diagnostic; its behavior must not be presented as a supported MRL configuration.

The `novi` proxy trained 24 policies: 2 dimensions × 4 active fractions × 3 sampling seeds. Each received 4,096 sampled-action interactions on the inherited 32 training texts. Mean greedy top-1 accuracy on the 48 known new texts was:

| Dimensions | 2% active | 10% active | 20% active | 50% active |
|---:|---:|---:|---:|---:|
| 64 | 46.53% | 44.44% | 40.97% | 31.94% |
| 128 | **57.64%** | 50.00% | 46.53% | 45.14% |

The fixed 128d, 2%-versus-20% contrast was then applied to 32 later texts without checkpoint selection or retraining. Mean accuracy was **52.08%** at 2% and **37.50%** at 20%; paired seed differences were 12.5, 18.75, and 12.5 percentage points. On those same 32 items, a supervised prototype reached **81.25%** at both 128d and 768d. The large gap is a negative result for the learned policy: sparse activity helped relative to 20% under this setup but remained far below a simple supervised feature-space readout.

The three seeds are stochastic training histories applied to the same examples, not independent datasets. English/Korean items and semantic templates are related, so an item-level bootstrap would overstate precision. No confidence interval or biological inference is claimed.

## 3. Context memory: oracle tags and engineered routing

Fifteen runs compare five architectures over three seeds. Every model learns context A, then a +1 rotated label mapping in context B, then refreshes A. A/B is a trusted oracle tag. Test and Korean splits are inherited and reused as frozen phase-boundary probes; they are not a new evaluation split.

| Architecture | A after A | A retained after B | B after B | A after refresh | Parameter note |
|---|---:|---:|---:|---:|---|
| Blind | 51.39% | 34.72% | 50.00% | 50.00% | one engine |
| Concat | 43.06% | 16.67% | 72.22% | 31.94% | one engine, two context ports |
| Concat normalized | 43.06% | 16.67% | 72.22% | 31.94% | post hoc unit-norm control |
| Conjunctive | 51.39% | 51.39% | 55.56% | 54.17% | context selects disjoint input ports |
| Modular | 51.39% | 51.39% | 55.56% | 54.17% | two engines, **2× plastic weights** |

These are greedy top-1 accuracies. Correct-class softmax means for the displayed test probes range from about 0.25 to 0.39 and are not calibrated confidence or sampled-action success rates. Conjunctive input routing and a second modular engine preserve A here, but the modular comparison doubles parameters and neither design discovers context. The normalized concat control was added after observing concat and produced essentially identical outputs because it applies a constant scaling to already normalized semantic features plus a one-hot context. The combined artifact must therefore remain exploratory; not all settings were frozen before the post hoc addition.

This is a reversal/interference diagnostic. It does not test extinction, erasure, renewal, reinstatement, or a mushroom-body compartment mechanism.

## 4. Delayed credit: batched tagged replay

The completed delayed-credit artifact contains 36 main runs (3 seeds × 3 lags × 4 conditions) and six immediate amplitude controls, for 42 runs. Labels determine only environment reward: +1 for the sampled mapped action and −1 otherwise. The learner receives the observation, sampled action, and scalar reward.

Each cell below is **test greedy accuracy / mean correct-class probability**, averaged across three seeds. Korean values follow in parentheses.

| Lag | Wrong current | Exact tagged | Decayed tagged | Zero reward |
|---:|---:|---:|---:|---:|
| 0 | 62.50/0.266 (58.33/0.271) | 62.50/0.266 (58.33/0.271) | 62.50/0.266 (58.33/0.271) | 23.61/0.250 (19.44/0.250) |
| 4 | 25.00/0.250 (25.00/0.249) | 59.72/0.266 (58.33/0.271) | 63.89/0.260 (61.11/0.263) | 23.61/0.250 (19.44/0.250) |
| 16 | 27.78/0.250 (25.00/0.250) | 59.72/0.266 (58.33/0.271) | 65.28/0.253 (58.33/0.254) | 23.61/0.250 (19.44/0.250) |

Immediate controls scaled reward by the same constants as lag-4 and lag-16 decay. Their test accuracies were 68.06% and 65.28%; Korean accuracy was 58.33% for both. They matched or exceeded delayed-decayed means, so the experiment supplies no evidence that delay improves learning. Fixed-lag decay is also just a constant reward-amplitude change.

The result has an important batching artifact. All 32 actions in a rollout are sampled from one frozen weight snapshot, and learning waits for a 32-reward batch. Lags 4 and 16 therefore fall within one sampling phase rather than testing fully sequential online learning across intervening updates. Exact tagging demonstrates replay of the stored row/action against a deliberately wrong-current control under that batching scheme. It is not a biological eligibility curve.

The pending deque itself is delay-limited, but `seen` and `trials_by_id` retain all trial IDs and rows. The overall history is therefore unbounded over a long-lived process. During terminal drain, wrong-current credit repeatedly uses the last trial because no new current trial exists. Both facts limit production interpretation.

### Streaming follow-up

The follow-up contains 18 runs: 3 seeds × 2 lags × 3 credit rules. It performs one inference, schedules and delivers due feedback, and applies a batch-1 update before the next action. It uses 4,096 rewarded actions, plus 16 fresh, unrewarded warm-down actions at lag 16 so every scheduled outcome can arrive without reusing the last trial. The learning rate is 0.009375, exactly 0.3/32, to make update scale more comparable with the older 32-item batches. Shared context indices and sampling uniforms are fixed within each seed across conditions.

| Lag | Wrong current | Exact tagged | Decayed tagged |
|---:|---:|---:|---:|
| 0 | 54.17/0.266 (58.33/0.271) | 54.17/0.266 (58.33/0.271) | 54.17/0.266 (58.33/0.271) |
| 16 | 26.39/0.250 (25.00/0.249) | 55.56/0.266 (58.33/0.271) | 56.94/0.253 (61.11/0.254) |

Cells again show test greedy accuracy / mean correct-class probability, with Korean in parentheses. At lag zero, the three rules are identical and produced identical measured outputs within the run. At lag 16, exact action-ID credit retained approximately its lag-zero aggregate performance while wrong-current credit fell near chance. This fixes the older sampling-phase artifact and demonstrates chronological **software** credit assignment under current-weight updates.

The storage implementation retains only pending rows, rejects stale or duplicate IDs with a monotonic scalar, and recorded maximum pending size 17. It therefore fixes the unbounded `seen`/row history in the older experiment. The conclusion remains narrow: seed-level baseline test accuracies varied from 87.5% to 50% to 25%, only three seeds were run, and the same inherited test items were reused. Decayed tagging multiplies every lag-16 reward by `exp(-2)`, so its difference from exact tagging remains an amplitude confound. No confidence interval, molecular eligibility claim, or physical-environment claim is supported.

## Biological evidence and proxy boundaries

The research rationale and primary sources are in [RESEARCH.md](RESEARCH.md). In brief, [Lin et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/) causally linked APL output, sparse/decorrelated KC responses, and discrimination of similar odors, but the authors did not prove sparsity was the sole mediator. [Hige et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/) and [Handler et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012144/) support compartment- and order-sensitive dopamine plasticity in defined fly preparations. [Felsenberg et al. (2018)](https://doi.org/10.1016/j.cell.2018.08.021) supports parallel opposing memory in a specific extinction task. [Krashes et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2780032/) supports hunger-dependent gating of appetitive memory expression.

| Software element | Operational role | It does not establish |
|---|---|---|
| Frozen EmbeddingGemma | repeatable language feature source | PN/KC odor coding or semantic neurons |
| `active_fraction` top-k | controlled sparse computation | APL inhibition or pattern separation mechanism |
| Sampled-action scalar update | small policy-learning diagnostic | dopamine identity, valence, or synaptic biology |
| Oracle context bit / routed engine | explicit task-state separation | learned context, MB compartment identity, or extinction |
| Tagged delay queue | exact host bookkeeping | molecular eligibility trace |
| FunctionGemma tool call | proposed executive symbol | action execution, trusted state, fly PFC, or neural homology |

## Architecture roadmap

| Timescale / responsibility | Current component | Required next test |
|---|---|---|
| Sensory representation | frozen EmbeddingGemma vector | family-held-out multilingual and ambiguity sets |
| Intent proposal | FunctionGemma typed call | fresh abstention split and self-generated history rollouts |
| Fast memory | bounded host event record | variable delays, out-of-order delivery, and overload recovery |
| Learned value/action | `novi` plastic readout | environment-owned outcomes and safety costs |
| Longer context | explicit state tag or separate module | inferred, partially observed context without oracle leakage |
| Execution | schema validator and host | state preconditions, ordering, and physical/simulated transitions |

The intended separation is memory versus intent: FunctionGemma proposes what a request means; the host and policy retain outcome-linked state and decide whether an operation is valid. There is no PFC analogy.

## Reproduction

Run from the repository root. The commands use local model paths and existing inherited weights. Function and embedding generation use Apple MPS in the current scripts; context and delayed-credit studies use CPU `novi`. These commands overwrite their named result files/checkpoints, so preserve the recorded artifacts if exact provenance matters.

```bash
python examples/bio_bridge/function_scenarios.py --out results/bio-bridge/function-base.json
python examples/bio_bridge/function_scenarios.py --intent-policy --out results/bio-bridge/function-policy.json
python examples/bio_bridge/function_scenarios.py --adapter results/function-bridge/lora --out results/bio-bridge/function-adapter.json
python examples/bio_bridge/function_scenarios.py --adapter results/function-bridge/lora --intent-policy --out results/bio-bridge/function-adapter-policy.json

python examples/bio_bridge/embedding_scenarios.py
python examples/bio_bridge/confirm_sparsity.py
python examples/bio_bridge/context_memory.py --output results/bio-bridge/context-memory.json
python examples/bio_bridge/context_memory.py --append-normalized-control
python examples/bio_bridge/delayed_credit.py
python examples/bio_bridge/streaming_credit.py
python -m unittest discover -s examples/bio_bridge -p 'test_*.py' -q
```

Do not run `context_memory.py` without understanding that a fresh full run replaces the report containing the appended post hoc normalized control. The JSON files contain hashes, per-case outputs, per-seed values, probabilities, checkpoints, and limitations; they are the numerical source of record.

The queue enforces capacity `delay + 1`; overflow leaves the next ID unchanged. The guard-fix rerun matched every metric and training count. Raw checkpoint file hashes changed across runs because metadata key serialization order is not canonical; this is not proof of byte-identical retraining. Current files and their hashes are recorded in the report.

![Biological and Gemma diagnostic comparisons](../../results/bio-bridge/biological-mechanisms.png)

Follow-up: [continuous input, one-shot cues, signal strength, and repeated learning](SIGNAL_RESULTS.md), with [primary-source research](SIGNAL_RESEARCH.md).

Mechanism extraction: [measured adaptation, hidden inhibition, and eligibility comparisons](MECHANISM_RESULTS.md), with [primary-source derivations](MECHANISM_RESEARCH.md).

Embedding-based state readout: [measured KC/MBON reconstruction and causal controls](THOUGHT_RESULTS.md), [research and actual-fly measurement requirements](THOUGHT_RESEARCH.md).
