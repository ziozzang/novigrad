# Semantic readout from a frozen fly-inspired bridge

## Result boundary

This experiment decoded **authored semantic descriptions from language-driven simulated activity**. It did not record a fly, recover subjective experience, generate text from neurons, or demonstrate inner language. “Thought” appears in filenames as the experiment name; the measured object is semantic information supplied by frozen EmbeddingGemma and propagated through a fixed connectome-constrained input-to-KC transform.

The design follows [THOUGHT_RESEARCH.md](THOUGHT_RESEARCH.md). The larger proposed hunger–benefit–cost experiment, behavioral state switches, and tests on actual neural recordings have not been performed.

No core API or default changed. All model weights were frozen for decoding and intervention. The evaluation sets were already used in earlier repository experiments, so these are compatibility diagnostics rather than a fresh external benchmark.

![Semantic decoding and intervention results](../../results/thought-bridge/semantic-readout.png)

Machine-readable overview: [summary.json](../../results/thought-bridge/summary.json). A vector figure is available as [semantic-readout.svg](../../results/thought-bridge/semantic-readout.svg).

## Pipeline

```text
input text
  -> frozen EmbeddingGemma-300m, 128-dimensional MRL embedding
  -> signed 256-port representation, padded to 319 input ports
  -> fixed PN-to-KC transform
  -> native 2% top-k: 104 active units among 5,177 KCs
  -> ridge reconstruction of the 128-dimensional input embedding
  -> cosine search over 40 pre-authored candidate descriptions
```

The candidate catalog contains 32 English/Korean descriptions of four needs—water, food, warmth, and rest—and eight unsupported-state distractors. Candidates were written and embedded before evaluation. Cosine scores are similarities, not probabilities or calibrated confidence. The decoder has no calibrated rejection rule.

The aligned semantic decoder was trained on 32 pairs `(KC activity, original frozen input embedding)`, using ridge `lambda=0.1`. It was **not** fit directly to the four class labels. A nearest candidate supplies a description and its catalog category after reconstruction.

Three controls answer different questions:

- `raw_embedding`: nearest-description search directly from the frozen input embedding; this is the encoding-only ceiling.
- `mean_embedding`: the same constant mean training embedding for every case; this is a no-information reference.
- `shuffled_*`: five ridge decoders trained with shuffled KC–embedding pairings; these test whether correct alignment matters.

## Semantic category decoding

| Decoder | Old English test, 24 | Old Korean test, 12 | Reused confirmation, 32 |
|---|---:|---:|---:|
| Raw frozen embedding | 91.67% | 91.67% | 78.13% |
| KC → aligned embedding ridge | 70.83% | 66.67% | 56.25% |
| Learned MBON rates → aligned embedding ridge, post hoc | 75.00% | 41.67% | 59.38% |
| Mean-embedding control | 25.00% | 25.00% | 25.00% |
| Shuffled-pair ridge, mean of 5 | — | — | 35.63% |
| Shuffled-pair ridge, best of 5 | — | — | 50.00% |

The aligned decoder retained some of the category information already present in the frozen embedding, but it lost substantial performance relative to the raw input. Its 56.25% confirmation result is above the 35.63% mean of the five shuffled controls, although one shuffled run reached 50%. Five shuffled seeds are a small empirical null and are not a significance test.

The aligned decoder's mean cosine to the original input embedding was 0.8214 English, 0.8370 Korean, and 0.8091 confirmation. The constant-mean control still produced high cosines—0.7606, 0.7632, and 0.7489—while category accuracy stayed at 25%. High cosine alone is therefore a weak success criterion in this embedding space.

### Catalog identity diagnostic

A post hoc diagnostic asked whether the exact matching input row could be recovered from a known catalog. It is not free-text reconstruction. Exact catalog identity was 37.50% English, 58.33% Korean, and 40.63% confirmation for the aligned decoder, versus 4.17%, 8.33%, and 3.13% for the mean control. This supports preservation of some case-specific structure, within a closed catalog.

No evaluated decoder selected an unsupported-state candidate. That does **not** demonstrate OOD rejection: there is no calibrated rejection threshold, and the evaluation did not establish coverage for unknown states. A zero hidden vector is explicitly returned as `no_signal` rather than assigned a semantic category.

### Post hoc MBON-stage reconstruction

A later control applied the same reconstruction question at 96 MBON-like rates, after the learned KC-to-MBON weights and before output gains and action aggregation. A separate ridge map with the same fixed `lambda=0.1` was fit on the same 32 `(MBON rates, input embedding)` training pairs, with no hyperparameter tuning. It reached 93.75% on the training rows, 75.00% on the 24-case English test, 41.67% on the 12-case Korean test, and 59.38% on the reused 32-case confirmation set.

This result shows that input semantics remain decodable after the learned readout stage. The target is still the supplied input embedding. It is not an independently measured intent, expected value, decision variable, or internal state. Because this control was added after the KC result and uses one learned policy checkpoint (seed 601), it is exploratory and cannot establish that learning created a semantic state.

Applying the trained MBON decoder to the corresponding untrained MBON rates reduced confirmation category accuracy to 25.00%, and the maximum reconstructed-embedding change was 0.226663. Unlike KC activity, MBON activity changes with the learned weights. This establishes weight dependence of this fitted downstream reconstruction; it does not establish what variable the native policy uses.

The saved MBON decoder round-tripped exactly. Its serialization path was corrected to require C-contiguous arrays after a regression test exposed an incompatible layout; the test now asserts that layout explicitly. This is artifact validation, not scientific evidence.

Source: [decoding.json](../../results/thought-bridge/decoding.json).

## The untrained control separates encoding from policy learning

The PN-to-KC transform precedes the plastic KC-to-MBON readout. A post hoc control saved an untrained native checkpoint with the same fixed input-to-hidden path.

| Quantity on the reused 32 cases | Untrained versus trained seed 601 |
|---|---:|
| Maximum KC hidden difference | 0.0 |
| Maximum semantic-decoder difference | 0.0 |
| Native policy accuracy | 21.88% → 50.00% |

Semantic decoding was identical before and after policy learning even though native action accuracy changed. This is decisive for the scope of the result: the semantic signal belongs to the fixed language/input encoding path. It is not evidence that reward learning created an internal thought representation. The other policy checkpoints also had identical KC hidden activity; they differ downstream in learned MBON weights.

## A separate supervised class-probe intervention

The causal intervention uses a different readout and must not be conflated with the embedding reconstruction above. It fits an independent supervised four-class ridge probe to L2-normalized KC activity from the old 32 training labels. It then ranks active KC contributions to the probe's baseline decoded class.

For an active KC `i` and the probe-decoded class `c`, the ranking quantity is:

```text
contribution_i,c = hidden_i * class_coefficient_i,c
```

Silencing occurs after native 2% top-k and max normalization. The native MBON policy receives the altered activity without renormalization. Ground-truth labels are used only for evaluator metrics, not target selection.

On the reused 32 confirmation inputs, repeated through three downstream policy checkpoints, the baseline class probe was 65.63% correct and the native policy was 52.08% correct. The 96 recorded rows contain only 32 unique inputs: the input-to-KC features and class probe are identical across checkpoints. They are not 96 independent cases or three independent encoders.

### Intervention effects

| Intervention | Units changed or removed | Removed activity L1 | Class-probe flip | Native-policy flip |
|---|---:|---:|---:|---:|
| Top contribution, 5% active count | 6 removed | 7.02% | 50.00% | 26.04% |
| Random active, 5% count-matched | 6 removed | 5.75% | 9.38% | 4.17% |
| Top contribution, 25% active count | 26 removed | 27.47% | 87.50% | 51.04% |
| Random active, 25% count-matched | 26 removed | 24.96% | 15.63% | 9.38% |
| Random, 25% L1-matched post hoc | mean 29.22 changed | 27.47% | 6.25% | 13.54% |
| Top contribution, 50% active count | 52 removed | 52.80% | 100.00% | 57.29% |
| Random active, 50% count-matched | 52 removed | 50.32% | 21.88% | 12.50% |

The original equal-count comparison confounded ranking with removed activity magnitude. The post hoc L1 control addresses that specific confound at 25%: it follows a deterministic random order, fully silences units until near the top-ranked L1 target, and partially attenuates one final unit. It exactly matched the mean removed L1 fraction of 27.47%; the top-ranked intervention still produced much more class-probe flipping (87.50% versus 6.25%) and native-policy flipping (51.04% versus 13.54%).

This shows that the supervised probe identified units whose removal strongly disrupts that probe and often changes the downstream policy in this frozen engineered model. It does not show that these units contain subjective thought. The target class comes from the probe, the data are reused, the L1 match was added after the initial result, and unit selection plus evaluation remain within one small semantic domain. Changing a policy action can also reflect generic representation damage. A future test must select units on training families and estimate effects on disjoint semantic families and tasks.

Sparse shadow-forward probabilities matched the native engine within maximum absolute error `2.98e-8`. This validates the numerical intervention path, not its biological interpretation.

Source: [interventions.json](../../results/thought-bridge/interventions.json).

## CLI demonstration

`thought_embedding.py --text` performs one nearest-candidate readout and also reports native action probabilities. It does not generate language. The default is the KC decoder; `--stage mbon` selects the learned MBON-stage decoder. In the recorded KC smoke demonstration:

```text
input: 목이 마르지만 지금은 따뜻한 곳이 더 필요해요.
nearest category: warmth
nearest description cosine: 0.86077
native warmth probability: 0.27683
```

This is one smoke case, not a benchmark or confidence-calibrated decision. Its record is [demo.json](../../results/thought-bridge/demo.json).

## What an actual fly experiment requires

Testing a fly latent state would require paired biological observations unavailable here:

1. Record identified neural populations while measuring stimulus, movement, choice, outcome, and a controlled internal-state manipulation.
2. Align neural time to cue onset, deliberation, action, and outcome; train a time-causal decoder so post-choice or reward activity cannot predict a pre-choice variable.
3. Hold out entire flies, sessions, and stimulus families. Fit preprocessing, unit selection, and decoder parameters within training folds.
4. Compare neural decoding with stimulus-only, movement, choice-history, and shuffled-label baselines.
5. Perturb candidate neurons during the predeclared time window and test a selective behavioral interaction, such as hunger changing benefit sensitivity while cue identity remains intact.
6. Test recovery, state switching, and anatomically matched control neurons.

Only this combination can distinguish sensory reconstruction, movement correlation, and a behaviorally used latent state. FlyWire paths and synapse counts can nominate cells but cannot provide the missing activity or motivational state.

## Conclusions allowed by these runs

- Frozen EmbeddingGemma already separated the four supplied need categories well.
- A fixed sparse connectome-constrained KC representation preserved enough information for a paired ridge map to recover semantic category above the constant and average shuffled controls.
- The same semantic decoder worked identically on an untrained native model, locating the information upstream of reward learning.
- A post hoc MBON-stage ridge decoded the supplied input semantics above chance from learned rates, but failed on untrained MBON rates; its target was not independent intent or value.
- A separate supervised class probe found ranked KC contributions whose silencing disrupted both its own class and the native policy more than count- or L1-matched random controls.

These findings justify further tests of semantic encoding and causal policy dependence in this software bridge. They do not extract a fly's thinking.

## Reproduction

Using the existing local model, embeddings, and checkpoints:

```bash
cd /Users/a405394/fly/nobi
python examples/bio_bridge/encode_thought_descriptions.py
python examples/bio_bridge/thought_embedding.py
python examples/bio_bridge/thought_embedding.py --text '목이 마르지만 지금은 따뜻한 곳이 더 필요해요.'
python examples/bio_bridge/thought_embedding.py --stage mbon --text '목이 마르지만 지금은 따뜻한 곳이 더 필요해요.'
python examples/bio_bridge/thought_interventions.py
python examples/bio_bridge/thought_interventions.py --append-l1-control
python examples/bio_bridge/summarize_thought.py
python -m unittest discover -s examples/bio_bridge -p 'test_thought*.py'
```

The encoder requires the local EmbeddingGemma weights. The downstream decoding, intervention, and summary steps reuse the saved artifacts.
