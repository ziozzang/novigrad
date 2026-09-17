# Frozen Gemma → novi semantic bridge

This runnable example connects original **float32 EmbeddingGemma 300M** to the existing Rust ALPN→KC→MBON policy through the native Python binding. Gemma stays frozen; environment reward trains novi. It is a four-action contextual bandit, not a complete navigation simulator or anatomical brain interface.

## Measured results

M2 Ultra, three fresh policy runs (seeds 101/202/303), 32 training sentences, 12 validation sentences, 24 held-out English sentences and 12 Korean sentences. Four balanced needs: water, food, warmth, rest. Each seed independently permutes the actuator mapping; chance is 25%.

| Policy | English test mean | Korean test mean |
|---|---:|---:|
| novi / reward | 84.72% | 83.33% |
| novi / shuffled_reward | 25.00% | 25.00% |
| linear / reward | 83.33% | 86.11% |
| linear / shuffled_reward | 25.00% | 25.00% |

A full-768-dimensional nearest-training-example baseline scored 91.67% in both tests. It uses labeled exemplars and a larger representation, so it is a semantic-encoder control rather than a matched reward learner. The matched linear policy receives the same 319 input ports as novi. These tiny exploratory samples do not establish a statistically significant advantage for either policy. Per-seed predictions/confusions and all learning-rate candidates are retained in [experiment.json](../../results/gemma-bridge/experiment.json).

The original reward learner failed after actuator meanings were rotated: novi fell to 25% after a second 4,096-interaction learning phase. This is an important negative result: first-task learning did not imply ongoing adaptation. A follow-up adds 20% uniform exploration, importance-corrected reward gradients and entropy regularization. [adaptation.json](../../results/gemma-bridge/adaptation.json) retains this experiment; it improved some seeds but did not reliably solve reversal. Further low-gain experiments are reported separately. Follow-ups reuse the holdout and are exploratory, not fresh confirmations.

Save/reload predictions were exactly equal for all three original novi runs across all 80 feature rows. A live, previously unlisted Korean request for cool water produced the water actuator with probability 0.9394. This one demonstration is not an additional benchmark.

## What is connected and learned

EmbeddingGemma runs the bundled sentence-transformer pipeline (bidirectional transformer, mean pooling, two dense projections, normalization) with the bundled `Classification` prompt. We use the first 128 coordinates, renormalize, split signs into 256 nonnegative channels, and pad to 319. No label-dependent projection is fitted. The projection loses the other 640 coordinates; port assignment is an engineered convention, not demonstrated odor coding.

`Engine.from_edges` freshly initializes the plastic graph from `data/pn_kc.tsv` and `data/kc_mbon.tsv`. It uses 5,177 KCs, 96 MBONs, 20% active hidden cells, no output L1 homeostasis, and an opponent decoder with four action groups. Existing vision/driving trained weights are not used. The anatomical initialization is deterministic; seeds vary observation sampling, actions and actuator permutation rather than randomizing the connectome.

Each learner receives eight sampled observations per update, chooses actions from its own policy, and gets +1 for the environment's correct actuator or −1 otherwise. There is no teacher action passed into learning. In the shuffled-reward control, rewards are permuted within the same batch, preserving the batch's reward histogram but removing which action earned each reward. The fixed environment itself knows labels to calculate rewards; this is not autonomous discovery of its objective.

Six validation-only candidates use the same interaction budget: novi learning rates 0.03/0.1/0.3, linear 0.3/1/3. Selection uses validation accuracy then correct-class probability. Each final phase has 512 batches / 4,096 interactions. At reversal the same learned policy continues with a rotated mapping; it is deliberately not reset, since adaptation of existing weights is the question.

## Run on a Mac

From the repository root:

```sh
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -r requirements-gemma.txt
.venv/bin/python examples/gemma_bridge/encode.py --model /path/to/google_embeddinggemma-300m
.venv/bin/python examples/gemma_bridge/experiment.py
.venv/bin/python examples/gemma_bridge/adaptation.py
.venv/bin/python examples/gemma_bridge/demo.py --model /path/to/google_embeddinggemma-300m '목이 말라서 시원한 물을 마시고 싶어요.'
.venv/bin/python -m unittest discover -s examples/gemma_bridge -p 'test_*.py' -v
```

Prepare the pinned connectome files with the repository's existing data workflow first. Supply the original local model directory including its pooling/dense modules. Model weights, generated embeddings and trained checkpoints are excluded from Git; commands regenerate them. Google model weights retain their own Gemma terms; the repository's MIT license covers our bridge code, not those weights. No additional quantization or encoder fine-tuning was applied.

The text encoder uses PyTorch **MPS** for the validated reference path. The policy uses native Rust CPU. This example does not yet port EmbeddingGemma to MLX-C, and does not reuse the earlier sparse Metal policy benchmark as a language-encoder speed claim. Initial measurements encoded all 80 sentences in 0.479 seconds at batch size 16; individual calls ranged from about 30 to 462 ms with shape/cache warmup effects. They are not a stable end-to-end real-time guarantee. Exact versions, original weight SHA-256 values and raw timings are in [encoder.json](../../results/gemma-bridge/encoder.json).

## Biological interface research

[Research review](RESEARCH.md) / [한국어 연구 검토](RESEARCH.ko.md) distinguishes mushroom-body value learning from FC2/EPG/PFL goal–heading computation. [anatomy.py](anatomy.py) audits real sparse paths, and the [candidate port manifest](../../results/gemma-bridge/candidate-ports.json) lists proposed neuron IDs without inventing directional phases. None of the FC2/EPG/PFL candidate ports are driven by the current semantic example.

[한국어](README.ko.md)

## Reversal mechanism experiments

| Experiment / policy | Before reversal English | Relearned English | Relearned Korean |
|---|---:|---:|---:|
| adaptation.json / novi | 86.11% | 38.89% | 38.89% |
| adaptation.json / linear | 93.06% | 40.28% | 38.89% |
| adaptation-lowgain.json / novi | 86.11% | 45.83% | 47.22% |
| adaptation-lowgain.json / linear | 93.06% | 48.61% | 41.67% |
| adaptation-entropy05.json / novi | 87.50% | 86.11% | 83.33% |
| adaptation-entropy05.json / linear | 93.06% | 50.00% | 50.00% |

`adaptation.json`: epsilon 0.2, entropy coefficient 0.1, 512 updates/phase, original gain 6. `adaptation-lowgain.json`: gain 1 for novi, novi learning rate 0.3, 2,048 updates/phase, entropy 0.1. `adaptation-entropy05.json`: same longer low-gain schedule, entropy 0.5. Linear retains its originally validation-selected rate and has no separate logit gain. Each phase therefore uses 16,384 actual environment interactions in the longer runs; additional rows compute an exact entropy gradient, not additional environmental observations.

The final exploratory novi condition recovers 83.33–87.50% English after reversal without resetting weights. This supports investigating policy saturation/exploration, but it does not isolate one factor causally or establish superiority over a separately retuned linear learner. More entropy deliberately reduces certainty; evaluate task success and flexibility rather than expecting maximal action probability.

```sh
.venv/bin/python examples/gemma_bridge/adaptation.py --gain 1 --steps 2048 --novi-lr .3 --beta .5 --filename adaptation-entropy05.json
```

All original and failed results are preserved. A new independent evaluation set is needed before treating the follow-up as a confirmed improvement. The saved `novi-*.safetensors` bundles currently correspond to the original experiment before reversal; adaptation experiments record metrics and can be rerun, but do not overwrite those bundles.
