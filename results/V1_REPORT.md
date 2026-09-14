# Novigrad v1 learning validation

Reward learning and relearning after a rule reversal were validated in a rate model constrained by real FlyWire ALPN→KC→MBON connectivity. The final confirmation used seeds 101–108, held out from tuning, across three tasks and three reward conditions: 72 runs total. [한국어 보고서](V1_REPORT.ko.md).

Each run had 2,000 training episodes and 2,000 relearning episodes. Each evaluation used 1,024 independently generated noisy inputs. The learning rate was 0.02, logit gain 12, with an incoming-weight-sum limit per output. Values below are means over eight seeds. “Correct probability” is the probability assigned to the correct action.

| Task | Greedy before | Greedy after | Correct probability | Zero reward | Random reward | Immediately after reversal | After relearning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary input discrimination | 46.40% | 100.00% | 99.55% | 49.61% | 48.56% | 0.45% | 99.60% |
| Four-input classification | 21.75% | 100.00% | 98.29% | 24.58% | 22.79% | 0.57% | 98.32% |
| XOR | 48.32% | 100.00% | 89.98% | 50.11% | 51.61% | 9.96% | 90.12% |

All tasks passed the predetermined criteria: mean correct probability at least 85% after task reward and relearning, at least 20 percentage points above each control, at least 75% for each seed, and changed learned weights. No weights changed in the zero-reward condition. Evaluation also preserved weights.

Weights on existing KC→MBON connections did change. Per-output weight-sum normalization can also affect inactive synapses, so `changed_weights` must not be interpreted as the number of reward-eligible connections.

## Checkpoint and resume

Official Python Safetensors read the three circuit models and the generic example model. An independent NumPy forward pass agreed with Rust CLI inference. For the real circuit runner, predictions before and after saving matched, and resumed learning under an identical subsequent input/action/reward stream produced bit-identical weights.

Maximum absolute error in the independent inference comparison: 4.89e-07, within the tolerance of 2e-6 plus relative error.

## M2 Ultra throughput

This is a hot-loop measurement of fixed-input forward plus zero reward. File loading was measured separately. It is not throughput for actual reward updates.

```text
load_ms=8.026 input_neurons=319 hidden_neurons=5177 output_neurons=96 input_edges=27848 plastic_edges=62261
iterations=10000 elapsed_ms=1027.532 episodes_per_second=9732.1
```

Training-experiment durations are in `metrics.csv`. They include evaluation, checkpoint checks, and some reproduction runs, so they should not be used as pure training throughput.

## Reproduction and limits

- [All measurements](v1-confirmation/metrics.csv), [pass/fail summary](v1-confirmation/summary.txt), and [input, model, and code hashes](v1-confirmation/manifest.json).
- Trained [binary](v1-confirmation/binary.safetensors), [four-way](v1-confirmation/fourway.safetensors), and [XOR](v1-confirmation/xor.safetensors) models.
- The nine conditions for seed 101 were rerun with the final code. Every metric except elapsed time matched the prior run. On this single seed, random reward also achieved a high binary relearning score, causing the single-seed judgment to fail. The final success judgment uses the predetermined eight-seed mean ([reproduction record](reproduction.json)).
- Relearning failures of an earlier unnormalized model remain in `v1-validation/` and `v1-final/`. The performance criteria were not lowered; engine normalization was changed.
- The circuit uses real connectivity constraints, but input patterns and rewards are synthetic, and the output-to-action mapping is engineered. This is not learning in the whole fly brain or a model of actual dopamine action.
- This is a two-layer model with fixed ALPN→KC and plastic KC→MBON connections. The results do not validate learning in arbitrary-depth or recurrent circuits, or biological time constants.
- XOR evaluation covers all four combinations seen during training with newly sampled noise. It does not show inference on unseen combinations or generalization to real sensory data.
- These experiments do not compare the connectome's merit with random connectivity. The scope is an initial CPU engine that supports generic inputs and topology.
