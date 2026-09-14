# MAGI-inspired three-policy voting

Three independently trained Novigrad circuits observe the same driving state and vote. This is an engineering ensemble inspired by the fictional three-part decision system, not a simulation of biological fly swarm cognition.

Each member starts from the same untrained FlyWire-derived weights, then learns with its own action-sampling RNG and environment seeds. The task, reward, architecture, and learning rate are the same. Checkpoints are selected on validation games. All members are frozen for the new test games.

Two predeclared rules are compared:

- **Majority vote:** each member chooses its highest-probability action. The action with most votes wins. If all three differ, mean probability breaks the tie among tied actions; an exact tie uses the lowest action index.
- **Probability mean:** average all three distributions and select the largest probability. A confident dissenter can outweigh two weak preferences, unlike majority voting.

No hidden “safe action” rule overrides these decisions. The experiment compares every member, the best single member chosen on validation, both ensembles, and the always-slower rule. Disagreement is measured on the same observations within each ensemble trajectory. Different policies generate different trajectories, so decisions from separate games are not paired as if their observations matched.

```sh
.venv/bin/python -m pip install -r requirements-simulation.txt
cargo build --release --features api --bin novi_engine --bin novi_api
.venv/bin/python scripts/train_magi.py --out results/magi-reproduction
```

The three members train for 128 episodes each (384 total); each single policy trains for 128. This is not an equal-training-budget comparison. Training environment seed ranges are 5000–5127,6000–6127,7000–7127; validation 30000–30007; test 40000–40029. All ranges are separate from the earlier driving experiment. No test-time learning or test-selected weights are used.

Models remain separate Safetensors files. `bundle.json` lists members, action meanings, encoding, and the default voting rule. `ensemble_policy.py` provides the reusable categorical aggregation mechanism. Members must share input-port order and action meanings; the constructor checks port order, while callers are responsible for compatible action semantics.

Three correlated mistakes can still produce a unanimous wrong decision. Different training seeds do not guarantee complementary skills. A useful ensemble requires competent, meaningfully different members; adding copies of the same policy cannot create new evidence. The unit tests cover clone invariance, majority versus confidence averaging, invalid probabilities, and all-different tie resolution.

To watch a saved ensemble (or write a GIF without a window):

```sh
.venv/bin/python scripts/play_magi.py results/magi-reproduction/bundle.json --human
.venv/bin/python scripts/play_magi.py results/magi-reproduction/bundle.json \
  --mode mean --gif /tmp/novi-magi.gif
```

Three API processes keep their models loaded; inference is currently sequential across members and measured accordingly. This avoids reloading checkpoints at every decision. No special MAGI roles or different reward objectives are assigned in this first experiment: diversity comes only from independent training experience and exploration.

Measured outcome: best single0/30 collisions, majority20/30, probability mean3/30. The two weaker members voted alike on every ensemble observation. See the [full report](../../results/MAGI_REPORT.md).
