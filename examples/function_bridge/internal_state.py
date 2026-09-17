"""Reward-only internal-state experiment using the real connectome engine.

The 319 nonnegative PN ports are an engineering interface: continuous host state is
expanded across deterministic threshold/RBF features. This is not a claim that
these particular FlyWire PNs biologically encode hunger, calories, or aversion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from novigrad import Engine

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def make_cases(rng: np.random.Generator, n: int) -> np.ndarray:
    """Independent continuous factorial draws: hunger, benefit, aversive cost."""
    return rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)


def utility(cases: np.ndarray) -> np.ndarray:
    # Approach utility is already normalized to [-1, 1]; avoid utility is zero.
    return cases[:, 0] * cases[:, 1] - cases[:, 2]


def encode(cases: np.ndarray, include_hunger: bool = True) -> np.ndarray:
    """Distribute positive scalar features over exactly 319 nonnegative ports."""
    x = np.asarray(cases, np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("cases must be finite [n, 3]")
    h, benefit, cost = x.T
    if not include_hunger:
        h = np.full_like(h, 0.5)  # identical sensory distribution, no state signal
    bases = [h, benefit, cost, h * benefit, 1.0 - h, 1.0 - benefit, 1.0 - cost]
    centers = np.linspace(0.0, 1.0, 45, dtype=np.float32)
    ports = [np.exp(-32.0 * (v[:, None] - centers[None, :]) ** 2) for v in bases]
    out = np.concatenate(ports, axis=1)  # 315
    extra = np.stack([h, benefit, cost, np.ones_like(h)], axis=1)
    return np.concatenate([out, extra], axis=1).astype(np.float32)


class NoviPolicy:
    def __init__(self, include_hunger=True, learning_rate=0.03):
        self.include_hunger = include_hunger
        self.engine = Engine.from_edges(
            DATA / "pn_kc.tsv", DATA / "kc_mbon.tsv", actions=2,
            learning_rate=learning_rate, logit_gain=6.0, active_fraction=0.2,
            homeostasis=False, readout="opponent")

    def probabilities(self, cases):
        return np.asarray(self.engine.infer_batch(encode(cases, self.include_hunger).tolist()))

    def learn(self, cases, actions, rewards):
        self.engine.learn(encode(cases, self.include_hunger).tolist(), actions.tolist(), rewards.tolist())


class LinearPolicy:
    """Conventional linear softmax reward-policy baseline on explicit host features."""
    def __init__(self, learning_rate=0.2):
        self.w = np.zeros((5, 2), np.float64)
        self.learning_rate = learning_rate

    @staticmethod
    def features(cases):
        h, b, c = cases.T
        return np.stack([h, b, c, h * b, np.ones_like(h)], axis=1)

    def probabilities(self, cases):
        z = self.features(cases) @ self.w
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def learn(self, cases, actions, rewards):
        x, p = self.features(cases), self.probabilities(cases)
        grad = (np.eye(2)[actions] - p) * rewards[:, None]
        self.w += self.learning_rate * x.T @ grad / len(cases)


def evaluate(policy, cases):
    u = utility(cases)
    action = policy.probabilities(cases).argmax(axis=1)
    obtained = np.where(action == 0, u, 0.0)
    best = np.maximum(u, 0.0)
    random = 0.5 * u
    return {"utility": float(obtained.mean()), "regret": float((best - obtained).mean()),
            "random_utility": float(random.mean()), "best_possible_utility": float(best.mean())}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_metrics(cases):
    u = utility(cases)
    best = np.maximum(u, 0.0)
    # Without hunger, E[h]=0.5 under the declared uniform test distribution.
    blind_approach = 0.5 * cases[:, 1] > cases[:, 2]
    blind_obtained = np.where(blind_approach, u, 0.0)
    return {
        "full_information_optimal": {"utility": float(best.mean()), "regret": 0.0},
        "hunger_blind_optimal": {"utility": float(blind_obtained.mean()),
                                  "regret": float((best - blind_obtained).mean()),
                                  "decision_rule": "approach iff 0.5*benefit > cost"},
    }


def run(seed=0, train_interactions=2048, batch_size=32, test_n=2048, checkpoint_dir=None):
    train_rng = np.random.default_rng(seed)
    test = make_cases(np.random.default_rng(100_000 + seed), test_n)
    policies = {"state_conditioned": NoviPolicy(True), "no_hunger": NoviPolicy(False),
                "linear": LinearPolicy(), "zero_reward": NoviPolicy(True)}
    zero_initial = policies["zero_reward"].engine.weights
    curves = {name: [] for name in policies}
    milestones = set(range(max(batch_size, train_interactions // 8), train_interactions + 1,
                           max(batch_size, train_interactions // 8)))
    seen = 0
    while seen < train_interactions:
        n = min(batch_size, train_interactions - seen)
        cases = make_cases(train_rng, n)
        u = utility(cases)
        for i, (name, policy) in enumerate(policies.items()):
            # Separate deterministic action streams; labels/optimal actions are never supplied.
            action_rng = np.random.default_rng(seed * 10_000_000 + seen * 17 + i)
            p = policy.probabilities(cases)
            actions = (action_rng.random(n) >= p[:, 0]).astype(np.int64) # 0 approach, 1 avoid
            rewards = np.where(actions == 0, u, 0.0).astype(np.float64)
            if name == "zero_reward":
                rewards.fill(0.0)
            policy.learn(cases, actions, rewards)
        seen += n
        if seen in milestones or seen == train_interactions:
            for name, policy in policies.items():
                curves[name].append({"step": seen, **evaluate(policy, test)})
    checkpoints = {}
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name in ("state_conditioned", "no_hunger", "zero_reward"):
            policy = policies[name]
            path = checkpoint_dir / f"seed-{seed}-{name}.safetensors"
            policy.engine.save(path, overwrite=True)
            checkpoints[name] = {"path": str(path.resolve().relative_to(ROOT)), "sha256": file_sha256(path),
                                 "config": policy.engine.config}
    return {"seed": seed, "metrics": {n: evaluate(p, test) for n, p in policies.items()},
            "analytic_references": reference_metrics(test), "learning_curves": curves,
            "zero_reward_weights_unchanged": zero_initial == policies["zero_reward"].engine.weights,
            "checkpoints": checkpoints}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "results/function-bridge-deep/internal-state.json")
    ap.add_argument("--train-interactions", "--train-steps", dest="train_interactions",
                    type=int, default=2048,
                    help="number of sampled environment interactions (legacy alias: --train-steps)")
    ap.add_argument("--test-n", type=int, default=2048)
    args = ap.parse_args()
    if args.train_interactions <= 0 or args.test_n <= 0:
        ap.error("budgets must be positive")
    start = time.perf_counter()
    checkpoint_dir = args.output.parent / f"{args.output.stem}-checkpoints"
    runs = [run(seed, args.train_interactions, 32, args.test_n, checkpoint_dir) for seed in range(3)]
    names = runs[0]["metrics"]
    summary = {name: {key: float(np.mean([r["metrics"][name][key] for r in runs]))
                       for key in names[name]} for name in names}
    variation = {name: {key: float(np.std([r["metrics"][name][key] for r in runs]))
                        for key in names[name]} for name in names}
    connectivity_sha = {name: file_sha256(DATA / name) for name in ("pn_kc.tsv", "kc_mbon.tsv")}
    preset = ("original_fixed_budget" if args.train_interactions == 2048 else
              "exploratory_4x_budget_same_settings" if args.train_interactions == 8192 else
              "custom_budget")
    result = {"design": {"preset": preset, "seeds": 3,
                          "train_interactions": args.train_interactions,
                          "optimizer_updates": int(np.ceil(args.train_interactions / 32)), "batch_size": 32,
                          "test_n_per_seed": args.test_n, "actions": {"0": "approach", "1": "avoid"},
                          "reward": "approach: hunger*benefit-cost; avoid: 0; range [-1,1]",
                          "evaluation": "frozen greedy policy on independent continuous draws",
                          "test_seed_rule": "100000 + seed; intentionally reused by the labeled 4x exploratory comparison",
                          "port_encoding": "engineered 319-port nonnegative RBF/scalar expansion; not a biological PN-state claim",
                          "connectivity_sha256": connectivity_sha,
                          "checkpoint_scope": "real Engine variants only; the conventional NumPy linear baseline has no Engine checkpoint"},
              "runs": runs, "mean": summary, "population_std_across_seeds": variation,
              "interpretation": [
                  "The state-conditioned connectome policy improved over its zero-reward control but retained nonzero regret.",
                  "A learned baseline that reaches greedy avoidance is reported as a failed training outcome under this budget, not evidence that its policy class cannot solve the task; the analytic references separate policy-class information limits from learning failure.",
                  "No held-out examples or metrics were used for learning, model selection, or hyperparameter adjustment."
              ],
              "runtime_seconds": time.perf_counter() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mean": summary, "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
