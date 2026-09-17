"""Causal input probe for frozen, matched state-conditioned connectome policies."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import numpy as np
from novigrad import Engine

from internal_state import ROOT, encode, make_cases, utility

SOURCE = ROOT / "results/function-bridge-deep/internal-state-long.json"
OUTPUT = ROOT / "results/function-bridge-deep/state-probe.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actions(engine, encoded):
    return np.asarray(engine.infer_batch(encoded.tolist())).argmax(axis=1)


def score(selected, original_actions, original_utility):
    obtained = np.where(selected == 0, original_utility, 0.0)
    best = np.maximum(original_utility, 0.0)
    return {"utility": float(obtained.mean()),
            "regret": float((best - obtained).mean()),
            "approach_rate": float((selected == 0).mean()),
            "action_change_fraction_vs_original": float((selected != original_actions).mean())}


def main():
    start = time.perf_counter()
    source = json.loads(SOURCE.read_text())
    runs = []
    for source_run in source["runs"]:
        seed = source_run["seed"]
        record = source_run["checkpoints"]["state_conditioned"]
        checkpoint = ROOT / record["path"]
        hash_before = sha256(checkpoint)
        if hash_before != record["sha256"]:
            raise RuntimeError(f"seed {seed}: checkpoint hash differs from training record")
        engine = Engine.load(checkpoint)
        weights_before = engine.weights
        contexts = make_cases(np.random.default_rng(100_000 + seed),
                              source["design"]["test_n_per_seed"])
        actual_utility = utility(contexts)
        original_actions = actions(engine, encode(contexts))

        clamped = contexts.copy()
        clamped[:, 0] = 0.5
        shuffled = contexts.copy()
        shuffled[:, 0] = shuffled[np.random.default_rng(900_000 + seed).permutation(len(shuffled)), 0]
        variants = {
            "original": score(original_actions, original_actions, actual_utility),
            "hunger_clamped_0.5": score(actions(engine, encode(clamped)), original_actions, actual_utility),
            "hunger_shuffled": score(actions(engine, encode(shuffled)), original_actions, actual_utility),
        }
        hash_after = sha256(checkpoint)
        runs.append({"seed": seed, "checkpoint": record["path"], "checkpoint_sha256": hash_before,
                     "checkpoint_hash_unchanged": hash_before == hash_after,
                     "weights_unchanged": weights_before == engine.weights, "metrics": variants})

    names = runs[0]["metrics"]
    mean = {name: {key: float(np.mean([r["metrics"][name][key] for r in runs]))
                   for key in names[name]} for name in names}
    std = {name: {key: float(np.std([r["metrics"][name][key] for r in runs]))
                  for key in names[name]} for name in names}
    result = {
        "design": {"source": str(SOURCE.relative_to(ROOT)), "seeds": [0, 1, 2],
                   "test_context_seed_rule": "100000 + seed; exactly matches long experiment",
                   "shuffle_seed_rule": "900000 + seed",
                   "evaluation": "frozen greedy actions; utility always computed from original contexts",
                   "purpose": "matched-policy causal input sensitivity; no retraining and no parameter selection"},
        "runs": runs, "mean": mean, "population_std_across_seeds": std,
        "runtime_seconds": time.perf_counter() - start,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mean": mean, "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
