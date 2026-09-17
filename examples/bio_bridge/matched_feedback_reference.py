#!/usr/bin/env python3
"""Post-hoc matched-first-command host references for causal feedback.

These controls were designed after results from the first two models had been
seen.  They replay saved text only; they do not run the language model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

import bilateral_lm as lm
import causal_feedback as cf
from feedback_environment import FeedbackConfig, HiddenGoalEnvironment, posterior


OUTPUT = cf.OUT / "matched-reference.json"
MODES = ("forced_first_bayes", "forced_first_no_repeat")
CONFIG = FeedbackConfig(max_steps=4, action_cost=.05)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parsed_command(raw):
    goal = lm.parse_goal(raw) if isinstance(raw, str) else None
    return None if goal is None else lm.GOALS.index(goal)


def choose_after_first(mode, belief, prior_order, used_commands):
    """Choose from controller state only; there is deliberately no target argument."""
    if mode == "forced_first_bayes":
        return int(np.argmax(np.asarray(belief, dtype=np.float64)))
    if mode == "forced_first_no_repeat":
        return next((int(command) for command in prior_order if int(command) not in used_commands), None)
    raise ValueError("unknown matched-reference mode")


def matched_rollout(prior, target, mapping, first_raw, mode, case_index=0):
    """Run one immediate-feedback episode while forcing the saved first command."""
    masses = np.asarray(prior, dtype=np.float64)
    action_map = np.asarray(mapping)
    if masses.shape != (4,) or not np.isfinite(masses).all() or np.any(masses < 0) or masses.sum() <= 0:
        raise ValueError("prior must contain four finite nonnegative masses")
    if action_map.shape != (4,) or action_map.dtype.kind not in "iu" or np.any((action_map < 0) | (action_map >= 4)):
        raise ValueError("mapping must contain four in-range integer actions")
    if mode not in MODES:
        raise ValueError("unknown matched-reference mode")
    if not isinstance(target, (int, np.integer)) or isinstance(target, (bool, np.bool_)) or not 0 <= int(target) < 4:
        raise ValueError("target must be an in-range integer")

    belief = masses / masses.sum()
    order = np.argsort(-belief, kind="stable")
    first = parsed_command(first_raw)
    if first is None:
        return {"case_index": int(case_index), "target": int(target), "trace": [{
            "step": 1, "raw": first_raw, "invalid": True, "belief": belief.tolist(),
            "controller": "forced_saved_first_command"}], "success": False, "decisions": 1,
            "physical_actions": 0, "utility": 0., "first_wrong": True, "rescued": False,
            "invalid": True, "repeated_executed_actions": 0, "success_by_step": [False] * 4}

    # The hidden target enters only the environment and the evaluator below.
    env = HiddenGoalEnvironment([int(target)], seed=2901 + int(case_index), config=CONFIG)
    env.reset()
    history, trace, used = [], [], set()
    first_wrong = True
    for step in range(CONFIG.max_steps):
        command = first if step == 0 else choose_after_first(mode, belief, order, used)
        if command is None:
            break
        used.add(command)
        executed = int(action_map[command])
        observation = env.step(executed)
        physical_success = executed == int(target)
        if step == 0:
            first_wrong = not physical_success
        history.append(observation)
        updated = posterior(masses, history, reliability=.9, action_cost=CONFIG.action_cost)
        trace.append({"step": step + 1, "raw": first_raw if step == 0 else None,
                      "invalid": False, "command": int(command), "executed_action": executed,
                      "controller": "forced_saved_first_command" if step == 0 else mode,
                      "belief": belief.tolist(), "next_belief": updated.tolist(),
                      "observation": observation, "physical_success": bool(physical_success)})
        belief = updated
        if observation["done"]:
            break
    success = any(row["physical_success"] for row in trace)
    actions = [row["executed_action"] for row in trace]
    return {"case_index": int(case_index), "target": int(target), "trace": trace,
            "success": bool(success), "decisions": len(trace), "physical_actions": len(actions),
            "utility": int(success) - CONFIG.action_cost * len(actions), "first_wrong": first_wrong,
            "rescued": bool(first_wrong and success), "invalid": False,
            "repeated_executed_actions": len(actions) - len(set(actions)),
            "success_by_step": [any(row["physical_success"] and row["step"] <= k for row in trace)
                                for k in range(1, 5)]}


def _validate_original_first(original_row, first_raw):
    if not original_row.get("trace"):
        raise ValueError("original case has no decision trace")
    first = original_row["trace"][0]
    if first.get("raw") != first_raw:
        raise AssertionError("saved first raw generation identity changed")
    parsed = parsed_command(first_raw)
    if bool(first.get("invalid")) != (parsed is None):
        raise AssertionError("original parse status disagrees with strict parser")
    if parsed is not None and int(first.get("command", -1)) != parsed:
        raise AssertionError("original first command disagrees with strict parser")
    return parsed


def _comparison(rows, belief_rows):
    success = np.asarray([row["success"] for row in rows], dtype=np.int8)
    reference = np.asarray([row["success"] for row in belief_rows], dtype=np.int8)
    invalid = sum(row["invalid"] for row in rows)
    return {"success_gap_vs_belief_full": float((success - reference).mean()),
            "helped_vs_belief_full": int(np.sum((success == 1) & (reference == 0))),
            "hurt_vs_belief_full": int(np.sum((success == 0) & (reference == 1))),
            "first_invalid_count": int(invalid), "first_valid_success_ceiling_count": len(rows) - int(invalid),
            "first_valid_success_ceiling_rate": float((len(rows) - invalid) / len(rows)),
            "mean_physical_actions": float(np.mean([row["physical_actions"] for row in rows])),
            "belief_full_mean_physical_actions": float(np.mean([row["physical_actions"] for row in belief_rows]))}


def build_report(results, lock, tensors, cases):
    conditions, comparisons = {}, {}
    for model in cf.MODELS:
        priors = tensors[model + "/prior"].numpy()
        for policy, map_record in lock["maps"].items():
            base = "/".join((model, policy, "immediate"))
            original = results[base + "/original"]["cases"]
            belief_rows = results[base + "/belief_full"]["cases"]
            if len(original) != 64 or len(belief_rows) != 64 or len(cases) != 64:
                raise ValueError("expected 64 aligned reused cases")
            first_raws = [row["trace"][0].get("raw") for row in original]
            for row, raw in zip(original, first_raws):
                _validate_original_first(row, raw)
            for mode in MODES:
                rows = [matched_rollout(priors[i], lm.GOALS.index(cases[i]["class"]),
                                        map_record["actions"], first_raws[i], mode, i)
                        for i in range(64)]
                for old, new in zip(original, rows):
                    old_command = _validate_original_first(old, old["trace"][0].get("raw"))
                    new_command = None if new["invalid"] else new["trace"][0]["command"]
                    if old_command != new_command:
                        raise AssertionError("same-case first command identity failed")
                key = base + "/" + mode
                conditions[key] = {"metrics": cf.summarize(rows), "cases": rows}
                comparisons[key] = _comparison(rows, belief_rows)
    return {"design": {"status": "post_hoc_after_first_two_model_results_were_observed",
                       "scope": "descriptive stronger falsification on reused64; no model selection or fresh-generalization claim",
                       "first_action": "Exact strict-parsed raw first generation from each corresponding original case is forced; invalid first generations stop before an environment action",
                       "bayes": "After the forced command, host posterior argmax with fixed reliability .9; controller receives observations but never target",
                       "no_repeat": "After the forced command, unused commands follow the initial prior ranking once each",
                       "environment": vars(CONFIG), "scenarios": ["immediate"],
                       "models": list(cf.MODELS), "policies": list(lock["maps"])},
            "conditions": conditions, "comparisons": comparisons}


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite matched-reference report")
    required = [cf.OUT / name for name in ("results.json", "protocol-lock.json", "inputs.safetensors",
                                            "cases.json", "evaluation-lock.json")]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("completed causal-feedback artifacts are required")
    evaluation = json.loads((cf.OUT / "evaluation-lock.json").read_text())
    for name, digest in evaluation.items():
        if sha256(cf.OUT / name) != digest:
            raise ValueError("causal-feedback artifact differs from evaluation lock: " + name)
    lock = cf.checked()
    results = json.loads((cf.OUT / "results.json").read_text())
    cases = json.loads((cf.OUT / "cases.json").read_text())
    tensors = load_file(str(cf.OUT / "inputs.safetensors"))
    report = build_report(results, lock, tensors, cases)
    sources = [Path(__file__), Path(__file__).with_name("feedback_environment.py"),
               Path(__file__).with_name("causal_feedback.py")]
    report["provenance"] = {"source_sha256": {str(path.relative_to(cf.ROOT)): sha256(path) for path in sources},
                            "input_sha256": {str(path.relative_to(cf.ROOT)): sha256(path) for path in required},
                            "evaluation_lock_verified": True, "language_model_executed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
