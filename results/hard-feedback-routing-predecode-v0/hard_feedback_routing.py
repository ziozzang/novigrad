#!/usr/bin/env python3
"""Hard posterior-to-prototype routing diagnostic for saved feedback cases.

This post-hoc hypothesis was selected after seeing the continuous-mixture
feedback results.  The host chooses a labeled prototype; this is an interface
diagnostic, not evidence that the LM or circuit performs that reasoning.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import bilateral_lm as lm
from bilateral_closed_loop import DEFAULT_INDICES
import causal_feedback as cf


OUT = cf.ROOT / "results/hard-feedback-routing"
POLICY = "native701"
MODE = "belief_full"
SCENARIO = "immediate"


def hard_next_prefix(initial, prototypes, belief, mode, last_action):
    """Route the host posterior argmax to one old-train labeled prototype."""
    if mode != MODE:
        raise ValueError("hard override is defined only for belief_full")
    weights = np.asarray(belief, dtype=np.float64)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("belief must contain four finite nonnegative masses")
    if tuple(prototypes.shape[:1]) != (4,) or prototypes.shape[1:] != initial.shape:
        raise ValueError("expected four prototypes matching initial prefix shape")
    selected = prototypes[int(np.argmax(weights))]
    source_norm, selected_norm = initial.norm(), selected.norm()
    if not torch.isfinite(source_norm) or not torch.isfinite(selected_norm) or source_norm <= 1e-12 or selected_norm <= 1e-12:
        raise ValueError("degenerate prefix")
    return selected * (source_norm / selected_norm)


@contextmanager
def temporary_hard_routing():
    """Temporarily replace only causal_feedback.next_prefix, then restore it."""
    original = cf.next_prefix
    cf.next_prefix = hard_next_prefix
    try:
        yield
    finally:
        cf.next_prefix = original


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _frozen_files():
    own = Path(__file__)
    names = ("causal_feedback.py", "feedback_environment.py", "bilateral_lm.py",
             "bilateral_closed_loop.py", "test_hard_feedback_routing.py")
    inputs = [cf.OUT / name for name in ("protocol-lock.json", "evaluation-lock.json", "results.json",
                                          "inputs.safetensors", "cases.json", "generation-cache.json",
                                          "generation-prefixes.safetensors")]
    return [own, *[own.with_name(name) for name in names], *inputs]


def prepare():
    if OUT.exists():
        raise FileExistsError("use a new output directory for another protocol")
    cf.checked()
    evaluation = json.loads((cf.OUT / "evaluation-lock.json").read_text())
    for name, digest in evaluation.items():
        if cf.sha(cf.OUT / name) != digest:
            raise ValueError("causal-feedback artifact differs from evaluation lock: " + name)
    OUT.mkdir(parents=True)
    causal_lock = json.loads((cf.OUT / "protocol-lock.json").read_text())
    record = causal_lock["maps"][POLICY]
    write(OUT / "protocol-lock.json", {
        "version": 1,
        "status": "post_hoc_hypothesis_selected_after_continuous_mixture_results",
        "scope": "Reused64 interface diagnostic; no fresh-generalization or model-selection claim",
        "models": list(cf.MODELS), "policy": POLICY, "scenario": SCENARIO, "mode": MODE,
        "cases_per_model": 64, "selected_actual_generation_replay_indices": list(DEFAULT_INDICES),
        "executor_actions": record["actions"], "executor_checkpoint_sha256": record["checkpoint_sha256"],
        "change_contract": "Inside an exception-safe context, replace only causal_feedback.next_prefix for belief_full. It chooses argmax(host posterior), routes one old-train labeled prototype, and rescales it to the initial prefix norm. causal_feedback.rollout, environment, prior, executor, first raw generation, and four-step budget remain unchanged.",
        "boundaries": ["The external host already selects a labeled class prototype; this is not LM or biological reasoning",
                       "No base, adapter, circuit, prior, or prototype training",
                       "First decision is the exact saved original generation for the same case",
                       "Existing prefix generations are reused by digest; only unseen hard-routing prefixes require LM decoding"],
        "frozen": {str(path.relative_to(cf.ROOT)): cf.sha(path) for path in _frozen_files()},
        "causal_feedback_evaluation_lock_sha256": cf.sha(cf.OUT / "evaluation-lock.json")})
    print("prepared", cf.sha(OUT / "protocol-lock.json"))


def checked():
    cf.checked()
    lock_path = OUT / "protocol-lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("run prepare before decoding")
    lock = json.loads(lock_path.read_text())
    for relative, digest in lock["frozen"].items():
        if cf.sha(cf.ROOT / relative) != digest:
            raise ValueError("frozen dependency changed: " + relative)
    if lock["executor_actions"] != json.loads((cf.OUT / "protocol-lock.json").read_text())["maps"][POLICY]["actions"]:
        raise ValueError("executor mapping changed")
    return lock


def _original_rows(results, model):
    return results["/".join((model, POLICY, SCENARIO, "original"))]["cases"]


def _continuous_rows(results, model):
    return results["/".join((model, POLICY, SCENARIO, MODE))]["cases"]


def _assert_first_identity(rows, originals):
    for row, old in zip(rows, originals):
        left, right = row["trace"][0], old["trace"][0]
        if left.get("raw") != right.get("raw") or left.get("command") != right.get("command") or left.get("invalid") != right.get("invalid"):
            raise AssertionError("same-case first raw/command identity failed")


def _comparison(rows, continuous):
    hard = np.asarray([row["success"] for row in rows], dtype=np.int8)
    soft = np.asarray([row["success"] for row in continuous], dtype=np.int8)
    return {"success_gap_hard_minus_continuous": float((hard - soft).mean()),
            "hard_helped": int(np.sum((hard == 1) & (soft == 0))),
            "hard_hurt": int(np.sum((hard == 0) & (soft == 1))),
            "continuous_successes": int(soft.sum()), "hard_successes": int(hard.sum()),
            "hard_mean_physical_actions": float(np.mean([row["physical_actions"] for row in rows])),
            "continuous_mean_physical_actions": float(np.mean([row["physical_actions"] for row in continuous]))}


def run():
    lock = checked()
    if (OUT / "evaluation-lock.json").exists() or (OUT / "started.json").exists():
        raise FileExistsError("refusing repeated evaluation")
    write(OUT / "started.json", {"protocol_sha256": cf.sha(OUT / "protocol-lock.json")})
    tensors = load_file(str(cf.OUT / "inputs.safetensors"))
    cases = json.loads((cf.OUT / "cases.json").read_text())
    old_results = json.loads((cf.OUT / "results.json").read_text())
    old_cache = json.loads((cf.OUT / "generation-cache.json").read_text())
    runtime = lm.Runtime.load()
    before = runtime.memory_fingerprint()
    cache, new_prefixes, replay_keys = {m: {} for m in cf.MODELS}, {}, {m: set() for m in cf.MODELS}
    conditions, comparisons = {}, {}
    with temporary_hard_routing():
        for model in cf.MODELS:
            initial = tensors[model + "/initial"].to("mps")
            prototypes = tensors[model + "/prototypes"].to("mps")
            priors = tensors[model + "/prior"].numpy()
            def generate(prefix, digest, case_index, model=model):
                if digest in cache[model]:
                    if case_index in DEFAULT_INDICES and digest not in old_cache[model]:
                        replay_keys[model].add(digest)
                    return cache[model][digest]
                if digest in old_cache[model]:
                    raw = old_cache[model][digest]
                else:
                    raw = runtime.generate(prefix)
                    new_prefixes[model + "/" + digest] = prefix.detach().cpu().contiguous()
                cache[model][digest] = raw
                if case_index in DEFAULT_INDICES and digest not in old_cache[model]:
                    replay_keys[model].add(digest)
                return raw
            rows = [cf.rollout(initial[i], prototypes, priors[i], lm.GOALS.index(case["class"]),
                               lock["executor_actions"], MODE, cf.SCENARIOS[SCENARIO], i, generate)
                    for i, case in enumerate(cases)]
            original, continuous = _original_rows(old_results, model), _continuous_rows(old_results, model)
            _assert_first_identity(rows, original)
            conditions[model] = {"metrics": cf.summarize(rows), "cases": rows}
            comparisons[model] = _comparison(rows, continuous)
    after = runtime.memory_fingerprint()
    if after != before:
        raise AssertionError("frozen base model memory changed")
    save_file(new_prefixes, str(OUT / "new-prefixes.safetensors"))
    write(OUT / "generation-cache.json", cache)
    write(OUT / "results.json", {"conditions": conditions, "comparisons": comparisons,
          "boundary": "Post-hoc reused64 interface diagnostic; the host selects the class prototype"})
    write(OUT / "run-manifest.json", {"base_memory_before_sha256": before, "base_memory_after_sha256": after,
          "old_generation_cache_sha256": cf.sha(cf.OUT / "generation-cache.json"),
          "new_unique_generations": len(new_prefixes), "selected_replay_keys": {m: sorted(v) for m, v in replay_keys.items()},
          "cached_existing_generations_are_not_independent_passes": True})
    files = ("protocol-lock.json", "started.json", "new-prefixes.safetensors", "generation-cache.json", "results.json", "run-manifest.json")
    write(OUT / "evaluation-lock.json", {name: cf.sha(OUT / name) for name in files})


def verify():
    checked()
    evaluation = json.loads((OUT / "evaluation-lock.json").read_text())
    for name, digest in evaluation.items():
        if cf.sha(OUT / name) != digest:
            raise ValueError("evaluation artifact changed: " + name)
    manifest = json.loads((OUT / "run-manifest.json").read_text())
    runtime = lm.Runtime.load()
    if runtime.memory_fingerprint() != manifest["base_memory_before_sha256"]:
        raise AssertionError("base model identity changed")
    cache = json.loads((OUT / "generation-cache.json").read_text())
    prefixes = load_file(str(OUT / "new-prefixes.safetensors"))
    counts = {}
    for model, keys in manifest["selected_replay_keys"].items():
        for digest in keys:
            prefix = prefixes[model + "/" + digest]
            if cf.prefix_key(prefix) != digest or runtime.generate(prefix.to("mps")) != cache[model][digest]:
                raise AssertionError("selected actual generation replay differs")
        counts[model] = len(keys)
    write(OUT / "verification.json", {"selected_actual_generation_exact": True,
          "case_indices": list(DEFAULT_INDICES), "replayed_new_prefixes": counts,
          "all_evaluation_hashes_pass": True,
          "boundary": "Only new hard-routing prefixes reached by eight prespecified cases are rerun through the actual LM"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    globals()[parser.parse_args().stage]()
