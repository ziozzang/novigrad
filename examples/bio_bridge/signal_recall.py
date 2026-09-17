"""Frozen-policy cue schedules and amplitude diagnostics, with explicit host memory.

No learning, language-model generation, or biological time constant is implied.
Run baseline, temporal, then strength; each stage writes a separate artifact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from safetensors.numpy import load_file
from novigrad import Engine
from delayed_credit import semantic_ports

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/signal-bridge"
SEEDS = [601, 602, 603]
MODES = ["direct", "hold", "ttl8", "leaky4", "leaky16", "leaky16_cutoff"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases():
    case_path = Path(__file__).with_name("sparsity_confirmation.json")
    feature_path = ROOT / "results/bio-bridge/confirmation-features.safetensors"
    cases = json.loads(case_path.read_text())
    features = load_file(str(feature_path))["embeddings"]
    labels = np.array([["water", "food", "warmth", "rest"].index(r["class"]) for r in cases])
    return semantic_ports(features), labels, {str(p.relative_to(ROOT)): sha(p) for p in [case_path, feature_path]}


def checkpoint(seed):
    return ROOT / f"results/bio-bridge/sparse-d128-f0.02-s{seed}.safetensors"


def trace(rows, mode):
    """Transform one episode, resetting all host state at its boundary.

    Hold/TTL detect exact nonzero input, not a trusted cue marker. A nonzero
    distractor refreshes them too. Cutoff gates emitted activity, not stored state.
    """
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or not np.isfinite(rows).all() or (rows < 0).any():
        raise ValueError("expected finite nonnegative time x ports")
    if mode not in MODES:
        raise ValueError("unknown host-memory mode")
    state = np.zeros(rows.shape[1], np.float32)
    emitted = []
    age = 0
    for row in rows:
        if mode == "direct":
            state = row.copy()
        elif mode in ("hold", "ttl8"):
            if np.any(row):
                state = row.copy()
                age = 0
            elif mode == "ttl8":
                age += 1
                if age >= 8:
                    state.fill(0)
        else:
            tau = 4. if mode == "leaky4" else 16.
            state = np.float32(np.exp(-1. / tau)) * state + row
        output = state.copy()
        if mode == "leaky16_cutoff" and np.linalg.norm(output) < .05:
            output.fill(0)
        emitted.append(output)
    return np.asarray(emitted)


def sequences(x, labels, schedule, dose, background):
    """24 ticks: need changes at tick12; paired rows rotate classes deterministically."""
    if schedule not in ("continuous", "one_shot", "periodic4"):
        raise ValueError(schedule)
    if dose not in ("per_pulse", "equal_total"):
        raise ValueError(dose)
    if not np.isscalar(background) or not np.isfinite(background) or background < 0:
        raise ValueError("background must be finite and nonnegative")
    if x.ndim != 2 or len(x) != len(labels) or set(labels) != {0, 1, 2, 3}:
        raise ValueError("matched feature rows and all four classes required")
    step = {"continuous": 1, "one_shot": 12, "periodic4": 4}[schedule]
    amplitude = 1. if dose == "per_pulse" else step / 12.
    rows, targets, cue_rows = [], [], []
    for i, label in enumerate(labels):
        candidates = np.flatnonzero(labels == (label + 1) % 4)
        other = x[candidates[i % len(candidates)]]
        distractors = np.flatnonzero(labels == (label + 2) % 4)
        distractor = x[distractors[i % len(distractors)]]
        episode = np.zeros((24, x.shape[1]), np.float32)
        for t in range(24):
            cue = x[i] if t < 12 else other
            episode[t] = background * distractor
            if t % step == 0:
                episode[t] += amplitude * cue
        rows.append(episode)
        targets.append([int(label)] * 12 + [int((label + 1) % 4)] * 12)
        cue_rows.append(np.stack([x[i]] * 12 + [other] * 12))
    return np.asarray(rows), np.asarray(targets), np.asarray(cue_rows), np.arange(24) % step == 0


def metrics(p, labels):
    pred = p.argmax(axis=-1)
    correct = np.take_along_axis(p, labels[..., None], axis=-1)[..., 0]
    return {"accuracy": float(np.mean(pred == labels)),
            "mean_correct_probability": float(correct.mean()),
            "mean_entropy_nats": float((-p * np.log(np.maximum(p, 1e-30))).sum(axis=-1).mean())}


def evaluate(engine, rows, targets, mode, pulse_mask):
    transformed = np.stack([trace(episode, mode) for episode in rows])
    p = np.asarray(engine.infer_batch(transformed.reshape(-1, rows.shape[-1]).tolist()))
    p = p.reshape(len(rows), rows.shape[1], 4)
    blank_probe = np.array([t % 12 != 0 for t in range(24)])
    return {"all_ticks": metrics(p, targets),
            "post_initial_cue_ticks": metrics(p[:, blank_probe], targets[:, blank_probe]),
            "cue_on_ticks": metrics(p[:, pulse_mask], targets[:, pulse_mask]),
            "cue_off_ticks": metrics(p[:, ~pulse_mask], targets[:, ~pulse_mask]) if (~pulse_mask).any() else None,
            "first_phase": metrics(p[:, :12], targets[:, :12]),
            "second_phase": metrics(p[:, 12:], targets[:, 12:]),
            "cue_on_tick_count": int(pulse_mask.sum()),
            "cue_off_tick_count": int((~pulse_mask).sum()),
            "accuracy_by_tick": (p.argmax(-1) == targets).mean(0).tolist(),
            "correct_probability_by_tick": np.take_along_axis(p, targets[..., None], -1)[..., 0].mean(0).tolist(),
            "mean_input_norm_by_tick": np.linalg.norm(transformed, axis=-1).mean(0).tolist()}, p


def base_report(stage, provenance):
    return {"stage": stage, "design": {
        "data": "Reused 32-text confirmation set; exploratory mechanistic audit, not fresh holdout",
        "policy": "frozen 128D 2%-active policies from prior campaign; three training seeds, same anatomy",
        "no_learning": True, "logical_ticks_not_seconds": True,
        "primary": "post-initial-cue accuracy at ticks1..11 and13..23 on clean one-shot episodes",
        "settings": "All grids specified before running this study; no best-configuration selection",
        "zero_input": "uniform policy; argmax ties select class0, yielding 25% on balanced labels",
        "source_sha256": sha(Path(__file__)), "provenance": provenance,
        "checkpoints": {str(checkpoint(s).relative_to(ROOT)): sha(checkpoint(s)) for s in SEEDS}}, "runs": []}


def baseline(x, labels, provenance):
    report = base_report("baseline", provenance)
    rows, targets, _, pulse_mask = sequences(x, labels, "one_shot", "per_pulse", 0.)
    for seed in SEEDS:
        engine = Engine.load(checkpoint(seed)); before = engine.weights
        result, p = evaluate(engine, rows, targets, "direct", pulse_mask)
        np.testing.assert_array_equal(p[:, ~pulse_mask], .25)
        assert before == engine.weights
        report["runs"].append({"seed": seed, **result})
    report["primary_mean"] = float(np.mean([r["post_initial_cue_ticks"]["accuracy"] for r in report["runs"]]))
    return report


def temporal(x, labels, provenance):
    report = base_report("temporal", provenance)
    report["design"].update({"modes": MODES, "schedules": ["continuous", "one_shot", "periodic4"],
        "doses": {"per_pulse": "unit amplitude per pulse", "equal_total": "sum cue amplitude=1 per12-tick phase"},
        "backgrounds": [0., .25, 1.], "trace": "host preprocessing only, no trained recurrence",
        "hold_detection": "any nonzero input; background refreshes memory too, no oracle event gating",
        "cue_off": "actual schedule pulse mask; continuous schedule has no cue-off probes",
        "decay_boundary": "native hidden-max normalization removes nonzero scalar decay; only direction, exact zero, host cutoff or numerical underflow changes the effective signal",
        "cutoff": "fixed emitted-state norm .05; uncalibrated absolute amplitude gate"})
    for seed in SEEDS:
        engine = Engine.load(checkpoint(seed)); before = engine.weights
        for schedule in ["continuous", "one_shot", "periodic4"]:
            for dose in ["per_pulse", "equal_total"]:
                for background in [0., .25, 1.]:
                    rows, targets, cue_rows, pulse_mask = sequences(x, labels, schedule, dose, background)
                    isolated = np.asarray(engine.infer_batch(cue_rows.reshape(-1, x.shape[1]).tolist())).reshape(len(x), 24, 4)
                    for mode in MODES:
                        result, p = evaluate(engine, rows, targets, mode, pulse_mask)
                        result["isolated_cue_prediction_agreement"] = float((p.argmax(-1) == isolated.argmax(-1)).mean())
                        result["mean_abs_probability_change_vs_isolated_cue"] = float(np.abs(p-isolated).mean())
                        report["runs"].append({"seed": seed, "schedule": schedule, "dose": dose, "background": background, "mode": mode, **result})
        assert before == engine.weights
    report["clean_one_shot_primary"] = {m: float(np.mean([r["post_initial_cue_ticks"]["accuracy"] for r in report["runs"] if r["mode"] == m and r["schedule"] == "one_shot" and r["dose"] == "per_pulse" and r["background"] == 0])) for m in MODES}
    return report


def strength(x, labels, provenance):
    report = base_report("strength", provenance)
    report["global_gain"], report["relative_mixture"], report["decay_silence"] = [], [], []
    gains = [0., 1e-6, .01, .1, 1., 10., 100., 1e6]
    report["design"].update({"global_gains": gains, "mixture_ratios": [0., .1, .25, 1., 4., 10.],
        "decay_delays": [0, 4, 16, 32, 64, 128, 256],
        "strength_scope": "global scalar vs relative feature direction; neither is calibrated odor concentration"})
    distractors = []
    for i, label in enumerate(labels):
        candidates = np.flatnonzero(labels == (label+1)%4)
        distractors.append(x[candidates[i % len(candidates)]])
    distractors = np.stack(distractors)
    for seed in SEEDS:
        engine = Engine.load(checkpoint(seed)); before = engine.weights
        reference = np.asarray(engine.infer_batch(x.tolist()))
        for gain in gains:
            p = np.asarray(engine.infer_batch((gain*x).tolist()))
            report["global_gain"].append({"seed": seed, "gain": gain, **metrics(p, labels),
                "max_probability_change": float(np.abs(p-reference).max()),
                "changed_predictions": int((p.argmax(-1) != reference.argmax(-1)).sum())})
        for ratio in [0., .1, .25, 1., 4., 10.]:
            mixed = x + ratio*distractors
            mixed /= np.linalg.norm(mixed, axis=1, keepdims=True)
            p = np.asarray(engine.infer_batch(mixed.tolist()))
            report["relative_mixture"].append({"seed": seed, "distractor_ratio": ratio, **metrics(p, labels),
                "mean_target_minus_distractor_probability": float((p[np.arange(len(x)), labels]-p[np.arange(len(x)), (labels+1)%4]).mean()),
                "max_probability_change": float(np.abs(p-reference).max()),
                "changed_predictions": int((p.argmax(-1) != reference.argmax(-1)).sum())})
        for mode in ["leaky16", "leaky16_cutoff", "ttl8"]:
            for gain in [.01, 1., 100.]:
                silence = np.zeros((257, x.shape[1]), np.float32)
                states = []
                delays = [0, 4, 16, 32, 64, 128, 256]
                for row in x:
                    silence[0] = gain * row
                    states.append(trace(silence, mode)[delays])
                states = np.stack(states)
                p = np.asarray(engine.infer_batch(states.reshape(-1, x.shape[1]).tolist())).reshape(len(x), len(delays), 4)
                report["decay_silence"].append({"seed": seed, "mode": mode, "gain": gain,
                    "delays": delays, "accuracy": (p.argmax(-1) == labels[:,None]).mean(0).tolist(),
                    "mean_correct_probability": np.take_along_axis(p, np.broadcast_to(labels[:,None,None], (len(x),len(delays),1)), -1)[...,0].mean(0).tolist(),
                    "mean_state_norm": np.linalg.norm(states,axis=-1).mean(0).tolist()})
        assert before == engine.weights
    report["global_positive_max_probability_change"] = max(r["max_probability_change"] for r in report["global_gain"] if r["gain"] > 0)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["baseline", "temporal", "strength"])
    args = parser.parse_args(); start=time.perf_counter()
    x, labels, provenance = load_cases()
    report = globals()[args.stage](x, labels, provenance)
    report["runtime_seconds"] = time.perf_counter()-start
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f"{args.stage}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"stage": args.stage, "seconds": report["runtime_seconds"],
        "primary": report.get("primary_mean", report.get("clean_one_shot_primary")),
        "max_positive_gain_delta": report.get("global_positive_max_probability_change")}))


if __name__ == "__main__":
    main()
