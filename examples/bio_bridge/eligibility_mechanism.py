"""Conventional linear eligibility traces; not Novi and not exact fly biology."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import deque
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file

from delayed_credit import ROOT, GEMMA, load_data

OUTPUT = ROOT / "results/mechanism-bridge/eligibility.json"
CHECKPOINTS = OUTPUT.parent / "eligibility-checkpoints"


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def gradient_event(x, action, probabilities):
    direction = -np.asarray(probabilities, np.float64)
    direction = direction.copy()
    direction[action] += 1.0
    return np.outer(np.asarray(x, np.float64), direction)


def norm_match(trace, historical_event):
    trace_norm = np.linalg.norm(trace)
    target_norm = np.linalg.norm(historical_event)
    return np.zeros_like(trace) if trace_norm == 0 else trace * (target_norm / trace_norm)


class LinearTracePolicy:
    def __init__(self, features=319, actions=4, learning_rate=.03, tau=None):
        self.weights = np.zeros((features, actions), np.float64)
        self.trace = np.zeros_like(self.weights)
        self.learning_rate = learning_rate
        self.tau = tau
        self.decay = 0.0 if tau is None else float(np.exp(-1.0 / tau))

    def probabilities(self, x):
        return softmax(np.asarray(x, np.float64) @ self.weights)

    def event(self, x, action, probabilities=None):
        p = self.probabilities(x) if probabilities is None else probabilities
        return gradient_event(x, action, p)

    def advance_trace(self, event):
        self.trace *= self.decay
        self.trace += event

    def update(self, reward, credit):
        delta = self.learning_rate * float(reward) * credit
        self.weights += delta
        return float(np.linalg.norm(delta))


def sample_action(p, uniform):
    return min(3, int(np.searchsorted(np.cumsum(p / p.sum()), uniform, side="right")))


def score(policy, x, labels):
    before = policy.weights.copy()
    p = np.asarray([policy.probabilities(row) for row in x], np.float64)
    prediction = p.argmax(axis=1)
    entropy = -(p * np.log(np.maximum(p, 1e-15))).sum(axis=1)
    return {"accuracy": float((prediction == labels).mean()),
            "mean_correct_probability": float(p[np.arange(len(p)), labels].mean()),
            "mean_entropy": float(entropy.mean()), "predictions": prediction.tolist(),
            "dtype": str(p.dtype), "weights_unchanged": bool(np.array_equal(before, policy.weights))}


def train(policy, x, labels, indices, uniforms, delay, condition, actions_scheduled=2048):
    pending = deque()
    blank = np.zeros(x.shape[1], np.float64)
    update_norms, event_norms = [], []
    positive = 0
    for logical_time in range(actions_scheduled + delay):
        row = x[indices[logical_time]] if logical_time < actions_scheduled else blank
        p = policy.probabilities(row)
        action = sample_action(p, uniforms[logical_time])
        event = policy.event(row, action, p)
        event_norms.append(float(np.linalg.norm(event)))
        if condition.startswith("trace_tau"):
            policy.advance_trace(event)
        if logical_time < actions_scheduled:
            reward = 1.0 if action == labels[indices[logical_time]] else -1.0
            positive += int(reward > 0)
            pending.append((logical_time + delay, reward, event))
        while pending and pending[0][0] <= logical_time:
            _, reward, cached_event = pending.popleft()
            if condition == "current_only":
                credit = event
            elif condition == "exact_cached_gradient":
                credit = cached_event
            elif condition.startswith("trace_tau"):
                credit = (norm_match(policy.trace, cached_event)
                          if condition.endswith("_normmatched") else policy.trace)
            elif condition == "zero_reward":
                credit, reward = event, 0.0
            else:
                raise ValueError(condition)
            update_norms.append(policy.update(reward, credit))
    if pending:
        raise RuntimeError("warmdown did not drain queue")
    return {"rewarded_actions": actions_scheduled, "warmdown_steps": delay,
            "updates": len(update_norms), "positive_rewards": positive,
            "negative_rewards": actions_scheduled - positive,
            "mean_event_gradient_norm": float(np.mean(event_norms[:actions_scheduled])),
            "mean_update_norm": float(np.mean(update_norms)),
            "max_update_norm": float(np.max(update_norms)),
            "final_weight_l2": float(np.linalg.norm(policy.weights)),
            "final_trace_l2": float(np.linalg.norm(policy.trace)),
            "all_finite": bool(np.isfinite(policy.weights).all() and np.isfinite(policy.trace).all())}


def checkpoint(policy, name):
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / f"{name}.safetensors"
    metadata = {"format": "novigrad.linear_eligibility_reference", "version": "1",
                "learning_rate": str(policy.learning_rate),
                "tau": "none" if policy.tau is None else str(policy.tau),
                "feature_count": str(policy.weights.shape[0]),
                "action_count": str(policy.weights.shape[1])}
    save_file({"weights": policy.weights, "trace": policy.trace}, str(path), metadata=metadata)
    restored = load_file(str(path))
    exact = np.array_equal(restored["weights"], policy.weights) and np.array_equal(restored["trace"], policy.trace)
    clone = LinearTracePolicy(policy.weights.shape[0], policy.weights.shape[1],
                              policy.learning_rate, policy.tau)
    clone.weights[:] = restored["weights"]
    clone.trace[:] = restored["trace"]
    probe = np.linspace(0., 1., policy.weights.shape[0], dtype=np.float64)
    prediction_exact = np.array_equal(clone.probabilities(probe), policy.probabilities(probe))
    event = gradient_event(probe, 0, policy.probabilities(probe))
    policy_next = policy.trace * policy.decay + event
    clone.advance_trace(event)
    next_trace_exact = np.array_equal(clone.trace, policy_next)
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "roundtrip_exact": bool(exact), "restore_prediction_exact": bool(prediction_exact),
            "next_trace_update_exact": bool(next_trace_exact), "metadata": metadata}


def impulse_diagnostics(x):
    """Event-order and cross-talk controls without rewards or learned labels."""
    p = np.full(4, .25)
    cue_event = gradient_event(x[0], 0, p)
    before_cue = LinearTracePolicy()
    before_weights = before_cue.weights.copy()
    before_cue.update(1., before_cue.trace)
    immediate = LinearTracePolicy(tau=8.)
    immediate.advance_trace(cue_event)
    plain_delta = immediate.learning_rate * cue_event
    matched_delta = immediate.learning_rate * norm_match(immediate.trace, cue_event)
    out = {"reward_before_cue_update_norm": float(np.linalg.norm(before_cue.weights - before_weights)),
           "single_impulse_immediate_normmatched_max_abs_difference": float(np.max(np.abs(plain_delta - matched_delta))),
           "single_impulse_plain_update_l2": float(np.linalg.norm(plain_delta)),
           "single_impulse_normmatched_update_l2": float(np.linalg.norm(matched_delta)),
           "tau": {}}
    for tau in (2., 8., 32.):
        decay = np.exp(-1. / tau)
        by_delay = {}
        for delay in (0, 4, 16):
            trace = cue_event.copy()
            for _ in range(delay):
                trace *= decay  # blank cue has zero event
            blank_ratio = np.linalg.norm(trace) / np.linalg.norm(cue_event)
            continuous = cue_event.copy()
            for j in range(delay):
                continuous = decay * continuous + gradient_event(x[1 + j % (len(x) - 1)], 0, p)
            projection = float(np.vdot(continuous, cue_event) / np.vdot(cue_event, cue_event))
            residual = continuous - projection * cue_event
            by_delay[str(delay)] = {"single_cue_blank_trace_ratio": float(blank_ratio),
                                    "analytic_decay": float(np.exp(-delay / tau)),
                                    "continuous_original_cue_projection": projection,
                                    "continuous_crosstalk_l2": float(np.linalg.norm(residual))}
        out["tau"][str(int(tau))] = by_delay
    return out


def main():
    start = time.perf_counter()
    data = load_data()
    x_train, y_train = data["train"]
    conditions = {0: ["current_only", "exact_cached_gradient", "trace_tau8",
                      "trace_tau8_normmatched", "zero_reward"],
                  4: ["current_only", "exact_cached_gradient", "trace_tau8",
                      "trace_tau8_normmatched", "zero_reward"],
                  16: ["current_only", "exact_cached_gradient", "trace_tau2", "trace_tau8",
                       "trace_tau8_normmatched", "trace_tau32", "trace_tau32_normmatched",
                       "zero_reward"]}
    runs = []
    for seed in (501, 502, 503):
        rng = np.random.default_rng(seed + 910_000)
        indices = rng.integers(len(x_train), size=2048 + 16)
        uniforms = rng.random(2048 + 16)
        for delay, names in conditions.items():
            for name in names:
                tau = (float(name.removeprefix("trace_tau").split("_")[0])
                       if name.startswith("trace_tau") else None)
                policy = LinearTracePolicy(tau=tau)
                training = train(policy, x_train, y_train, indices, uniforms, delay, name)
                test_first = score(policy, *data["test"])
                test_second = score(policy, *data["test"])
                runs.append({"seed": seed, "delay": delay, "condition": name,
                             "tau": tau, "training": training, "test": test_first,
                             "korean": score(policy, *data["korean"]),
                             "read_only_retention_exact": test_first == test_second,
                             "checkpoint": checkpoint(policy, f"seed-{seed}-lag-{delay}-{name}")})
    summary = {}
    for delay, names in conditions.items():
        summary[str(delay)] = {}
        for name in names:
            selected = [r for r in runs if r["delay"] == delay and r["condition"] == name]
            summary[str(delay)][name] = {
                split: {key: float(np.mean([r[split][key] for r in selected]))
                        for key in ("accuracy", "mean_correct_probability", "mean_entropy")}
                for split in ("test", "korean")}
    provenance_paths = [Path(__file__).resolve(), GEMMA / "dataset.json",
                        GEMMA / "embeddings.safetensors", ROOT / "data/pn_kc.tsv",
                        ROOT / "data/kc_mbon.tsv"]
    report = {"design": {"status": "original 42 settings predeclared; 12 norm-matched controls added post-hoc after initial results",
                          "scope": "conventional matched linear eligibility reference, not Novi implementation or exact fly biology",
                          "seeds": [501, 502, 503], "actions": 4, "identity_class_map": True,
                          "rewarded_actions": 2048, "reward": "+1 correct sampled action, -1 incorrect",
                          "learning_rate": .03, "features": "frozen 319-port signed embedding encoding with unit L2 norm",
                          "delays": [0, 4, 16], "trace_taus": {"main": 8, "delay16_controls": [2, 32]},
                          "shared_stream": "context indices and uniforms fixed per seed; sampled actions may diverge with policy",
                          "exact_oracle": "queues the historical policy-gradient event computed at action time",
                          "trace": "e_t=lambda*e_(t-1)+outer(x_t,onehot(action)-p_t); delayed scalar reward applies to current accumulated trace",
                          "norm_matched_trace": "post-hoc oracle diagnostic rescales current trace to the L2 norm of the corresponding historical cached gradient event; uses old-gradient norm and is not a deployment or biological mechanism",
                          "exposure": "one delivered +/-1 scalar per scheduled action; accumulating traces intentionally include cross-talk and are not norm-matched to cached events",
                          "label_access": "labels compute scalar environment reward only; learner receives no target action or future label",
                          "weights": "float64, no clipping; gradient and update norms reported",
                          "limitation": "cached-gradient replay is an oracle audit; eligibility is a conventional engineered trace, not a molecular claim. Both use gradients saved at event time while weights subsequently change."},
              "runs": runs, "mean_across_seeds": summary,
              "impulse_and_intervening_cue_controls": impulse_diagnostics(x_train),
              "provenance": {"base_git_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in provenance_paths}},
              "interpretation": [
                  "Current-only credit fell to chance with delayed reward, while the cached-gradient oracle retained most zero-delay performance.",
                  "The accumulating trace suffered cross-trial interference: even at delay zero tau=8 underperformed event-specific credit because prior cue gradients remained active.",
                  "At delay 16 every tested trace constant was near chance; longer traces became low-entropy and accumulated larger cross-talk rather than recovering oracle assignment.",
                  "Post-hoc norm matching brought trace update norms near cached-event update norms but did not recover performance; tau8 worsened at lags 0 and 4 and all lag-16 controls stayed at chance, implicating credit direction and cross-talk rather than magnitude alone.",
                  "For an isolated cue followed by blanks, trace magnitude exactly followed exp(-delay/tau); independent intervening cues added substantial residual trace, separating decay from continuous-stream interference.",
                  "Zero reward remained at the measured 25% frozen baseline, and repeated read-only scoring preserved weights and predictions exactly."
              ],
              "runtime_seconds": time.perf_counter() - start}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mean": summary, "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
