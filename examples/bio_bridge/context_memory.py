#!/usr/bin/env python3
"""Exploratory context-memory experiment on frozen real embedding features.

Context is an externally supplied, trusted task identifier. The context gates
below are engineered input/routing mechanisms, not learned context discovery,
biological compartments, or claims about a fly memory circuit.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
from novigrad import Engine
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results/gemma-bridge"
SEEDS = (401, 402, 403)
ARCHITECTURES = ("blind", "concat", "concat_normalized", "conjunctive", "modular")
SPLITS = ("test", "korean")


@dataclass(frozen=True)
class Config:
    embedding_dimensions: int = 64
    signed_ports: int = 128
    input_ports: int = 319
    actions: int = 4
    learning_rate: float = 0.3
    logit_gain: float = 1.0
    active_fraction: float = 0.2
    weight_limit: float = 1.0
    homeostasis: bool = False
    readout: str = "opponent"
    batch_size: int = 8
    acquisition_updates: int = 512
    reversal_updates: int = 512
    refresh_updates: int = 256
    diagnostic_interval: int = 8
    diagnostic_accuracy_threshold: float = 0.75


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(data_dir=DATA_DIR):
    rows = json.loads((data_dir / "dataset.json").read_text())
    embeddings = load_file(str(data_dir / "embeddings.safetensors"))["embeddings"]
    if len(rows) != len(embeddings):
        raise ValueError("dataset and embedding row counts differ")
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    masks = {split: np.asarray([row["split"] == split for row in rows]) for split in ("train", *SPLITS)}
    return embeddings, labels, masks


def signed_features(embeddings, config=Config()):
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < config.embedding_dimensions or not np.isfinite(x).all():
        raise ValueError("expected a finite embedding matrix with at least 64 dimensions")
    base = x[:, :config.embedding_dimensions].copy()
    base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    return np.concatenate((np.maximum(base, 0), np.maximum(-base, 0)), axis=1)


def encode_context(base, architecture, context, config=Config()):
    """Map signed features and an oracle A/B context to 319 engine ports."""
    if architecture not in ARCHITECTURES or context not in (0, 1):
        raise ValueError("unknown architecture or context")
    base = np.asarray(base, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != config.signed_ports:
        raise ValueError("base features must have shape [n, 128]")
    rates = np.zeros((len(base), config.input_ports), dtype=np.float32)
    if architecture in ("blind", "modular"):
        rates[:, :config.signed_ports] = base
    elif architecture in ("concat", "concat_normalized"):
        rates[:, :config.signed_ports] = base
        rates[:, config.signed_ports + context] = 1.0
        if architecture == "concat_normalized":
            rates /= np.maximum(np.linalg.norm(rates, axis=1, keepdims=True), 1e-12)
    else:
        start = context * config.signed_ports
        rates[:, start:start + config.signed_ports] = base
    return rates


def context_labels(labels, context):
    labels = np.asarray(labels, dtype=np.int64)
    if context not in (0, 1) or np.any((labels < 0) | (labels >= 4)):
        raise ValueError("labels/context out of range")
    return labels.copy() if context == 0 else (labels + 1) % 4


def new_engine(config=Config()):
    return Engine.from_edges(
        ROOT / "data/pn_kc.tsv", ROOT / "data/kc_mbon.tsv",
        actions=config.actions, learning_rate=config.learning_rate,
        logit_gain=config.logit_gain, active_fraction=config.active_fraction,
        weight_limit=config.weight_limit, homeostasis=config.homeostasis,
        readout=config.readout,
    )


class ContextPolicy:
    def __init__(self, architecture, config=Config()):
        if architecture not in ARCHITECTURES:
            raise ValueError("unknown architecture")
        self.architecture = architecture
        self.config = config
        self.engines = [new_engine(config) for _ in range(2 if architecture == "modular" else 1)]

    def _engine(self, context):
        return self.engines[context] if self.architecture == "modular" else self.engines[0]

    def rates(self, base, context):
        return encode_context(base, self.architecture, context, self.config)

    def predict(self, base, context):
        return np.asarray(self._engine(context).infer_batch(self.rates(base, context).tolist()), dtype=np.float64)

    def learn(self, base, context, actions, rewards):
        self._engine(context).learn(
            self.rates(base, context).tolist(), list(map(int, actions)), list(map(float, rewards))
        )


def score(policy, base, labels, context):
    target = context_labels(labels, context)
    probabilities = policy.predict(base, context)
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(np.mean(prediction == target)),
        "mean_correct_probability": float(probabilities[np.arange(len(target)), target].mean()),
    }


def train_phase(policy, base, labels, context, rng, updates, config=Config(), diagnostic_target=None):
    target = context_labels(labels, context)
    correct = 0
    reward_sum = 0.0
    first_target_update = None
    if diagnostic_target is not None and score(policy, base, labels, context)["accuracy"] >= diagnostic_target:
        first_target_update = 0
    start = time.perf_counter()
    for update in range(1, updates + 1):
        indices = rng.integers(0, len(base), size=config.batch_size)
        batch = base[indices]
        probabilities = policy.predict(batch, context)
        uniforms = rng.random(config.batch_size)
        actions = np.sum(uniforms[:, None] > np.cumsum(probabilities, axis=1), axis=1)
        actions = np.minimum(actions, config.actions - 1)
        rewards = np.where(actions == target[indices], 1.0, -1.0)
        policy.learn(batch, context, actions, rewards)
        correct += int(np.sum(actions == target[indices]))
        reward_sum += float(np.sum(rewards))
        if diagnostic_target is not None and first_target_update is None and update % config.diagnostic_interval == 0:
            if score(policy, base, labels, context)["accuracy"] >= diagnostic_target:
                first_target_update = update
    interactions = updates * config.batch_size
    return {
        "updates": updates, "interactions": interactions,
        "sampled_correct_actions": correct,
        "sampled_correct_fraction": correct / interactions,
        "mean_reward": reward_sum / interactions,
        "seconds": time.perf_counter() - start,
        "first_diagnostic_update_reaching_target": first_target_update,
    }


def frozen_probe(policy, features, labels, masks):
    return {
        f"context_{name}": {
            split: score(policy, features[masks[split]], labels[masks[split]], context)
            for split in SPLITS
        }
        for context, name in enumerate(("A_identity", "B_rotate_plus_1"))
    }


def save_checkpoints(policy, architecture, seed, output_dir):
    paths = []
    suffixes = ("A", "B") if architecture == "modular" else ("shared",)
    for engine, suffix in zip(policy.engines, suffixes):
        path = output_dir / f"context-memory-{architecture}-{seed}-{suffix}.safetensors"
        engine.save(path, overwrite=True)
        paths.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "config": engine.config})
    return paths


def run_architecture(architecture, seed, features, labels, masks, output_dir, config=Config()):
    policy = ContextPolicy(architecture, config)
    initial_weight_hashes = [hashlib.sha256(np.asarray(e.weights, np.float32).tobytes()).hexdigest() for e in policy.engines]
    rng = np.random.default_rng(seed)
    train_x, train_y = features[masks["train"]], labels[masks["train"]]
    run = {
        "architecture": architecture, "seed": seed,
        "initial_engine_weight_hashes": initial_weight_hashes,
        "initial": frozen_probe(policy, features, labels, masks),
    }
    run["acquisition_A"] = train_phase(
        policy, train_x, train_y, 0, rng, config.acquisition_updates, config,
        diagnostic_target=config.diagnostic_accuracy_threshold,
    )
    run["after_A"] = frozen_probe(policy, features, labels, masks)
    run["reversal_B"] = train_phase(policy, train_x, train_y, 1, rng, config.reversal_updates, config)
    run["after_B"] = frozen_probe(policy, features, labels, masks)
    # Re-attainment is measured on training data and never changes settings.
    # Held-out splits remain frozen phase-boundary probes only.
    run["refresh_A"] = train_phase(
        policy, train_x, train_y, 0, rng, config.refresh_updates, config,
        diagnostic_target=config.diagnostic_accuracy_threshold,
    )
    run["after_A_refresh"] = frozen_probe(policy, features, labels, masks)
    a_after = run["after_A"]["context_A_identity"]["test"]["accuracy"]
    a_retained = run["after_B"]["context_A_identity"]["test"]["accuracy"]
    b_before = run["after_A"]["context_B_rotate_plus_1"]["test"]["accuracy"]
    b_after = run["after_B"]["context_B_rotate_plus_1"]["test"]["accuracy"]
    run["descriptive_memory"] = {
        "A_test_accuracy_change_after_B": a_retained - a_after,
        "A_test_retention_ratio": None if a_after == 0 else a_retained / a_after,
        "B_test_reversal_gain": b_after - b_before,
        "A_test_refresh_gain": run["after_A_refresh"]["context_A_identity"]["test"]["accuracy"] - a_retained,
        "training_diagnostic_accuracy_threshold": config.diagnostic_accuracy_threshold,
        "acquisition_updates_to_threshold_or_none": run["acquisition_A"]["first_diagnostic_update_reaching_target"],
        "refresh_updates_to_training_target_or_none": run["refresh_A"]["first_diagnostic_update_reaching_target"],
    }
    acquisition_count = run["acquisition_A"]["first_diagnostic_update_reaching_target"]
    refresh_count = run["refresh_A"]["first_diagnostic_update_reaching_target"]
    run["descriptive_memory"]["savings_updates_or_none"] = (
        acquisition_count - refresh_count
        if acquisition_count is not None and refresh_count is not None else None
    )
    run["checkpoints"] = save_checkpoints(policy, architecture, seed, output_dir)
    return run


def aggregate_runs(runs, architectures=ARCHITECTURES):
    aggregate = {}
    for architecture in architectures:
        selected = [run for run in runs if run["architecture"] == architecture]
        if not selected:
            continue
        aggregate[architecture] = {
            phase: {
                context: {
                    split: {
                        "mean_accuracy": float(np.mean([run[phase][context][split]["accuracy"] for run in selected])),
                        "values_by_seed": [run[phase][context][split]["accuracy"] for run in selected],
                    }
                    for split in SPLITS
                }
                for context in ("context_A_identity", "context_B_rotate_plus_1")
            }
            for phase in ("after_A", "after_B", "after_A_refresh")
        }
        aggregate[architecture]["descriptive_counts"] = {
            name: [run["descriptive_memory"][name] for run in selected]
            for name in (
                "acquisition_updates_to_threshold_or_none",
                "refresh_updates_to_training_target_or_none",
                "savings_updates_or_none",
            )
        }
    return aggregate


def build_report(output, config=Config()):
    embeddings, labels, masks = load_data()
    features = signed_features(embeddings, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            run = run_architecture(architecture, seed, features, labels, masks, output.parent, config)
            runs.append(run)
            print(seed, architecture, run["after_B"]["context_A_identity"]["test"]["accuracy"],
                  run["after_B"]["context_B_rotate_plus_1"]["test"]["accuracy"], flush=True)
    base_weights = len(ContextPolicy("blind", config).engines[0].weights)
    aggregate = aggregate_runs(runs)
    report = {
        "status": "exploratory reuse of the original test and Korean splits; no tuning or model selection",
        "architecture_list": list(ARCHITECTURES),
        "context_boundary": "A/B is an externally supplied oracle task context; the system does not infer context",
        "mechanism_boundary": "input gating and engine routing are engineered and are not a biological compartment model",
        "protocol": {**asdict(config), "seeds": list(SEEDS), "phase_order": ["A acquisition", "B reversed mapping", "A refresh"],
                     "reward": "+1 only for the actually sampled correct action, otherwise -1; no teacher action update",
                     "original_four_settings_frozen_before_initial_evaluation": True,
                     "normalized_control_added_after_initial_results": True,
                     "normalized_control_choices_fixed_before_control_run": True,
                     "normalized_control_preregistered": False,
                     "normalized_control_timing": "exact unit-L2 normalization, unchanged seeds, parameters, and schedule were fixed before the control run, but the control was not preregistered"},
        "features": {"source": "results/gemma-bridge/embeddings.safetensors", "sha256": sha256(DATA_DIR / "embeddings.safetensors"),
                     "dataset_sha256": sha256(DATA_DIR / "dataset.json"), "encoder_frozen": True},
        "parameter_comparison": {
            "blind": {"engines": 1, "plastic_weights": base_weights, "feature_ports_used": 128},
            "concat": {"engines": 1, "plastic_weights": base_weights, "feature_ports_used": 130},
            "concat_normalized": {"engines": 1, "plastic_weights": base_weights, "feature_ports_used": 130,
                                  "input_row_l2_norm": 1.0,
                                  "note": "post-hoc exploratory control added after observing the original concat result; normalization strength was fixed exactly at unit norm"},
            "conjunctive": {"engines": 1, "plastic_weights": base_weights, "feature_ports_used": 256},
            "modular": {"engines": 2, "plastic_weights": 2 * base_weights, "feature_ports_used_per_engine": 128,
                        "note": "twice the engine parameters; trusted context routes to separate engines"},
        },
        "initialization": "all engines load the same frozen pn_kc.tsv and kc_mbon.tsv connectome; modular A/B engines begin as identical copies",
        "comparison_boundary": "architectures change port routing as well as context availability; results do not identify a causal effect of context alone",
        "aggregate": aggregate, "runs": runs,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def append_normalized_control(output, config=Config()):
    """Append the post-hoc normalized concat control without rerunning old rows."""
    report = json.loads(output.read_text())
    old_runs = report["runs"]
    if any(run["architecture"] == "concat_normalized" for run in old_runs):
        raise ValueError("concat_normalized control already exists")
    embeddings, labels, masks = load_data()
    features = signed_features(embeddings, config)
    new_runs = []
    for seed in SEEDS:
        run = run_architecture("concat_normalized", seed, features, labels, masks, output.parent, config)
        new_runs.append(run)
        print(seed, "concat_normalized", run["after_B"]["context_A_identity"]["test"]["accuracy"],
              run["after_B"]["context_B_rotate_plus_1"]["test"]["accuracy"], flush=True)
    report["runs"] = old_runs + new_runs
    report["architecture_list"] = list(ARCHITECTURES)
    report["aggregate"] = aggregate_runs(report["runs"])
    base_weights = report["parameter_comparison"]["blind"]["plastic_weights"]
    report["parameter_comparison"]["concat_normalized"] = {
        "engines": 1, "plastic_weights": base_weights, "feature_ports_used": 130,
        "input_row_l2_norm": 1.0,
        "note": "post-hoc exploratory control added after observing the original concat result; normalization strength was fixed exactly at unit norm",
    }
    report["comparison_boundary"] = "architectures change port routing as well as context availability; results do not identify a causal effect of context alone"
    report["status"] = "exploratory reuse of the original test and Korean splits; normalized concat is post-hoc after observing original concat; no normalization tuning"
    report["protocol"].pop("all_settings_frozen_before_evaluation", None)
    report["protocol"].update({
        "original_four_settings_frozen_before_initial_evaluation": True,
        "normalized_control_added_after_initial_results": True,
        "normalized_control_choices_fixed_before_control_run": True,
        "normalized_control_preregistered": False,
        "normalized_control_timing": "exact unit-L2 normalization, unchanged seeds, parameters, and schedule were fixed before the control run, but the control was not preregistered",
    })
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/bio-bridge/context-memory.json")
    parser.add_argument("--append-normalized-control", action="store_true")
    args = parser.parse_args()
    if args.append_normalized_control:
        append_normalized_control(args.output)
    else:
        build_report(args.output)


if __name__ == "__main__":
    main()
