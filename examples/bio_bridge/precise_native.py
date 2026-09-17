#!/usr/bin/env python3
"""Native reward-policy training for frozen embedding-to-port mappings.

Unlike the semantic ridge probes, this script updates native KC-to-MBON policy
weights using rewards from actually sampled actions. Port mappings are fit on
training embeddings without labels and frozen before validation/final use.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from novigrad import Engine
from safetensors.numpy import load_file

from delayed_credit import ROOT
from precise_bridge import MODES, OUT, PortBridge, dataset


SEEDS = (701, 702, 703)
EPOCHS = 64
BATCH_SIZE = 8


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_engine():
    return Engine.from_edges(
        ROOT / "data/pn_kc.tsv", ROOT / "data/kc_mbon.tsv", actions=4,
        learning_rate=0.3, logit_gain=1.0, active_fraction=0.02,
        homeostasis=False, readout="opponent",
    )


def training_stream(seed, n):
    rng = np.random.default_rng(seed + 910_000)
    indices = np.concatenate([rng.permutation(n) for _ in range(EPOCHS)])
    uniforms = np.random.default_rng(seed + 920_000).random(len(indices))
    return indices, uniforms


def sample_actions(probabilities, uniforms):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(uniforms):
        raise ValueError("probability rows and uniforms must align")
    return np.minimum(probabilities.shape[1] - 1,
                      np.sum(uniforms[:, None] > np.cumsum(probabilities, axis=1), axis=1)).astype(np.int64)


def evaluate(engine, rates, labels, languages=None):
    probability = np.asarray(engine.infer_batch(rates.tolist()), dtype=np.float64)
    prediction = probability.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=int)
    for truth, predicted in zip(labels, prediction):
        confusion[int(truth), int(predicted)] += 1
    result = {
        "accuracy": float(np.mean(prediction == labels)),
        "mean_correct_probability": float(probability[np.arange(len(labels)), labels].mean()),
        "confusion": confusion.tolist(), "predictions": prediction.tolist(),
    }
    if languages is not None:
        result["accuracy_by_language"] = {
            language: float(np.mean(prediction[languages == language] == labels[languages == language]))
            for language in sorted(set(languages.tolist())) if np.any(languages == language)
        }
    return result


def train(engine, rates, labels, indices, uniforms):
    correct = 0
    for start in range(0, len(indices), BATCH_SIZE):
        chosen = indices[start:start + BATCH_SIZE]
        batch = rates[chosen]
        probability = np.asarray(engine.infer_batch(batch.tolist()), dtype=np.float64)
        actions = sample_actions(probability, uniforms[start:start + len(chosen)])
        rewards = np.where(actions == labels[chosen], 1.0, -1.0)
        correct += int(np.sum(actions == labels[chosen]))
        engine.learn(batch.tolist(), actions.tolist(), rewards.tolist())
    return {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "optimizer_updates": len(indices) // BATCH_SIZE,
            "interactions": len(indices), "sampled_correct_actions": correct,
            "sampled_correct_fraction": correct / len(indices),
            "mean_reward": (2 * correct - len(indices)) / len(indices)}


def artifact_record(path, kind, extra=None):
    try:
        stored_path = str(path.relative_to(ROOT))
    except ValueError:
        stored_path = str(path.resolve())
    record = {"kind": kind, "path": stored_path, "sha256": sha256(path)}
    if extra:
        record.update(extra)
    return record


def run_development(out_dir=OUT):
    out_dir = Path(out_dir)
    output = out_dir / "native-development.json"
    checkpoint_dir = out_dir / "native-checkpoints"
    data = dataset()
    train_embeddings, train_labels, _ = data["train"]
    validation_embeddings, validation_labels, _ = data["validation"]
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    mappings = {}
    for mode in MODES:
        bridge = PortBridge.fit(train_embeddings, mode)
        mapping_path = out_dir / f"native-ports-{mode}.safetensors"
        bridge.save(mapping_path)
        restored_bridge = PortBridge.load(mapping_path, mode)
        train_rates = bridge.encode(train_embeddings)
        validation_rates = bridge.encode(validation_embeddings)
        if not np.array_equal(train_rates, restored_bridge.encode(train_embeddings)):
            raise AssertionError("mapping roundtrip changed training rates")
        mappings[mode] = artifact_record(mapping_path, "embedding_to_port_mapping", {"mode": mode})
        for seed in SEEDS:
            indices, uniforms = training_stream(seed, len(train_rates))
            engine = new_engine()
            before_weights = engine.weights
            initial = evaluate(engine, validation_rates, validation_labels)
            training = train(engine, train_rates, train_labels, indices, uniforms)
            trained = evaluate(engine, validation_rates, validation_labels)
            path = checkpoint_dir / f"{mode}-seed-{seed}.safetensors"
            engine.save(path, overwrite=True)
            restored = Engine.load(path)
            all_rates = np.concatenate((train_rates, validation_rates))
            exact = engine.infer_batch(all_rates.tolist()) == restored.infer_batch(all_rates.tolist())
            if not exact:
                raise AssertionError("checkpoint roundtrip inference changed")
            stream_hash = hashlib.sha256(indices.tobytes() + uniforms.tobytes()).hexdigest()
            runs.append({
                "mode": mode, "seed": seed, "initial_validation": initial,
                "training": training, "trained_validation": trained,
                "weights_changed": before_weights != engine.weights,
                "stream_sha256": stream_hash,
                "checkpoint": artifact_record(path, "native_engine_checkpoint",
                                              {"config": engine.config,
                                               "roundtrip_inference_exact": exact}),
            })
    aggregate = {
        mode: {
            phase: {
                metric: float(np.mean([run[phase][metric] for run in runs if run["mode"] == mode]))
                for metric in ("accuracy", "mean_correct_probability")
            }
            for phase in ("initial_validation", "trained_validation")
        }
        for mode in MODES
    }
    immutable_inputs = [
        ROOT / "results/gemma-bridge/dataset.json",
        ROOT / "results/gemma-bridge/embeddings.safetensors",
        ROOT / "data/pn_kc.tsv", ROOT / "data/kc_mbon.tsv",
        Path(__file__), Path(__file__).with_name("precise_bridge.py"),
    ]
    report = {
        "stage": "development",
        "protocol": {
            "status": "all mappings, parameters, seeds, and exposure budgets fixed before validation results",
            "modes": list(MODES), "seeds": list(SEEDS), "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "interactions_per_run": EPOCHS * len(train_rates),
            "optimizer_updates_per_run": EPOCHS * len(train_rates) // BATCH_SIZE,
            "engine": {"learning_rate": 0.3, "logit_gain": 1.0,
                       "active_fraction": 0.02, "homeostasis": False,
                       "readout": "opponent"},
            "feedback": "+1 only when the sampled action equals the label, otherwise -1; no teacher-action update",
            "mapping_boundary": "PortBridge.fit receives training embeddings only and no feature/class labels",
            "stream_pairing": "epoch permutations and action uniforms are identical across mappings within seed; sampled actions can diverge",
            "model_boundary": "this trains native KC-to-MBON policy weights, unlike the separate supervised semantic ridge probes",
            "validation": "12-item development split; no final holdout read",
        },
        "mappings": mappings, "runs": runs, "aggregate": aggregate,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in immutable_inputs},
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_development_manifest(report):
    for relative, expected in report["input_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise AssertionError(f"development input hash changed: {relative}")
    for record in report["mappings"].values():
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise AssertionError("saved mapping hash changed")
    for run in report["runs"]:
        record = run["checkpoint"]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise AssertionError("saved checkpoint hash changed")


def run_final(out_dir=OUT):
    out_dir = Path(out_dir)
    output = out_dir / "native-final.json"
    development_path = out_dir / "native-development.json"
    development_report = json.loads(development_path.read_text())
    verify_development_manifest(development_report)
    case_path = ROOT / "examples/bio_bridge/precise_holdout.json"
    feature_path = out_dir / "holdout-embeddings.safetensors"
    rows = json.loads(case_path.read_text())
    if isinstance(rows, dict):
        rows = rows["cases"]
    embeddings = load_file(str(feature_path))["embeddings"]
    goals = ("water", "food", "warmth", "rest")
    labels = np.asarray([goals.index(row["class"]) for row in rows], dtype=np.int64)
    languages = np.asarray([row["language"] for row in rows])
    if len(rows) != len(embeddings):
        raise ValueError("holdout cases/features differ in length")
    runs = []
    for development_run in development_report["runs"]:
        mode, seed = development_run["mode"], development_run["seed"]
        bridge = PortBridge.load(ROOT / development_report["mappings"][mode]["path"], mode)
        engine = Engine.load(ROOT / development_run["checkpoint"]["path"])
        before = engine.weights
        result = evaluate(engine, bridge.encode(embeddings), labels, languages)
        if before != engine.weights:
            raise AssertionError("final evaluation changed native weights")
        runs.append({"mode": mode, "seed": seed, "holdout": result})
    aggregate = {
        mode: {
            metric: float(np.mean([run["holdout"][metric] for run in runs if run["mode"] == mode]))
            for metric in ("accuracy", "mean_correct_probability")
        }
        for mode in MODES
    }
    report = {
        "stage": "final",
        "boundary": "single fresh authored bilingual holdout evaluation after mappings, hyperparameters, streams, and checkpoints were frozen; no selection or retraining",
        "development_report_sha256": sha256(development_path),
        "holdout_sha256": {str(case_path.relative_to(ROOT)): sha256(case_path),
                           str(feature_path.relative_to(ROOT)): sha256(feature_path)},
        "manifest_hashes_verified": True, "runs": runs, "aggregate": aggregate,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _tensor_file_equal(left, right):
    from safetensors import safe_open
    with safe_open(str(left), framework="numpy") as a, safe_open(str(right), framework="numpy") as b:
        if a.metadata() != b.metadata() or a.keys() != b.keys():
            return False
        return all(np.array_equal(a.get_tensor(key), b.get_tensor(key)) for key in a.keys())


def replay(out_dir=OUT):
    """Retrain in isolation and require exact mapping/checkpoint/metric parity."""
    out_dir = Path(out_dir)
    original = json.loads((out_dir / "native-development.json").read_text())
    with tempfile.TemporaryDirectory(prefix="novi-native-replay-") as temporary:
        temporary = Path(temporary)
        reproduced = run_development(temporary)
        mapping_exact = {
            mode: _tensor_file_equal(ROOT / original["mappings"][mode]["path"],
                                     temporary / f"native-ports-{mode}.safetensors")
            for mode in MODES
        }
        checkpoint_exact = {}
        metrics_exact = {}
        for old, new in zip(original["runs"], reproduced["runs"]):
            key = f"{old['mode']}-seed-{old['seed']}"
            checkpoint_exact[key] = _tensor_file_equal(
                ROOT / old["checkpoint"]["path"],
                temporary / "native-checkpoints" / f"{key}.safetensors",
            )
            metrics_exact[key] = all(old[field] == new[field] for field in (
                "initial_validation", "training", "trained_validation", "weights_changed", "stream_sha256"
            ))
        passed = all(mapping_exact.values()) and all(checkpoint_exact.values()) and all(metrics_exact.values())
        if not passed:
            raise AssertionError("native development replay differs")
    record = {"status": "exact isolated CPU retraining replay", "passed": passed,
              "mapping_tensors_exact": mapping_exact,
              "checkpoint_tensors_exact": checkpoint_exact,
              "metrics_and_predictions_exact": metrics_exact,
              "original_development_sha256": sha256(out_dir / "native-development.json")}
    (out_dir / "native-replay.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--development", action="store_true")
    stage.add_argument("--final", action="store_true")
    stage.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.development:
        report = run_development()
    elif args.final:
        report = run_final()
    else:
        report = replay()
    print(json.dumps({"stage": report.get("stage", "replay"),
                      "aggregate": report.get("aggregate"), "passed": report.get("passed")}, indent=2))


if __name__ == "__main__":
    main()
