#!/usr/bin/env python3
"""Frozen APL-inspired hidden-inhibition mechanism comparison.

These are engineered rate-code transformations over a connectome-constrained
model. Global and local inhibition are functional proxies, not reconstructions
of APL anatomy, physiology, or compartments.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from delayed_credit import semantic_ports
from novigrad import Engine


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/mechanism-bridge/inhibition.json"
SEEDS = (601, 602, 603)
VARIANTS = ("native_topk_2", "topk_20", "subtractive_global", "subtractive_local")
MIXTURE_RATIOS = (0.0, 0.25, 1.0)
LOCAL_GROUPS = 32


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(seed):
    return ROOT / f"results/bio-bridge/sparse-d128-f0.02-s{seed}.safetensors"


def load_inputs():
    old_dir = ROOT / "results/gemma-bridge"
    rows = json.loads((old_dir / "dataset.json").read_text())
    old = semantic_ports(load_file(str(old_dir / "embeddings.safetensors"))["embeddings"])
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    masks = {name: np.asarray([row["split"] == source for row in rows])
             for name, source in (("old_train", "train"), ("old_calibration", "validation"))}
    case_path = ROOT / "examples/bio_bridge/sparsity_confirmation.json"
    feature_path = ROOT / "results/bio-bridge/confirmation-features.safetensors"
    cases = json.loads(case_path.read_text())
    confirmation = semantic_ports(load_file(str(feature_path))["embeddings"])
    confirmation_labels = np.asarray([
        ("water", "food", "warmth", "rest").index(row["class"]) for row in cases
    ], dtype=np.int64)
    datasets = {
        "old_train": (old[masks["old_train"]], labels[masks["old_train"]]),
        "old_calibration": (old[masks["old_calibration"]], labels[masks["old_calibration"]]),
        "reused_confirmation_32": (confirmation, confirmation_labels),
    }
    provenance = {str(path.relative_to(ROOT)): sha256(path) for path in (
        old_dir / "dataset.json", old_dir / "embeddings.safetensors", case_path, feature_path
    )}
    return datasets, provenance


class ShadowEngine:
    """SciPy sparse/NumPy forward model reconstructed from schema-4 tensors."""

    def __init__(self, path):
        with safe_open(str(path), framework="numpy") as model:
            self.metadata = dict(model.metadata())
            tensors = {key: model.get_tensor(key) for key in model.keys()}
        if self.metadata.get("format") != "nobi.plastic" or self.metadata.get("version") != "4":
            raise ValueError("expected nobi.plastic schema 4")
        self.input_count = len(tensors["input_ids"])
        self.hidden_count = len(tensors["hidden_ids"])
        self.output_count = len(tensors["output_ids"])
        input_pre = tensors["input_pre"].astype(np.int64)
        input_post = tensors["input_post"].astype(np.int64)
        counts = tensors["input_count"].astype(np.float64)
        totals = np.bincount(input_post, weights=counts, minlength=self.hidden_count)
        fixed = (counts / totals[input_post]).astype(np.float32) * tensors["input_sign"]
        self.input_matrix = sparse.csr_matrix(
            (fixed, (input_post, input_pre)), shape=(self.hidden_count, self.input_count), dtype=np.float32
        )
        self.plastic_matrix = sparse.csr_matrix(
            (tensors["plastic_weight"],
             (tensors["plastic_post"].astype(np.int64), tensors["plastic_pre"].astype(np.int64))),
            shape=(self.output_count, self.hidden_count), dtype=np.float32,
        )
        self.output_actions = tensors["output_actions"].astype(np.int64)
        self.output_gains = tensors["output_gains"].astype(np.float32)
        self.actions = int(self.metadata["actions"])
        self.logit_gain = np.float32(self.metadata["logit_gain"])
        self.group_counts = np.bincount(self.output_actions, minlength=self.actions).astype(np.float32)

    def raw_hidden(self, rows):
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != self.input_count or not np.isfinite(rows).all():
            raise ValueError("expected finite [rows, input_ports]")
        scale = np.maximum(np.max(np.abs(rows), axis=1, keepdims=True), 1.0)
        scale = np.where(np.max(np.abs(rows), axis=1, keepdims=True) > 16.0, scale, 1.0)
        hidden = np.asarray(self.input_matrix @ (rows / scale).T, dtype=np.float32).T
        np.maximum(hidden, 0, out=hidden)
        return hidden

    @staticmethod
    def _topk(hidden, fraction):
        output = hidden.copy()
        keep = max(1, min(output.shape[1], int(np.ceil(np.float32(output.shape[1]) * np.float32(fraction)))))
        for row in output:
            ranking = np.argsort(-row, kind="stable")
            row[ranking[keep:]] = 0.0
        return output

    @staticmethod
    def _normalize(hidden):
        maximum = hidden.max(axis=1, keepdims=True)
        return np.divide(hidden, maximum, out=np.zeros_like(hidden), where=maximum > 0)

    def inhibit(self, raw, variant, calibration):
        if variant == "native_topk_2":
            hidden = self._topk(raw, 0.02)
        elif variant == "topk_20":
            hidden = self._topk(raw, 0.20)
        elif variant == "subtractive_global":
            mean = raw.mean(axis=1, keepdims=True)
            hidden = np.maximum(raw - calibration["global_gain"] * mean, 0.0)
        elif variant == "subtractive_local":
            groups = np.arange(self.hidden_count) % LOCAL_GROUPS
            means = np.stack([raw[:, groups == group].mean(axis=1) for group in range(LOCAL_GROUPS)], axis=1)
            hidden = np.maximum(raw - calibration["local_gain"] * means[:, groups], 0.0)
        else:
            raise ValueError("unknown inhibition variant")
        return self._normalize(hidden)

    def probabilities(self, hidden):
        output = np.asarray(self.plastic_matrix @ hidden.T, dtype=np.float32).T
        weighted = output * self.output_gains
        logits = np.zeros((len(hidden), self.actions), dtype=np.float32)
        for action in range(self.actions):
            logits[:, action] = weighted[:, self.output_actions == action].sum(axis=1)
        logits *= self.logit_gain / self.group_counts
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def forward(self, rows, variant, calibration):
        hidden = self.inhibit(self.raw_hidden(rows), variant, calibration)
        return self.probabilities(hidden), hidden


def _gain_for_budget(raw, local=False, target=0.02):
    """Choose feedback gain on training activity only, nearest to target budget."""
    if local:
        groups = np.arange(raw.shape[1]) % LOCAL_GROUPS
        means = np.stack([raw[:, groups == group].mean(axis=1) for group in range(LOCAL_GROUPS)], axis=1)
        denominator = means[:, groups]
    else:
        denominator = raw.mean(axis=1, keepdims=True)
    ratios = np.divide(raw, denominator, out=np.zeros_like(raw), where=denominator > 0)
    candidates = np.unique(ratios)
    desired = target * raw.size
    active = np.searchsorted(np.sort(ratios.ravel()), candidates, side="right")
    remaining = raw.size - active
    gain = float(candidates[np.argmin(np.abs(remaining - desired))])
    achieved = float(np.mean(raw > gain * denominator))
    return gain, achieved


def calibrate(shadow, train_rows):
    raw = shadow.raw_hidden(train_rows)
    global_gain, global_fraction = _gain_for_budget(raw, False)
    local_gain, local_fraction = _gain_for_budget(raw, True)
    return {"global_gain": global_gain, "local_gain": local_gain,
            "global_train_active_fraction": global_fraction,
            "local_train_active_fraction": local_fraction,
            "target_active_fraction": 0.02}


def paired_distractors(rows, labels):
    result = []
    for index, label in enumerate(labels):
        candidates = np.flatnonzero(labels == (label + 1) % 4)
        if not len(candidates):
            raise ValueError("every class needs its next-class distractor")
        result.append(rows[candidates[index % len(candidates)]])
    return np.asarray(result, dtype=np.float32)


def row_cosine(a, b):
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a * b, axis=1), denominator, out=np.zeros(len(a)), where=denominator > 0)


def evaluate_variant(shadow, rows, labels, ratio, variant, calibration):
    distractors = paired_distractors(rows, labels)
    mixed = rows + np.float32(ratio) * distractors
    mixed /= np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-12)
    probabilities, hidden = shadow.forward(mixed, variant, calibration)
    _, clean_hidden = shadow.forward(rows, variant, calibration)
    _, distractor_hidden = shadow.forward(distractors, variant, calibration)
    prediction = probabilities.argmax(axis=1)
    correct = probabilities[np.arange(len(labels)), labels]
    distractor_labels = (labels + 1) % 4
    distractor_probability = probabilities[np.arange(len(labels)), distractor_labels]
    input_cosine = row_cosine(rows, distractors)
    hidden_cosine = row_cosine(clean_hidden, distractor_hidden)
    active = np.count_nonzero(hidden, axis=1)
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "mean_correct_probability": float(np.mean(correct)),
        "mean_active_count": float(np.mean(active)),
        "mean_active_fraction": float(np.mean(active / shadow.hidden_count)),
        "mean_hidden_cosine_to_clean": float(np.mean(row_cosine(hidden, clean_hidden))),
        "mean_input_next_class_cosine": float(np.mean(input_cosine)),
        "mean_hidden_next_class_cosine": float(np.mean(hidden_cosine)),
        "mean_pattern_separation_delta": float(np.mean(input_cosine - hidden_cosine)),
        "paired_class_mean_correct_minus_distractor_probability": float(np.mean(correct - distractor_probability)),
        "paired_class_correct_exceeds_distractor_fraction": float(np.mean(correct > distractor_probability)),
    }


def l2_rows(rows):
    rows = np.asarray(rows, dtype=np.float64)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized = np.divide(rows, norm, out=np.zeros_like(rows), where=norm > 0)
    return np.ascontiguousarray(normalized)


def fit_dual_ridge(train_hidden, train_labels, ridge=0.1, actions=4):
    train = l2_rows(train_hidden)
    labels = np.asarray(train_labels, dtype=np.int64)
    if train.ndim != 2 or len(labels) != len(train):
        raise ValueError("incompatible training hidden array or labels")
    if ridge <= 0 or np.any((labels < 0) | (labels >= actions)):
        raise ValueError("ridge must be positive and labels in range")
    targets = np.eye(actions, dtype=np.float64)[labels]
    coefficients = np.linalg.solve(train @ train.T + ridge * np.eye(len(train)), targets)
    return train, coefficients


def predict_dual_ridge(train_basis, coefficients, test_hidden):
    test = l2_rows(test_hidden)
    if train_basis.ndim != 2 or coefficients.ndim != 2 or test.shape[1] != train_basis.shape[1] or coefficients.shape[0] != len(train_basis):
        raise ValueError("incompatible saved probe parameters or test hidden array")
    return test @ train_basis.T @ coefficients


def dual_ridge_scores(train_hidden, train_labels, test_hidden, ridge=0.1, actions=4):
    """Equal supervised linear probe using the n_train x n_train dual kernel."""
    basis, coefficients = fit_dual_ridge(train_hidden, train_labels, ridge, actions)
    return predict_dual_ridge(basis, coefficients, test_hidden)


def _probe_metrics(scores, labels):
    prediction = scores.argmax(axis=1)
    correct = scores[np.arange(len(labels)), labels]
    distractor = scores[np.arange(len(labels)), (labels + 1) % 4]
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "mean_correct_coefficient_score": float(np.mean(correct)),
        "mean_correct_minus_next_class_score": float(np.mean(correct - distractor)),
        "correct_exceeds_next_class_fraction": float(np.mean(correct > distractor)),
    }


def matched_readout_control(report):
    """Post-hoc equal-readout probe; does not modify original metrics."""
    datasets, _ = load_inputs()
    shadow = ShadowEngine(checkpoint(SEEDS[0]))
    calibration = calibrate(shadow, datasets["old_train"][0])
    train_rows, train_labels = datasets["old_train"]
    test_rows, test_labels = datasets["reused_confirmation_32"]
    train_raw = shadow.raw_hidden(train_rows)
    test_distractors = paired_distractors(test_rows, test_labels)
    variants = {}
    probe_checkpoints = {}
    for variant in VARIANTS:
        train_hidden = shadow.inhibit(train_raw, variant, calibration)
        train_basis, coefficients = fit_dual_ridge(train_hidden, train_labels)
        probe_path = OUT.parent / f"inhibition-ridge-{variant}.safetensors"
        metadata = {"variant": variant, "ridge_lambda": "0.1", "format": "novi.supervised-ridge-probe",
                    "status": "posthoc_exploratory", "score_semantics": "uncalibrated_coefficient_score"}
        save_file({"train_hidden_basis": train_basis, "dual_coefficients": coefficients,
                   "train_labels": train_labels.astype(np.int64)}, str(probe_path), metadata=metadata)
        loaded = load_file(str(probe_path))
        memory_scores = predict_dual_ridge(train_basis, coefficients, train_hidden)
        restored_scores = predict_dual_ridge(loaded["train_hidden_basis"], loaded["dual_coefficients"], train_hidden)
        roundtrip_error = float(np.max(np.abs(memory_scores - restored_scores)))
        if roundtrip_error != 0.0:
            raise AssertionError(f"ridge probe roundtrip failed for {variant}: {roundtrip_error}")
        probe_checkpoints[variant] = {
            "path": str(probe_path.relative_to(ROOT)), "sha256": sha256(probe_path),
            "metadata": metadata, "roundtrip_max_absolute_score_error": roundtrip_error,
        }
        variant_result = {
            "train_resubstitution": _probe_metrics(memory_scores, train_labels),
            "mean_train_effective_active_count": float(np.count_nonzero(train_hidden, axis=1).mean()),
            "mixtures": {},
        }
        for ratio in MIXTURE_RATIOS:
            mixed = test_rows + np.float32(ratio) * test_distractors
            mixed /= np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-12)
            test_hidden = shadow.inhibit(shadow.raw_hidden(mixed), variant, calibration)
            scores = predict_dual_ridge(train_basis, coefficients, test_hidden)
            variant_result["mixtures"][str(ratio)] = {
                **_probe_metrics(scores, test_labels),
                "mean_effective_active_count": float(np.count_nonzero(test_hidden, axis=1).mean()),
            }
        variants[variant] = variant_result

    encoder_hashes = []
    for seed in SEEDS:
        candidate = ShadowEngine(checkpoint(seed)).input_matrix
        digest = hashlib.sha256(candidate.indptr.tobytes() + candidate.indices.tobytes() + candidate.data.tobytes()).hexdigest()
        encoder_hashes.append(digest)
    if len(set(encoder_hashes)) != 1:
        raise AssertionError("checkpoint input encoders differ")
    report["posthoc_matched_readout"] = {
        "status": "added after inspecting the frozen-readout mechanism results; exploratory and not preregistered",
        "purpose": "distinguish hidden representation effects from mismatch to MBON weights trained under native 2% top-k",
        "method": "same supervised dual-kernel ridge linear head for every hidden variant; per-row hidden L2 normalization; lambda fixed at 0.1 without tuning",
        "boundary": "supervised probe on the same 32 old training labels; not Novi reward learning, not a production policy, and coefficient scores are not probabilities or confidence",
        "feature_count": shadow.hidden_count, "training_labels": len(train_labels), "ridge_lambda": 0.1,
        "capacity_caveat": "all probes receive 5177 feature coordinates and 32 labels, but variants retain unequal effective active counts",
        "encoder_replicates": "PN-to-KC tensors and hidden features are identical across seeds 601/602/603; computed once, not treated as three independent encoder results",
        "shared_input_encoder_sha256": encoder_hashes[0],
        "source_sha256": sha256(__file__),
        "probe_checkpoints": probe_checkpoints,
        "variants": variants,
    }
    return report


def append_matched_readout(output=OUT):
    before_hash = sha256(output)
    report = json.loads(output.read_text())
    original_runs = report["runs"]
    original_aggregate = report["aggregate"]
    if "posthoc_matched_readout" in report:
        raise ValueError("matched-readout control already exists")
    matched_readout_control(report)
    report["posthoc_matched_readout"]["pre_append_report_sha256"] = before_hash
    assert report["runs"] == original_runs and report["aggregate"] == original_aggregate
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def run(output=OUT):
    datasets, provenance = load_inputs()
    runs = []
    parity = []
    calibrations = {}
    for seed in SEEDS:
        path = checkpoint(seed)
        shadow = ShadowEngine(path)
        calibration = calibrate(shadow, datasets["old_train"][0])
        calibrations[str(seed)] = calibration
        native = Engine.load(path)
        parity_rows = np.concatenate([datasets["old_train"][0][:4], datasets["reused_confirmation_32"][0][:4]])
        native_probability = np.asarray(native.infer_batch(parity_rows.tolist()))
        shadow_probability, _ = shadow.forward(parity_rows, "native_topk_2", calibration)
        maximum_error = float(np.max(np.abs(native_probability - shadow_probability)))
        if maximum_error > 2e-6:
            raise AssertionError(f"native shadow parity failed for seed {seed}: {maximum_error}")
        parity.append({"seed": seed, "rows": len(parity_rows), "max_absolute_probability_error": maximum_error})
        for dataset_name, (rows, labels) in datasets.items():
            for ratio in MIXTURE_RATIOS:
                for variant in VARIANTS:
                    runs.append({"seed": seed, "dataset": dataset_name, "mixture_ratio": ratio,
                                 "variant": variant,
                                 **evaluate_variant(shadow, rows, labels, ratio, variant, calibration)})
    aggregate = {}
    for dataset_name in datasets:
        aggregate[dataset_name] = {}
        for ratio in MIXTURE_RATIOS:
            aggregate[dataset_name][str(ratio)] = {}
            for variant in VARIANTS:
                selected = [row for row in runs if row["dataset"] == dataset_name and
                            row["mixture_ratio"] == ratio and row["variant"] == variant]
                aggregate[dataset_name][str(ratio)][variant] = {
                    key: {"mean": float(np.mean([row[key] for row in selected])),
                          "values_by_seed": [row[key] for row in selected]}
                    for key in selected[0] if key not in ("seed", "dataset", "mixture_ratio", "variant")
                }
    report = {
        "status": "APL-inspired engineered inhibition comparison; not a biological reconstruction",
        "protocol": {
            "frozen_before_evaluation": True, "no_training": True,
            "seeds": list(SEEDS), "variants": list(VARIANTS), "mixture_ratios": list(MIXTURE_RATIOS),
            "calibration_boundary": "global/local feedback gains selected only from old_train hidden activity to approximate 2% mean active fraction",
            "evaluation": "old training descriptive calibration check, old validation/calibration split, and reused 32-case confirmation set",
            "local_groups": f"{LOCAL_GROUPS} groups by hidden index modulo {LOCAL_GROUPS}; arbitrary engineering partition, not anatomical",
            "global_proxy": "relu(hidden - gain*row_mean_hidden); gain frozen from training only; functional global-feedback proxy",
            "local_proxy": "same subtractive rule using arbitrary group means; functional local competition proxy",
            "native_readout_caveat": "all variants reuse the MBON weights/readout trained under native 2% top-k, so readout mismatch can disadvantage alternative hidden mechanisms",
            "pairing_scope": "pattern-separation and discriminability metrics use deterministic next-class pairs only; no nearest-neighbor selection or similar-versus-dissimilar interaction is measured",
        },
        "parity": parity, "calibration": calibrations,
        "parameter_scope": "same frozen checkpoint weights per seed; only hidden inhibition transform changes",
        "provenance_sha256": {**provenance,
                              **{str(checkpoint(seed).relative_to(ROOT)): sha256(checkpoint(seed)) for seed in SEEDS},
                              str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
        "runs": runs, "aggregate": aggregate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    matched_readout_control(report)
    report["posthoc_matched_readout"]["pre_append_report_sha256"] = None
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--append-matched-readout", action="store_true")
    args = parser.parse_args()
    report = append_matched_readout(args.output) if args.append_matched_readout else run(args.output)
    print(json.dumps({"parity": report["parity"],
                      "confirmation_clean": report["aggregate"]["reused_confirmation_32"]["0.0"]}, indent=2))


if __name__ == "__main__":
    main()
