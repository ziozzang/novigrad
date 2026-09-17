#!/usr/bin/env python3
"""Causal ablations of a frozen rate-model KC representation.

The decoder uses frozen embedding-derived inputs and supervised labels. These
interventions test this engineered rate model, not thought extraction from a
fly brain and not a biological claim about individual Kenyon cells.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from inhibition_mechanism import (
    ROOT, SEEDS, ShadowEngine, calibrate, checkpoint, fit_dual_ridge,
    l2_rows, load_inputs, sha256,
)
from novigrad import Engine


OUT = ROOT / "results/thought-bridge/interventions.json"
FRACTIONS = (0.05, 0.25, 0.50)
RANDOM_SEED_BASE = 927_401
L1_MATCHED_RANDOM_SEED_BASE = 8_321


def decoder_parameters(shadow, train_rows, train_labels, calibration):
    hidden = shadow.inhibit(shadow.raw_hidden(train_rows), "native_topk_2", calibration)
    basis, dual = fit_dual_ridge(hidden, train_labels, ridge=0.1, actions=4)
    saliency = basis.T @ dual
    return basis, dual, saliency


def decoder_scores(hidden, saliency):
    return l2_rows(hidden) @ saliency


def selected_units(hidden, saliency, decoded_class, strategy, fraction, random_seed):
    active = np.flatnonzero(hidden != 0)
    count = min(len(active), int(np.ceil(len(active) * fraction)))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    contribution = hidden[active] * saliency[active, decoded_class]
    if strategy == "top_contribution":
        order = np.argsort(-contribution, kind="stable")
        return active[order[:count]]
    if strategy == "bottom_contribution":
        order = np.argsort(contribution, kind="stable")
        return active[order[:count]]
    if strategy == "random_active":
        return np.sort(np.random.default_rng(random_seed).choice(active, count, replace=False))
    raise ValueError("unknown intervention strategy")


def intervention_specs():
    specs = [{"name": "none", "strategy": "none", "fraction": 0.0}]
    for fraction in FRACTIONS:
        for strategy in ("top_contribution", "random_active", "bottom_contribution"):
            specs.append({"name": f"{strategy}_{int(100*fraction)}pct", "strategy": strategy,
                          "fraction": fraction})
    specs.append({"name": "all_active", "strategy": "all", "fraction": 1.0})
    return specs


def intervene(hidden, saliency, decoded_class, spec, random_seed):
    output = hidden.copy()
    active = np.flatnonzero(hidden != 0)
    if spec["strategy"] == "none":
        removed = np.empty(0, dtype=np.int64)
    elif spec["strategy"] == "all":
        removed = active
    else:
        removed = selected_units(hidden, saliency, decoded_class, spec["strategy"],
                                 spec["fraction"], random_seed)
    original_l1 = float(np.abs(hidden).sum())
    removed_l1 = float(np.abs(hidden[removed]).sum())
    output[removed] = 0.0
    return output, removed, (0.0 if original_l1 == 0 else min(1.0, removed_l1 / original_l1))


def random_l1_matched_intervention(hidden, target_removed_l1_fraction, random_seed):
    """Random-order attenuation matching a requested removed L1 fraction."""
    if not 0.0 <= target_removed_l1_fraction <= 1.0:
        raise ValueError("target removed L1 fraction must be in [0, 1]")
    output = hidden.copy()
    active = np.flatnonzero(hidden != 0)
    order = np.random.default_rng(random_seed).permutation(active)
    total = float(np.abs(hidden).sum())
    target = target_removed_l1_fraction * total
    removed = 0.0
    fully_removed = []
    partial = []
    for index in order:
        magnitude = abs(float(hidden[index]))
        remaining = target - removed
        if remaining <= 0:
            break
        if magnitude <= remaining + 1e-15:
            output[index] = 0.0
            fully_removed.append(int(index))
            removed += magnitude
        else:
            fraction = remaining / magnitude
            output[index] = hidden[index] * (1.0 - fraction)
            partial.append(int(index))
            removed += remaining
            break
    achieved = 0.0 if total == 0 else removed / total
    changed = fully_removed + partial
    return output, np.asarray(changed, dtype=np.int64), np.asarray(fully_removed, dtype=np.int64), np.asarray(partial, dtype=np.int64), achieved


def save_decoder(path, basis, dual, saliency, labels):
    metadata = {
        "format": "novi.supervised-kc-decoder", "ridge_lambda": "0.1",
        "status": "exploratory", "score_semantics": "uncalibrated_coefficient_score",
    }
    save_file({"train_hidden_basis": basis, "dual_coefficients": dual,
               "kc_class_saliency": saliency, "train_labels": labels.astype(np.int64)},
              str(path), metadata=metadata)
    restored = load_file(str(path))
    reference = basis @ basis.T @ dual
    roundtrip = restored["train_hidden_basis"] @ restored["train_hidden_basis"].T @ restored["dual_coefficients"]
    error = float(np.max(np.abs(reference - roundtrip)))
    if error != 0.0:
        raise AssertionError(f"decoder artifact roundtrip failed: {error}")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "metadata": metadata,
            "roundtrip_max_absolute_training_score_error": error}


def run(output=OUT):
    datasets, input_provenance = load_inputs()
    train_rows, train_labels = datasets["old_train"]
    test_rows, test_labels = datasets["reused_confirmation_32"]
    reference_shadow = ShadowEngine(checkpoint(SEEDS[0]))
    calibration = calibrate(reference_shadow, train_rows)
    basis, dual, saliency = decoder_parameters(reference_shadow, train_rows, train_labels, calibration)
    output.parent.mkdir(parents=True, exist_ok=True)
    decoder_artifact = save_decoder(output.parent / "thought-decoder-ridge.safetensors",
                                    basis, dual, saliency, train_labels)
    reference_hidden = reference_shadow.inhibit(
        reference_shadow.raw_hidden(test_rows), "native_topk_2", calibration
    )
    baseline_decoder_scores = decoder_scores(reference_hidden, saliency)
    baseline_decoded = baseline_decoder_scores.argmax(axis=1)
    specs = intervention_specs()
    records = []
    parity = []
    initial_weights = {}

    for seed in SEEDS:
        shadow = ShadowEngine(checkpoint(seed))
        hidden = shadow.inhibit(shadow.raw_hidden(test_rows), "native_topk_2", calibration)
        if not np.array_equal(hidden, reference_hidden):
            raise AssertionError("PN-to-KC hidden representation differs across checkpoints")
        native = Engine.load(checkpoint(seed))
        initial_weights[str(seed)] = hashlib.sha256(np.asarray(native.weights, np.float32).tobytes()).hexdigest()
        native_probability = np.asarray(native.infer_batch(test_rows.tolist()))
        shadow_probability = shadow.probabilities(hidden)
        error = float(np.max(np.abs(native_probability - shadow_probability)))
        if error > 2e-6:
            raise AssertionError(f"native parity failed for seed {seed}: {error}")
        parity.append({"seed": seed, "max_absolute_probability_error": error})
        baseline_native = native_probability.argmax(axis=1)

        for case_index, (row_hidden, truth, decoded, native_pred) in enumerate(
                zip(hidden, test_labels, baseline_decoded, baseline_native)):
            for spec_index, spec in enumerate(specs):
                random_seed = RANDOM_SEED_BASE + case_index * 101 + spec_index
                changed, removed, removed_l1_fraction = intervene(
                    row_hidden, saliency, int(decoded), spec, random_seed
                )
                changed_decoder_scores = decoder_scores(changed[None, :], saliency)[0]
                changed_decoded = int(np.argmax(changed_decoder_scores))
                changed_native_probability = shadow.probabilities(changed[None, :])[0]
                changed_native = int(np.argmax(changed_native_probability))
                records.append({
                    "seed": seed, "case_index": case_index, "condition": spec["name"],
                    "strategy": spec["strategy"], "fraction": spec["fraction"],
                    "active_count": int(np.count_nonzero(row_hidden)),
                    "removed_count": int(len(removed)),
                    "removed_activity_l1_fraction": removed_l1_fraction,
                    "ground_truth_evaluator_only": int(truth),
                    "baseline_decoded_class": int(decoded), "intervened_decoded_class": changed_decoded,
                    "decoder_flip": bool(changed_decoded != decoded),
                    "baseline_native_class": int(native_pred), "intervened_native_class": changed_native,
                    "native_policy_flip": bool(changed_native != native_pred),
                    "baseline_decoder_native_agreement": bool(int(decoded) == int(native_pred)),
                    "intervened_decoder_native_agreement": bool(changed_decoded == changed_native),
                    "decoder_correct": bool(changed_decoded == truth),
                    "native_correct": bool(changed_native == truth),
                    "decoded_target_score_change": float(changed_decoder_scores[decoded] - baseline_decoder_scores[case_index, decoded]),
                    "decoded_target_native_probability_change": float(changed_native_probability[decoded] - native_probability[case_index, decoded]),
                    "native_max_probability_change": float(np.max(np.abs(changed_native_probability - native_probability[case_index]))),
                })
        if initial_weights[str(seed)] != hashlib.sha256(np.asarray(native.weights, np.float32).tobytes()).hexdigest():
            raise AssertionError("native weights changed during frozen evaluation")

    aggregate = {}
    for spec in specs:
        selected = [row for row in records if row["condition"] == spec["name"]]
        aggregate[spec["name"]] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "removed_count", "removed_activity_l1_fraction", "decoder_flip",
                "native_policy_flip", "baseline_decoder_native_agreement",
                "intervened_decoder_native_agreement", "decoder_correct", "native_correct",
                "decoded_target_score_change", "decoded_target_native_probability_change",
                "native_max_probability_change",
            )
        }
        aggregate[spec["name"]]["case_count"] = len(selected)
        aggregate[spec["name"]]["unique_input_case_count"] = len(test_rows)

    report = {
        "status": "causal intervention in a frozen engineered rate model; not fly-brain thought extraction",
        "design": {
            "decoder": "independent supervised ridge class readout on L2-normalized native 2% KC hidden vectors; lambda 0.1 fixed without tuning",
            "decoder_boundary": "old 32 training labels only; native MBON policy is not used to select intervention targets",
            "target_selection": "baseline decoded class from the independent probe; ground truth is evaluator-only",
            "intervention": "after native top-k and max normalization; native MBON evaluation receives silenced activity without renormalization",
            "fractions": list(FRACTIONS), "random_seed_base": RANDOM_SEED_BASE,
            "matched_count": "top, random, and bottom conditions remove exactly ceil(fraction*active_count) active KC units",
            "magnitude_caveat": "the original top/random/bottom grid is count-matched but differs in removed L1; the later post-hoc random_l1matched25 control matches top-25% removed L1 exactly while changing the number of affected units",
            "replication_caveat": "three checkpoints change learned MBON weights only; PN-to-KC hidden features and decoder are identical, and repeated case rows are not independent encoder replicates",
            "agreement_proxy": "per-case decoded/native class agreement is a descriptive correlation proxy, not a statistical correlation coefficient",
            "saliency": "KC coefficient contribution is hidden_activity * (train_hidden_basis.T @ dual_coefficients) for the independently decoded class",
            "data": "reused 32-case confirmation set; exploratory, not a fresh holdout",
            "all_weights_frozen": True,
        },
        "decoder_artifact": decoder_artifact,
        "parity": parity, "initial_weight_hashes": initial_weights,
        "provenance_sha256": {**input_provenance,
                              **{str(checkpoint(seed).relative_to(ROOT)): sha256(checkpoint(seed)) for seed in SEEDS},
                              str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
        "aggregate": aggregate, "records": records,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def append_l1_matched_control(output=OUT):
    """Append the post-hoc random L1-matched 25% control only."""
    report = json.loads(output.read_text())
    if "posthoc_random_l1matched25" in report:
        raise ValueError("L1-matched control already exists")
    original_records = list(report["records"])
    datasets, _ = load_inputs()
    train_rows, train_labels = datasets["old_train"]
    test_rows, test_labels = datasets["reused_confirmation_32"]
    reference = ShadowEngine(checkpoint(SEEDS[0]))
    calibration = calibrate(reference, train_rows)
    _, _, saliency = decoder_parameters(reference, train_rows, train_labels, calibration)
    hidden = reference.inhibit(reference.raw_hidden(test_rows), "native_topk_2", calibration)
    baseline_scores = decoder_scores(hidden, saliency)
    decoded_classes = baseline_scores.argmax(axis=1)
    new_records = []
    for seed in SEEDS:
        shadow = ShadowEngine(checkpoint(seed))
        native_probability = shadow.probabilities(hidden)
        native_classes = native_probability.argmax(axis=1)
        for case_index, (row_hidden, truth, decoded, native_pred) in enumerate(
                zip(hidden, test_labels, decoded_classes, native_classes)):
            top_spec = {"strategy": "top_contribution", "fraction": 0.25}
            _, _, target_fraction = intervene(row_hidden, saliency, int(decoded), top_spec, 0)
            changed, changed_indices, fully_removed, partial, achieved = random_l1_matched_intervention(
                row_hidden, target_fraction, L1_MATCHED_RANDOM_SEED_BASE + case_index
            )
            if abs(achieved - target_fraction) > 1e-12:
                raise AssertionError("failed to match top-25% removed L1 target")
            changed_scores = decoder_scores(changed[None, :], saliency)[0]
            changed_decoded = int(np.argmax(changed_scores))
            changed_native_probability = shadow.probabilities(changed[None, :])[0]
            changed_native = int(np.argmax(changed_native_probability))
            new_records.append({
                "seed": seed, "case_index": case_index, "condition": "random_l1matched25",
                "strategy": "random_order_l1_matched", "fraction": 0.25,
                "active_count": int(np.count_nonzero(row_hidden)),
                "removed_count": int(len(fully_removed)), "changed_count": int(len(changed_indices)),
                "partially_attenuated_count": int(len(partial)),
                "removed_activity_l1_fraction": achieved,
                "top25_target_removed_activity_l1_fraction": target_fraction,
                "ground_truth_evaluator_only": int(truth),
                "baseline_decoded_class": int(decoded), "intervened_decoded_class": changed_decoded,
                "decoder_flip": bool(changed_decoded != decoded),
                "baseline_native_class": int(native_pred), "intervened_native_class": changed_native,
                "native_policy_flip": bool(changed_native != native_pred),
                "baseline_decoder_native_agreement": bool(int(decoded) == int(native_pred)),
                "intervened_decoder_native_agreement": bool(changed_decoded == changed_native),
                "decoder_correct": bool(changed_decoded == truth),
                "native_correct": bool(changed_native == truth),
                "decoded_target_score_change": float(changed_scores[decoded] - baseline_scores[case_index, decoded]),
                "decoded_target_native_probability_change": float(changed_native_probability[decoded] - native_probability[case_index, decoded]),
                "native_max_probability_change": float(np.max(np.abs(changed_native_probability - native_probability[case_index]))),
            })
    report["records"].extend(new_records)
    selected = new_records
    report["aggregate"]["random_l1matched25"] = {
        key: float(np.mean([row[key] for row in selected]))
        for key in (
            "removed_count", "changed_count", "partially_attenuated_count",
            "removed_activity_l1_fraction", "top25_target_removed_activity_l1_fraction",
            "decoder_flip", "native_policy_flip", "baseline_decoder_native_agreement",
            "intervened_decoder_native_agreement", "decoder_correct", "native_correct",
            "decoded_target_score_change", "decoded_target_native_probability_change",
            "native_max_probability_change",
        )
    }
    report["aggregate"]["random_l1matched25"].update({"case_count": len(selected),
                                                         "unique_input_case_count": len(test_rows)})
    report["posthoc_random_l1matched25"] = {
        "status": "added after observing the count-matched intervention results; exploratory and not preregistered",
        "random_seed_base": L1_MATCHED_RANDOM_SEED_BASE,
        "method": "random active-unit order; fully silence until the per-row top-25% L1 target, then partially attenuate one unit to match exactly",
        "comparison": "existing random_active_25pct is equal-count; random_l1matched25 is equal-removed-L1 but changes a variable number of units",
        "target_boundary": "uses the independent decoder prediction and top-25% contribution magnitude target, never the evaluator label",
    }
    report["design"]["magnitude_caveat"] = (
        "the original top/random/bottom grid is count-matched but differs in removed L1; "
        "the later post-hoc random_l1matched25 control matches top-25% removed L1 exactly "
        "while changing the number of affected units"
    )
    if report["records"][:len(original_records)] != original_records:
        raise AssertionError("original intervention records changed")
    report["provenance_sha256"][str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--append-l1-control", action="store_true")
    args = parser.parse_args()
    report = append_l1_matched_control(args.output) if args.append_l1_control else run(args.output)
    print(json.dumps({"parity": report["parity"], "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
