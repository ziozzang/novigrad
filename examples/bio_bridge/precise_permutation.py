"""Port-placement sensitivity controls for the precise embedding bridge.

The first 256 signed feature ports are permuted while the 63 padding ports and
the fly-derived PN-to-KC topology remain fixed.  This is an engineered mapping
control, not a biological sensory alignment.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from delayed_credit import ROOT
from inhibition_mechanism import ShadowEngine, checkpoint, sha256
from precise_bridge import OUT, MODES, RIDGES, PortBridge, dataset, hidden, score
from thought_embedding import GOALS, decode, fit_alignment


SEEDS = (811, 812, 813, 814, 815)
SIGNED_PORTS = 256
TOTAL_PORTS = 319


def port_permutation(seed):
    """Return output-to-source indices for the 256 signed feature ports."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    return np.random.default_rng(seed).permutation(SIGNED_PORTS).astype(np.int64)


def permute_ports(ports, permutation):
    """Set output feature port j from source feature port permutation[j]."""
    rows = np.asarray(ports, dtype=np.float32)
    order = np.asarray(permutation, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != TOTAL_PORTS or not np.isfinite(rows).all():
        raise ValueError("finite [rows,319] ports required")
    if order.shape != (SIGNED_PORTS,) or not np.array_equal(np.sort(order), np.arange(SIGNED_PORTS)):
        raise ValueError("permutation must contain each signed port exactly once")
    result = rows.copy()
    result[:, :SIGNED_PORTS] = rows[:, order]
    return result


def _artifact_path(out_dir, mode, seed):
    return Path(out_dir) / f"perm-{mode}-{seed}.safetensors"


def _save_decoder(path, basis, coefficients, permutation, mode, seed, ridge):
    save_file(
        {
            "basis": np.ascontiguousarray(basis),
            "coefficients": np.ascontiguousarray(coefficients),
            "permutation": np.ascontiguousarray(permutation, dtype=np.int64),
        },
        str(path),
        metadata={
            "format": "novigrad.permuted_port_semantic_alignment",
            "version": "1",
            "mode": mode,
            "seed": str(seed),
            "ridge": str(ridge),
            "permutation_convention": "output signed port j receives source signed port permutation[j]",
            "boundary": "software port-placement sensitivity control; not biological sensory alignment",
        },
    )


def _selected_trial(trials):
    return max(trials, key=lambda row: (row["accuracy"], row["cosine"], -row["ridge"]))


def run_development(out_dir=OUT):
    """Fit permutation-specific decoders using train32 and validation12 only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dataset()
    train_x, train_y, train_text = data["train"]
    val_x, val_y, val_text = data["validation"]
    shadow = ShadowEngine(checkpoint(601))
    results = {}

    for mode in MODES:
        bridge_path = out_dir / f"ports-{mode}.safetensors"
        if not bridge_path.exists():
            raise FileNotFoundError(f"run precise_bridge development first: {bridge_path}")
        bridge = PortBridge.load(bridge_path, mode)
        base_train = bridge.encode(train_x)
        base_val = bridge.encode(val_x)
        mode_rows = []
        for seed in SEEDS:
            permutation = port_permutation(seed)
            train_hidden = hidden(shadow, permute_ports(base_train, permutation))
            val_hidden = hidden(shadow, permute_ports(base_val, permutation))
            trials = []
            for ridge in RIDGES:
                basis, coefficients = fit_alignment(train_hidden, train_x[:, :128], ridge)
                measured = score(decode(basis, coefficients, val_hidden), val_x, val_y, val_text)
                trials.append({
                    "ridge": ridge,
                    "accuracy": measured["semantic_category_accuracy"],
                    "cosine": measured["mean_cosine_to_input_embedding"],
                })
            selected = _selected_trial(trials)
            basis, coefficients = fit_alignment(train_hidden, train_x[:, :128], selected["ridge"])
            path = _artifact_path(out_dir, mode, seed)
            _save_decoder(path, basis, coefficients, permutation, mode, seed, selected["ridge"])
            restored = load_file(str(path))
            np.testing.assert_array_equal(restored["permutation"], permutation)
            np.testing.assert_array_equal(
                decode(basis, coefficients, val_hidden),
                decode(restored["basis"], restored["coefficients"], val_hidden),
            )
            mode_rows.append({
                "seed": seed,
                "selected": selected,
                "trials": trials,
                "train": score(decode(basis, coefficients, train_hidden), train_x, train_y, train_text),
                "validation": score(decode(basis, coefficients, val_hidden), val_x, val_y, val_text),
                "artifact": str(path.relative_to(ROOT)),
                "artifact_sha256": sha256(path),
                "mean_active_kcs": float(np.count_nonzero(val_hidden, axis=1).mean()),
                "roundtrip_exact": True,
            })
        results[mode] = mode_rows

    report = {
        "status": "development only; final authored holdout not read",
        "design": {
            "modes": list(MODES),
            "seeds": list(SEEDS),
            "permutation": "first 256 signed ports only; output j receives source permutation[j]",
            "padding": "ports 256:319 unchanged",
            "topology": "same frozen seed601 PN-to-KC topology for every condition",
            "fit": "existing train-only PortBridge artifacts; decoder fit train32",
            "selection": "ridge among 0.01,0.1,1 by validation12 accuracy, cosine, then smaller ridge",
            "final_boundary": "no permutation or ridge may be selected using final results",
            "interpretation": "sensitivity to engineered coordinate placement, not biological PN semantics",
        },
        "results": results,
        "provenance_sha256": {
            "source": sha256(Path(__file__)),
            "dataset": sha256(ROOT / "results/gemma-bridge/dataset.json"),
            "embeddings": sha256(ROOT / "results/gemma-bridge/embeddings.safetensors"),
            "checkpoint": sha256(checkpoint(601)),
            **{f"ports_{mode}": sha256(out_dir / f"ports-{mode}.safetensors") for mode in MODES},
        },
    }
    destination = out_dir / "permutation-development.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _aggregate(rows):
    accuracy = np.asarray([row["all"]["semantic_category_accuracy"] for row in rows])
    cosine = np.asarray([row["all"]["mean_cosine_to_input_embedding"] for row in rows])
    return {
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_min": float(accuracy.min()),
        "accuracy_max": float(accuracy.max()),
        "cosine_mean": float(cosine.mean()),
        "values_are_five_fixed_permutations_not_independent_datasets": True,
    }


def run_final(out_dir=OUT):
    """Evaluate every frozen permutation on the locked authored holdout once."""
    out_dir = Path(out_dir)
    development_path = out_dir / "permutation-development.json"
    development = json.loads(development_path.read_text())
    cases = json.loads((ROOT / "examples/bio_bridge/precise_holdout.json").read_text())
    if isinstance(cases, dict):
        cases = cases["cases"]
    embeddings = load_file(str(out_dir / "holdout-embeddings.safetensors"))["embeddings"]
    labels = np.asarray([GOALS.index(row["class"]) for row in cases])
    texts = [row["text"] for row in cases]
    shadow = ShadowEngine(checkpoint(601))
    results = {}

    for mode in MODES:
        bridge = PortBridge.load(out_dir / f"ports-{mode}.safetensors", mode)
        base_ports = bridge.encode(embeddings)
        rows = []
        expected = {row["seed"]: row for row in development["results"][mode]}
        for seed in SEEDS:
            path = _artifact_path(out_dir, mode, seed)
            if sha256(path) != expected[seed]["artifact_sha256"]:
                raise ValueError(f"development artifact hash changed: {path}")
            saved = load_file(str(path))
            start = time.perf_counter()
            activity = hidden(shadow, permute_ports(base_ports, saved["permutation"]))
            predicted = decode(saved["basis"], saved["coefficients"], activity)
            elapsed = time.perf_counter() - start
            all_metrics = score(predicted, embeddings, labels, texts)
            languages = {}
            for language in ("en", "ko"):
                mask = np.asarray([row["language"] == language for row in cases])
                languages[language] = score(
                    predicted[mask], embeddings[mask], labels[mask],
                    [text for text, keep in zip(texts, mask) if keep],
                )
            rows.append({"seed": seed, "all": all_metrics, "languages": languages,
                         "batch_seconds": elapsed, "selected_development": expected[seed]["selected"]})
        results[mode] = {"permutations": rows, "aggregate": _aggregate(rows)}

    report = {
        "boundary": "All 15 development-frozen mode/permutation artifacts evaluated; no final selection or retuning. Authored bilingual families are dependent and are not biological recordings.",
        "development_sha256": sha256(development_path),
        "holdout_sha256": {
            "cases": sha256(ROOT / "examples/bio_bridge/precise_holdout.json"),
            "embeddings": sha256(out_dir / "holdout-embeddings.safetensors"),
        },
        "source_sha256": sha256(Path(__file__)),
        "results": results,
    }
    (out_dir / "permutation-final.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--development", action="store_true")
    phase.add_argument("--final", action="store_true")
    args = parser.parse_args()
    report = run_final() if args.final else run_development()
    print(json.dumps({"phase": "final" if args.final else "development", "modes": list(report["results"]) }))


if __name__ == "__main__":
    main()
