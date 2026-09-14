#!/usr/bin/env python3
"""Verify actual Safetensors portability with an independent NumPy forward pass."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors import safe_open


def verify(checkpoint, binary):
    with safe_open(str(checkpoint), framework="numpy") as model:
        metadata = model.metadata()
        tensors = {key: model.get_tensor(key) for key in model.keys()}
    assert metadata["format"] == "nobi.plastic"
    assert tensors["plastic_weight"].dtype == np.float32
    assert tensors["input_ids"].dtype == np.uint64
    assert np.isfinite(tensors["plastic_weight"]).all()
    n_hidden = len(tensors["hidden_ids"])
    n_output = len(tensors["output_ids"])
    actions = int(metadata["actions"])
    # Deterministic external input: independent of training driver and target labels.
    inputs = np.array([0.15 + ((int(root) * 17) % 101) / 100 for root in tensors["input_ids"]], dtype=np.float32)
    pre = tensors["input_pre"].astype(np.int64)
    post = tensors["input_post"].astype(np.int64)
    counts = tensors["input_count"].astype(np.float64)
    totals = np.bincount(post, weights=counts, minlength=n_hidden)
    fixed = (counts / totals[post]).astype(np.float32) * tensors["input_sign"]
    hidden = np.zeros(n_hidden, dtype=np.float32)
    np.add.at(hidden, post, fixed * inputs[pre])
    np.maximum(hidden, 0, out=hidden)
    keep = max(1, min(n_hidden, int(np.ceil(np.float32(n_hidden) * np.float32(metadata["active_fraction"])))))
    ranking = np.argsort(-hidden, kind="stable")
    hidden[ranking[keep:]] = 0
    if hidden.max() > 0:
        hidden /= hidden.max()
    output = np.zeros(n_output, dtype=np.float32)
    np.add.at(output, tensors["plastic_post"].astype(np.int64), tensors["plastic_weight"] * hidden[tensors["plastic_pre"].astype(np.int64)])
    output *= tensors.get("output_gains", np.ones(n_output, dtype=np.float32))
    groups = tensors["output_actions"].astype(np.int64)
    logits = np.zeros(actions, dtype=np.float32)
    np.add.at(logits, groups, output)
    divisor = np.bincount(groups, minlength=actions).astype(np.float32)
    if metadata["version"] == "2":
        divisor = np.sqrt(divisor)
    logits *= np.float32(metadata["logit_gain"]) / divisor
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    input_path = checkpoint.with_suffix(".verification-input.tsv")
    input_path.write_text("\t".join(f"{int(root)}:{float(x):.9g}" for root, x in zip(tensors["input_ids"], inputs)) + "\n")
    result = subprocess.run([str(binary.resolve()), "infer", str(checkpoint), str(input_path)], capture_output=True, text=True, check=True)
    rust = np.array([float(line.split("\t")[1]) for line in result.stdout.splitlines()])
    np.testing.assert_allclose(rust, probability, atol=2e-6, rtol=2e-6)
    record = {"checkpoint": str(checkpoint), "tensor_count": len(tensors), "input_neurons": len(inputs), "hidden_neurons": n_hidden, "output_neurons": n_output, "plastic_edges": len(tensors["plastic_weight"]), "numpy_probabilities": probability.tolist(), "rust_probabilities": rust.tolist(), "max_absolute_error": float(np.abs(rust-probability).max()), "passed": True}
    print(json.dumps(record))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("target/release/novi_engine"))
    parser.add_argument("--report", type=Path, default=Path("results/safetensors-verification.json"))
    args = parser.parse_args()
    records = [verify(path, args.binary) for path in args.checkpoints]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
