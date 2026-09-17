"""Encode fixed semantic candidates once; no training, tuning, or evaluation selection."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/Users/a405394/models/google_embeddinggemma-300m")
DEFAULT_CASES = ROOT / "examples/bio_bridge/thought_descriptions.json"
DEFAULT_OUT = ROOT / "results/thought-bridge"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text())
    rows = payload["descriptions"]
    texts = [row["text"] for row in rows]
    if len(rows) != 40 or len(set(texts)) != 40:
        raise ValueError("expected exactly 40 unique, finalized candidate descriptions")
    if sum(bool(row["supported"]) for row in rows) != 32:
        raise ValueError("expected 32 supported and 8 unsupported candidates")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    load_start = time.perf_counter()
    model = SentenceTransformer(str(args.model), device=device, local_files_only=True)
    load_seconds = time.perf_counter() - load_start
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.float32"]:
        raise ValueError(f"expected original FP32 parameters, got {parameter_dtypes}")

    # Warm-up is excluded from measured encoding, matching the original one-time run.
    model.encode(texts[:2], batch_size=2, prompt_name="Classification",
                 normalize_embeddings=True)
    if device == "mps":
        torch.mps.synchronize()
    encode_start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=16, prompt_name="Classification",
                              normalize_embeddings=True).astype(np.float32, copy=False)
    if device == "mps":
        torch.mps.synchronize()
    encode_seconds = time.perf_counter() - encode_start
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.shape != (40, 768) or not np.isfinite(embeddings).all():
        raise ValueError(f"unexpected embedding tensor: {embeddings.shape} {embeddings.dtype}")
    if not np.allclose(norms, 1.0, atol=2e-5):
        raise ValueError("embeddings are not normalized")

    args.out.mkdir(parents=True, exist_ok=True)
    embedding_path = args.out / "description-embeddings.safetensors"
    save_file({"embeddings": embeddings}, str(embedding_path), metadata={
        "format": "novigrad.semantic_candidate_embeddings", "version": "1",
        "prompt_name": "Classification", "normalized": "true", "dtype": "float32"})
    model_hashes = {str(path.relative_to(args.model)): sha256(path)
                    for path in sorted(args.model.rglob("*.safetensors"))}
    script_path = Path(__file__).resolve()
    report = {
        "purpose": "Frozen embeddings of authored semantic candidate descriptions; not decoded or observed biological thoughts.",
        "external_validity": "None established. Candidates were authored independently and were not trained, tuned, or selected using evaluation outputs.",
        "generator": str(script_path.relative_to(ROOT)), "generator_sha256": sha256(script_path),
        "model": str(args.model), "weight_sha256": model_hashes,
        "cases_path": str(args.cases.resolve().relative_to(ROOT)), "cases_sha256": sha256(args.cases),
        "embedding_path": str(embedding_path.resolve().relative_to(ROOT)),
        "embedding_sha256": sha256(embedding_path),
        "rows": 40, "supported_rows": 32, "unsupported_rows": 8,
        "row_order": "embedding row i corresponds to descriptions[i] in the cases JSON",
        "shape": list(embeddings.shape), "dtype": str(embeddings.dtype), "normalized": True,
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "prompt_name": "Classification", "device": device,
        "parameter_dtypes": parameter_dtypes, "model_load_seconds": load_seconds,
        "encode_seconds": encode_seconds, "batch_size": 16, "encoder_frozen": True,
        "versions": {name: importlib.metadata.version(name)
                     for name in ["torch", "transformers", "sentence-transformers", "safetensors"]},
    }
    (args.out / "description-provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
