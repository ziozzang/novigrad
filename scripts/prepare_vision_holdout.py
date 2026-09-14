#!/usr/bin/env python3
"""Prepare sealed final holdouts; never use these artifacts for model selection."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_vision import BASE, FILES, OUT, ROOT, digest, glyph, read_idx

SEED = 20260916
MANIFEST = ROOT / "results" / "improvement" / "holdout-manifest.json"


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fashion():
    source_names = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    sources = {name: OUT / "source" / name for name in source_names}
    for name, path in sources.items():
        require(digest(path, "md5") == FILES[name], f"Source checksum mismatch: {name}")
    images = read_idx(sources[source_names[0]], "images")
    labels = read_idx(sources[source_names[1]], "labels")
    require(len(images) == len(labels) == 10000, "Expected official 10,000-image test set")
    with np.load(OUT / "fashion_mnist.npz", allow_pickle=False) as original:
        excluded = original["test_source_indices"]
        require(excluded.dtype.kind in "iu", "Test indices must be integers")
        require(len(excluded) == len(np.unique(excluded)) == 2000, "Expected 2,000 unique original test indices")
        require(np.all((excluded >= 0) & (excluded < 10000)), "Test indices out of bounds")
        require(np.array_equal(original["test_images"], images[excluded]), "Original test images differ from source")
        require(np.array_equal(original["test_labels"], labels[excluded]), "Original test labels differ from source")
    selected = np.setdiff1d(np.arange(10000, dtype=np.int32), excluded)
    require(len(selected) == 8000 and not np.intersect1d(selected, excluded).size, "Holdout overlap")
    return images[selected], labels[selected], {
        "source": "Official Fashion-MNIST test set only",
        "source_files": {name: {"url": f"{BASE}/{name}", "md5": FILES[name], "sha256": digest(path)} for name, path in sources.items()},
        "selection": "Ascending official test indices excluding original test_source_indices",
        "source_indices_sha256": array_digest(selected.astype("<i4")),
        "excluded_test_indices_sha256": array_digest(excluded.astype("<i4")),
        "excluded_count": int(len(excluded)),
        "exclusion_assertions": {"original_test_indices_disjoint": True, "official_training_source_unused": True,
                                 "original_test_images_and_labels_match_source": True},
    }


def captcha():
    provenance = json.loads((OUT / "manifest.json").read_text())["captcha"]
    font_paths = [Path(path) for path in provenance["font_files"]]
    for path in font_paths:
        require(digest(path) == provenance["font_files"][str(path)], f"Original CAPTCHA font changed: {path}")
    fonts = [ImageFont.truetype(str(path), 42) for path in font_paths]
    excluded_by_split = {}
    with np.load(OUT / "captcha.npz", allow_pickle=False) as original:
        for split, expected in (("train", 3000), ("validation", 500), ("test", 500)):
            labels = original[f"{split}_labels"]
            require(labels.shape == (expected, 4) and labels.dtype.kind in "iu", "Unexpected original CAPTCHA labels")
            require(np.all(labels < 10), "CAPTCHA labels must be decimal digits")
            excluded_by_split[split] = labels.astype(np.int32) @ np.array([1000, 100, 10, 1], dtype=np.int32)
    excluded = np.concatenate(list(excluded_by_split.values()))
    require(len(np.unique(excluded)) == 4000, "Original CAPTCHA codes must be unique")
    rng = np.random.default_rng(SEED)
    codes = rng.permutation(np.setdiff1d(np.arange(10000), excluded))[:1000]
    require(len(np.unique(codes)) == 1000 and not np.intersect1d(codes, excluded).size, "CAPTCHA code overlap")
    labels = np.array([[int(ch) for ch in f"{code:04d}"] for code in codes], dtype=np.uint8)
    images = np.empty((1000, 28, 112), dtype=np.uint8)
    for i, row in enumerate(labels):
        cells = np.concatenate([glyph(str(d), rng, fonts) for d in row], axis=1).astype(np.int16)
        # Same nuisance strokes and noise sequence as prepare_vision.captcha.
        overlay = Image.new("L", (112, 28), 0)
        draw = ImageDraw.Draw(overlay)
        for _ in range(int(rng.integers(0, 3))):
            draw.line((int(rng.integers(112)), int(rng.integers(28)),
                       int(rng.integers(112)), int(rng.integers(28))),
                      fill=int(rng.integers(15, 55)), width=1)
        noise = rng.normal(0, 8, size=(28, 112))
        images[i] = np.clip(cells + np.asarray(overlay) + noise, 0, 255).astype(np.uint8)
    return images, labels, {
        "source": "Fresh synthetic four-digit CAPTCHA using prepare_vision.glyph and original noise process",
        "seed": SEED, "font_files": provenance["font_files"],
        "selection": "Random permutation of all four-digit codes absent from every original split; first 1000",
        "codes_sha256": array_digest(codes.astype("<i4")),
        "excluded_count": int(len(excluded)),
        "excluded_codes_sha256": array_digest(np.sort(excluded).astype("<i4")),
        "exclusion_assertions": {f"original_{split}_codes_disjoint": not bool(np.intersect1d(codes, values).size)
                                 for split, values in excluded_by_split.items()},
        "unique_holdout_codes": True,
        "limitations": "Synthetic fixed-cell CAPTCHA; not an arbitrary web CAPTCHA benchmark",
    }


def main():
    targets = {name: OUT / f"holdout_{name}.npz" for name in ("fashion", "captcha")}
    present = [path.exists() for path in targets.values()]
    require(not any(present) or all(present), "Partial holdout artifacts found; refusing to overwrite or fill gaps")
    existing = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    require(not any(present) or existing is not None, "Holdout data exist without provenance manifest")
    originals = {name: OUT / name for name in ("fashion_mnist.npz", "captcha.npz", "manifest.json")}
    before = {name: digest(path) for name, path in originals.items()}
    if existing:
        require(existing["original_artifact_sha256"] == before, "Original artifacts do not match sealed provenance")
    if all(present):
        for name, path in targets.items():
            require(digest(path) == existing["datasets"][name]["artifact_sha256"], f"Existing sealed artifact hash mismatch: {name}")
        print(json.dumps({"verified_existing": {name: entry["count"] for name, entry in existing["datasets"].items()}}))
        return
    prepared = {"fashion": fashion(), "captcha": captcha()}
    manifest = {
        "purpose": "Sealed final evaluation only; do not inspect predictions or accuracy until configuration selection is complete",
        "schema_version": 1,
        "original_artifact_sha256": before,
        "preparation_script_sha256": digest(Path(__file__).resolve()),
        "upstream_generator_sha256": digest(ROOT / "scripts" / "prepare_vision.py"),
        "datasets": {},
    }
    encoded = {}
    for name, (images, labels, provenance) in prepared.items():
        path = targets[name]
        output = io.BytesIO()
        np.savez_compressed(output, images=images, labels=labels)
        encoded[name] = output.getvalue()
        with np.load(io.BytesIO(encoded[name]), allow_pickle=False) as saved:
            require(set(saved.files) == {"images", "labels"}, "Unexpected artifact fields")
            require(np.array_equal(saved["images"], images) and np.array_equal(saved["labels"], labels), "Artifact round-trip mismatch")
        manifest["datasets"][name] = {**provenance, "artifact": str(path.relative_to(ROOT)),
            "artifact_sha256": hashlib.sha256(encoded[name]).hexdigest(), "count": int(len(labels)),
            "images_shape": list(images.shape), "labels_shape": list(labels.shape),
            "dtype": str(images.dtype), "images_sha256": array_digest(images), "labels_sha256": array_digest(labels)}
    require(before == {name: digest(path) for name, path in originals.items()}, "Original data changed")
    manifest["original_artifacts_unchanged"] = True
    if existing:
        # A clean clone contains the tracked manifest but no ignored data. Rebuild
        # only if every generated hash, exclusion assertion and count matches it.
        require(existing["datasets"] == manifest["datasets"], "Regenerated holdouts differ from sealed provenance")
        require(existing["upstream_generator_sha256"] == manifest["upstream_generator_sha256"], "Upstream generator differs from sealed provenance")
    for name, path in targets.items():
        with path.open("xb") as output:
            output.write(encoded[name])
    if not existing:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("x") as output:
            output.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: item["count"] for name, item in manifest["datasets"].items()}))


if __name__ == "__main__":
    main()
