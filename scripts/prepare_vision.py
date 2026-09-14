#!/usr/bin/env python3
"""Prepare bounded, image-based Fashion-MNIST and fixed-cell CAPTCHA splits."""

import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vision"
BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FONTS = [Path("/System/Library/Fonts/SFNSMono.ttf"), Path("/System/Library/Fonts/Geneva.ttf")]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(name, expected):
    cache = OUT / "source"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if not target.exists() or digest(target, "md5") != expected:
        temp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(f"{BASE}/{name}", temp)
        if digest(temp, "md5") != expected:
            temp.unlink()
            raise ValueError(f"source checksum mismatch: {name}")
        temp.replace(target)
    return target


def read_idx(path, kind):
    with gzip.open(path, "rb") as f:
        blob = f.read()
    if kind == "images":
        magic, count, rows, cols = struct.unpack(">IIII", blob[:16])
        assert (magic, rows, cols) == (2051, 28, 28)
        result = np.frombuffer(blob, dtype=np.uint8, offset=16).reshape(count, 28, 28)
    else:
        magic, count = struct.unpack(">II", blob[:8])
        assert magic == 2049
        result = np.frombuffer(blob, dtype=np.uint8, offset=8)
    assert len(result) == count
    return result


def sheet(images, path, columns=16, limit=64):
    chosen = images[:limit]
    h, w = chosen.shape[-2:]
    canvas = Image.new("L", (columns * w, ((len(chosen) + columns - 1) // columns) * h))
    for i, a in enumerate(chosen):
        canvas.paste(Image.fromarray(a), ((i % columns) * w, (i // columns) * h))
    canvas.resize((canvas.width * 3, canvas.height * 3), Image.Resampling.NEAREST).save(path)


def fashion():
    sources = {n: download(n, m) for n, m in FILES.items()}
    images = read_idx(sources["train-images-idx3-ubyte.gz"], "images")
    labels = read_idx(sources["train-labels-idx1-ubyte.gz"], "labels")
    test_images = read_idx(sources["t10k-images-idx3-ubyte.gz"], "images")
    test_labels = read_idx(sources["t10k-labels-idx1-ubyte.gz"], "labels")
    rng = np.random.default_rng(20260914)

    def choose(y, n):
        ids = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:n] for c in range(10)])
        rng.shuffle(ids)
        return ids.astype(np.int32)

    all_train = np.concatenate([rng.permutation(np.flatnonzero(labels == c)) for c in range(10)])
    train_ids = np.concatenate([all_train[c * 6000:c * 6000 + 1000] for c in range(10)])
    val_ids = np.concatenate([all_train[c * 6000 + 1000:c * 6000 + 1200] for c in range(10)])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    test_ids = choose(test_labels, 200)
    assert not set(train_ids).intersection(val_ids)
    np.savez_compressed(OUT / "fashion_mnist.npz",
        train_images=images[train_ids], train_labels=labels[train_ids],
        validation_images=images[val_ids], validation_labels=labels[val_ids],
        test_images=test_images[test_ids], test_labels=test_labels[test_ids],
        train_source_indices=train_ids.astype(np.int32), validation_source_indices=val_ids.astype(np.int32),
        test_source_indices=test_ids)
    sheet(images[train_ids], OUT / "fashion_mnist_samples.png")
    return {"source": "Zalando Research Fashion-MNIST", "source_url": "https://github.com/zalandoresearch/fashion-mnist",
            "source_files": {n: {"url": f"{BASE}/{n}", "md5": FILES[n], "sha256": digest(p)} for n, p in sources.items()},
            "classes": CLASSES, "seed": 20260914,
            "splits": {"train": 10000, "validation": 2000, "test": 2000},
            "split_index_sha256": {"train": hashlib.sha256(train_ids.astype('<i4').tobytes()).hexdigest(),
                                   "validation": hashlib.sha256(val_ids.astype('<i4').tobytes()).hexdigest(),
                                   "test": hashlib.sha256(test_ids.astype('<i4').tobytes()).hexdigest()},
            "split_provenance": "train and val disjoint within official train; test from official test; each class has 1000/200/200 images"}


def glyph(digit, rng, fonts):
    canvas = Image.new("L", (56, 56), 0)
    draw = ImageDraw.Draw(canvas)
    font = fonts[int(rng.integers(len(fonts)))]
    bbox = draw.textbbox((0, 0), digit, font=font)
    x = (56 - (bbox[2] - bbox[0])) // 2 - bbox[0] + int(rng.integers(-3, 4))
    y = (56 - (bbox[3] - bbox[1])) // 2 - bbox[1] + int(rng.integers(-3, 4))
    draw.text((x, y), digit, fill=int(rng.integers(195, 256)), font=font)
    canvas = canvas.rotate(float(rng.uniform(-13, 13)), resample=Image.Resampling.BICUBIC)
    return np.asarray(canvas.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.uint8)


def captcha():
    missing = [str(p) for p in FONTS if not p.exists()]
    if missing:
        raise FileNotFoundError(f"CAPTCHA fonts unavailable: {missing}")
    fonts = [ImageFont.truetype(str(p), 42) for p in FONTS]
    rng = np.random.default_rng(20260915)
    codes = rng.permutation(10000)[:4000]
    arrays = {}
    offset = 0
    for split, size in (("train", 3000), ("val", 500), ("test", 500)):
        labels = np.array([[int(ch) for ch in f"{code:04d}"] for code in codes[offset:offset + size]], dtype=np.uint8)
        offset += size
        images = np.empty((size, 28, 112), dtype=np.uint8)
        for i, row in enumerate(labels):
            cells = np.concatenate([glyph(str(d), rng, fonts) for d in row], axis=1).astype(np.int16)
            # Light nuisance strokes and pixel noise, independent of the answer.
            overlay = Image.new("L", (112, 28), 0)
            draw = ImageDraw.Draw(overlay)
            for _ in range(int(rng.integers(0, 3))):
                draw.line((int(rng.integers(112)), int(rng.integers(28)),
                           int(rng.integers(112)), int(rng.integers(28))),
                          fill=int(rng.integers(15, 55)), width=1)
            noise = rng.normal(0, 8, size=(28, 112))
            images[i] = np.clip(cells + np.asarray(overlay) + noise, 0, 255).astype(np.uint8)
        key = "validation" if split == "val" else split
        arrays[f"{key}_images"] = images
        arrays[f"{key}_labels"] = labels
    np.savez_compressed(OUT / "captcha.npz", **arrays)
    sheet(arrays["train_images"], OUT / "captcha_samples.png", columns=4, limit=32)
    return {"source": "locally generated synthetic four-digit CAPTCHA", "seed": 20260915,
            "font_files": {str(p): digest(p) for p in FONTS},
            "splits": {"train": 3000, "validation": 500, "test": 500},
            "layout": "fixed four 28x28 cells in each 28x112 uint8 grayscale image; labels are four digit values; codes unique across splits",
            "limitations": "Known fixed-cell segmentation; synthetic digits, not a benchmark of arbitrary web CAPTCHAs"}


def main():
    global FONTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "fashion", "captcha"], default="all")
    parser.add_argument("--fonts", nargs="+", type=Path, help="Override default macOS font paths for synthetic CAPTCHA")
    args = parser.parse_args()
    if args.fonts:
        FONTS = args.fonts
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.dataset in ("all", "fashion"):
        manifest["fashion_mnist"] = fashion()
    if args.dataset in ("all", "captcha"):
        manifest["captcha"] = captcha()
    for name in ("fashion_mnist", "captcha"):
        path = OUT / f"{name}.npz"
        if path.exists() and name in manifest:
            manifest[name]["artifact_sha256"] = digest(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v.get("splits") for k, v in manifest.items()}))


if __name__ == "__main__":
    main()
