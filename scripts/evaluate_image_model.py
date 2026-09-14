#!/usr/bin/env python3
"""Evaluate an image bundle in one label-free Rust batch, then score predictions.

--out is a new output directory containing metrics.json and predictions.tsv.
Do not run on final holdouts until model configuration selection is complete.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from safetensors.numpy import save_file

from image_adapter import ImageRateAdapter

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def provenance_path(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def load_images(dataset, split, length):
    prefix = f'{split}_' if split else ''
    with np.load(dataset, allow_pickle=False) as data:
        images = data[prefix + 'images']
        labels = data[prefix + 'labels']
    require(images.dtype == np.uint8 and images.ndim in (3, 4) and len(images) > 0, 'Images must be nonempty uint8 arrays')
    count = len(images)
    if images.shape == (count, 28, 28 * length):
        cells = images.reshape(count, 28, length, 28).transpose(0, 2, 1, 3).reshape(-1, 28, 28)
    elif images.shape == (count, length, 28, 28):
        cells = images.reshape(-1, 28, 28)
    else:
        raise ValueError(f'Expected images [N,28,{28 * length}] or [N,{length},28,28]')
    expected_labels = (count,) if length == 1 else (count, length)
    require(labels.shape == expected_labels, f'Expected labels shape {expected_labels}')
    require(labels.dtype.kind in 'iu' and np.all((labels >= 0) & (labels < 10)), 'Labels must be integer class indices 0..9')
    return cells, labels.reshape(-1).astype(np.int64), count


def parse_predictions(stdout, count):
    reader = csv.DictReader(io.StringIO(stdout), delimiter='\t')
    require(reader.fieldnames == ['index', 'predicted'] + [f'p{i}' for i in range(10)], 'Expected exactly 10 backend classes')
    rows = list(reader)
    require(len(rows) == count, 'Unexpected number of backend predictions')
    require(all(None not in row and None not in row.values() for row in rows), 'Malformed backend prediction row')
    indices = np.array([int(row['index']) for row in rows])
    predicted = np.array([int(row['predicted']) for row in rows])
    probabilities = np.array([[float(row[f'p{i}']) for i in range(10)] for row in rows])
    require(np.array_equal(indices, np.arange(count)), 'Backend row indices are not consecutive')
    require(np.all((predicted >= 0) & (predicted < 10)), 'Backend class index out of range')
    require(np.all(np.isfinite(probabilities)) and np.all((probabilities >= 0) & (probabilities <= 1)), 'Invalid backend probabilities')
    require(np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0), 'Backend probabilities must sum to 1')
    # Printed probabilities are rounded to nine decimal places; allow rounded ties.
    require(np.all(probabilities[np.arange(count), predicted] >= probabilities.max(axis=1) - 1e-9), 'Predicted class does not maximize probability')
    return predicted, probabilities


def evaluate(bundle_path, dataset_path, split, out, binary):
    bundle_path, dataset_path, out, binary = map(Path, (bundle_path, dataset_path, out, binary))
    require(not out.exists(), f'Refusing to overwrite evaluation directory: {out}')
    bundle = json.loads(bundle_path.read_text())
    require(bundle.get('format') == 'nobi.image_bundle.v1', 'Unsupported image bundle')
    require(bundle.get('cell_size') == [28, 28], 'Unsupported cell size')
    require(len(bundle.get('class_labels', [])) == 10, 'Expected 10 class labels')
    length = bundle.get('sequence_length')
    require(type(length) is int and length > 0, 'Invalid sequence length')
    adapter_path = bundle_path.parent / bundle['adapter']
    model_path = bundle_path.parent / bundle['model']
    cells, labels, samples = load_images(dataset_path, split, length)
    start = time.perf_counter()
    adapter = ImageRateAdapter.load(adapter_path)
    features = adapter.transform(cells)
    adapter_seconds = time.perf_counter() - start
    require(features.shape == (len(cells), len(adapter.input_ids)), 'Unexpected adapter output shape')
    require(np.all(np.isfinite(features)) and np.all(features >= 0), 'Invalid adapter features')
    with tempfile.TemporaryDirectory(prefix='novi-evaluate-') as temp:
        feature_path = Path(temp) / 'inputs.safetensors'
        # Labels are deliberately absent from the backend input file.
        save_file({'inputs': features, 'input_ids': adapter.input_ids}, str(feature_path))
        start = time.perf_counter()
        result = subprocess.run([str(binary.resolve()), str(model_path.resolve()), str(feature_path)],
                                capture_output=True, text=True, check=True)
        backend_seconds = time.perf_counter() - start
    predicted, probabilities = parse_predictions(result.stdout, len(cells))
    # Answer-bearing values enter only scoring below, after all inference is done.
    correct = predicted == labels
    confidence = probabilities[np.arange(len(cells)), predicted]
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    metrics = {
        'bundle': provenance_path(bundle_path), 'dataset': provenance_path(dataset_path), 'split': split or 'unprefixed',
        'samples': samples, 'prediction_count': len(cells), 'sequence_length': length,
        'accuracy': float(correct.mean()), 'exact_sequence_accuracy': float(correct.reshape(samples, length).all(axis=1).mean()),
        'cross_entropy': float(-np.log(np.maximum(probabilities[np.arange(len(cells)), labels], 1e-15)).mean()),
        'cross_entropy_note': 'Natural log of backend probabilities printed to 9 decimals, floored at 1e-15',
        'confidence': {'mean': float(confidence.mean()), 'min': float(confidence.min()), 'max': float(confidence.max()),
                       'correct_mean': float(confidence[correct].mean()) if correct.any() else None,
                       'incorrect_mean': float(confidence[~correct].mean()) if (~correct).any() else None},
        'confusion_matrix': confusion.tolist(), 'confusion_orientation': 'rows=true, columns=predicted',
        'class_labels': bundle['class_labels'],
        'adapter_seconds': adapter_seconds, 'backend_seconds': backend_seconds,
        'adapter_backend_seconds': adapter_seconds + backend_seconds,
        'timing_scope': 'Adapter load+transform and one Rust batch process; excludes dataset I/O, feature serialization and scoring',
        'model_bytes': model_path.stat().st_size, 'adapter_bytes': adapter_path.stat().st_size,
        'bundle_json_bytes': bundle_path.stat().st_size,
        'bundle_total_bytes': model_path.stat().st_size + adapter_path.stat().st_size + bundle_path.stat().st_size,
    }
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'predictions.tsv').open('w', newline='') as output:
        writer = csv.writer(output, delimiter='\t', lineterminator='\n')
        writer.writerow(['index', 'sample', 'position', 'true', 'predicted', 'correct', 'confidence'] + [f'p{i}' for i in range(10)])
        for index in range(len(cells)):
            writer.writerow([index, index // length, index % length, int(labels[index]), int(predicted[index]),
                             int(correct[index]), f'{confidence[index]:.9f}'] + [f'{p:.9f}' for p in probabilities[index]])
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default=None)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'target' / 'release' / 'classify_features')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.bundle, args.dataset, args.split, args.out, args.binary), indent=2))


if __name__ == '__main__':
    main()
