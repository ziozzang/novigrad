#!/usr/bin/env python3
"""Fixed, validation-selected offline three-action steering experiment.

Input NPZ contains train/validation/test_images uint8 [N,28,28], labels
int64 0=left, 1=straight, 2=right, and continuous steer values. Optional
*_groups must be disjoint across splits to protect a sequence holdout.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from image_adapter import prepare  # noqa: E402

SPLITS = ('train', 'validation', 'test')
CLASS_NAMES = ('left', 'straight', 'right')
CONFIGS = (('homeostasis_on', True), ('homeostasis_off', False))


def validated_data(path):
    data = np.load(path, allow_pickle=False)
    counts = {}
    groups = {}
    for split in SPLITS:
        x = data[f'{split}_images']
        y = data[f'{split}_labels']
        steer = data[f'{split}_steer']
        if x.dtype != np.uint8 or x.ndim != 3 or x.shape[1:] != (28, 28):
            raise ValueError(f'{split}_images must be uint8 [N,28,28]')
        if y.dtype.kind not in 'iu' or y.shape != (len(x),) or not np.isin(y, (0, 1, 2)).all():
            raise ValueError(f'{split}_labels must be integer classes 0,1,2')
        if steer.shape != (len(x),) or not np.issubdtype(steer.dtype, np.number) or not np.isfinite(steer).all():
            raise ValueError(f'{split}_steer must be finite numeric [N]')
        if np.any((steer < -1.0) | (steer > 1.0)):
            raise ValueError(f'{split}_steer must be in [-1,1]')
        expected = np.where(steer < -0.2, 0, np.where(steer > 0.2, 2, 1))
        if not np.array_equal(y, expected):
            raise ValueError(f'{split}_labels must match steer bins at ±0.2')
        if len(x) == 0:
            raise ValueError(f'{split} must be nonempty')
        counts[split] = len(x)
        key = f'{split}_groups'
        if key in data:
            if data[key].shape != (len(x),) or data[key].dtype.kind not in 'US':
                raise ValueError(f'{key} must be strings [N]')
            groups[split] = set(map(str, data[key]))
    if groups and set(groups) != set(SPLITS):
        raise ValueError('group IDs must be provided for every split or none')
    if groups:
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                if groups[a] & groups[b]:
                    raise ValueError(f'group leakage between {a} and {b}')
    train_y = data['train_labels'].astype(int)
    means = []
    for label in range(3):
        values = data['train_steer'][train_y == label]
        if not len(values):
            raise ValueError(f'training class {label} has no continuous steer values')
        means.append(float(np.mean(values)))
    return data, {'split_counts': counts, 'group_counts': {k: len(v) for k, v in groups.items()},
                  'train_class_steer_mean': means,
                  'train_constant_median_steer': float(np.median(data['train_steer']))}


def run_training(binary, features, out, mode, homeostasis):
    command = [str(binary), str(features), str(ROOT / 'data/pn_kc.tsv'),
               str(ROOT / 'data/kc_mbon.tsv'), str(out),
               '--epochs', '25', '--patience', '5', '--lr', '0.0001', '--gain', '12',
               '--active-fraction', '0.2', '--actions', '3', '--seed', '1',
               '--mode', mode, '--phase', 'tune', '--readout', 'opponent',
               '--homeostasis', str(homeostasis).lower(), '--batch-size', '1']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'stdout.log').write_text(result.stdout)
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    metrics = json.loads((out / 'metrics.json').read_text())
    if metrics['phase'] != 'tune' or metrics['test_accuracy'] is not None:
        raise RuntimeError('training command accessed the test phase')
    return metrics


def predictions(binary, model, features, output):
    result = subprocess.run([str(binary), str(model), str(features)], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    output.write_text(result.stdout)
    lines = result.stdout.splitlines()
    if lines[0] != 'index\tpredicted\tp0\tp1\tp2':
        raise RuntimeError('unexpected classify_features header')
    rows = [line.split('\t') for line in lines[1:]]
    if any(len(row) != 5 or int(row[0]) != i for i, row in enumerate(rows)):
        raise RuntimeError('invalid inference row ordering')
    predicted = np.array([int(row[1]) for row in rows], dtype=np.int64)
    probabilities = np.array([[float(v) for v in row[2:]] for row in rows], dtype=np.float64)
    if (not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            or not np.array_equal(predicted, probabilities.argmax(axis=1))):
        raise RuntimeError('invalid output probabilities')
    return predicted, probabilities


def summarize(data, settings, predicted, probabilities):
    truth = data['test_labels'].astype(np.int64)
    actual_steer = data['test_steer'].astype(np.float64)
    if len(predicted) != len(truth):
        raise RuntimeError('inference sample count mismatch')
    matrix = np.zeros((3, 3), dtype=int)
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess] += 1
    recalls = [float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None for i in range(3)]
    balanced = float(np.mean([r for r in recalls if r is not None]))
    means = np.asarray(settings['train_class_steer_mean'], dtype=np.float64)
    decoded = probabilities @ means
    median = settings['train_constant_median_steer']
    result = {'accuracy': float(np.mean(predicted == truth)), 'balanced_accuracy': balanced,
              'per_class_recall': recalls, 'confusion_rows_actual_columns_predicted': matrix.tolist(),
              'probability_decoded_steer_mae': float(np.mean(np.abs(decoded - actual_steer))),
              'train_median_constant_steer_mae': float(np.mean(np.abs(median - actual_steer)))}
    if 'test_groups' in data:
        groups = data['test_groups'].astype(str)
        result['per_group'] = {name: {'count': int(np.sum(groups == name)),
                                      'accuracy': float(np.mean(predicted[groups == name] == truth[groups == name]))}
                               for name in sorted(set(groups))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'target/release',
                        help='directory containing train_classifier and classify_features')
    parser.add_argument('--build', action='store_true', help='build release binaries into --binary-dir')
    args = parser.parse_args()
    data_path = args.data.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'output directory already contains experiment records: {out}')
    out.mkdir(parents=True, exist_ok=True)
    data, settings = validated_data(data_path)
    binary_dir = args.binary_dir.resolve()
    if args.build:
        if binary_dir.name != 'release':
            raise ValueError('--build requires --binary-dir ending in release')
        env = os.environ.copy()
        env['CARGO_TARGET_DIR'] = str(binary_dir.parent)
        subprocess.run(['cargo', 'build', '--release', '--bin', 'train_classifier', '--bin', 'classify_features'],
                       cwd=ROOT, env=env, check=True)
    trainer = binary_dir / 'train_classifier'
    infer = binary_dir / 'classify_features'
    if not trainer.exists() or not infer.exists():
        raise FileNotFoundError('steering experiment binaries absent; pass --build or --binary-dir')
    feature_dir = out / 'features'
    prepare(data_path, ROOT / 'data/pn_kc.tsv', feature_dir, components=64, whitening=0.5)
    features = feature_dir / 'features.safetensors'
    candidates = []
    for name, homeostasis in CONFIGS:
        folder = out / name
        metrics = run_training(trainer, features, folder, 'supervised', homeostasis)
        candidates.append((name, homeostasis, metrics))
    # The untouched test split is materialized for inference only after selection.
    selected = max(candidates, key=lambda item: (item[2]['validation_accuracy'],
                                                 -item[2]['validation_cross_entropy']))
    selected_name, selected_homeostasis, selected_metrics = selected
    selection = {'selected': selected_name, 'criterion': 'highest validation accuracy, then lowest validation cross-entropy',
                 'candidates': {name: {'homeostasis': h, 'baseline_validation_accuracy': m['baseline_validation_accuracy'],
                                      'validation_accuracy': m['validation_accuracy'],
                                      'validation_cross_entropy': m['validation_cross_entropy'],
                                      'best_epoch': m['best_epoch']}
                                for name, h, m in candidates}}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    control_dir = out / 'shuffled_control'
    control_metrics = run_training(trainer, features, control_dir, 'shuffled', selected_homeostasis)
    bundle = out / 'bundle'
    bundle.mkdir(exist_ok=True)
    shutil.copy2(feature_dir / 'adapter.safetensors', bundle / 'adapter.safetensors')
    shutil.copy2(out / selected_name / 'model.safetensors', bundle / 'model.safetensors')
    bundle_json = {'format': 'nobi.image_bundle.v1', 'adapter': 'adapter.safetensors',
                   'model': 'model.safetensors', 'sequence_length': 1, 'cell_size': [28, 28],
                   'segmentation': 'whole image', 'class_labels': list(CLASS_NAMES),
                   'components': json.loads((feature_dir / 'bundle.json').read_text())['components'], 'whitening': 0.5,
                   'train_class_steer_mean': settings['train_class_steer_mean'],
                   'steer_decoding': 'probability-weighted training class means'}
    (bundle / 'bundle.json').write_text(json.dumps(bundle_json, indent=2) + '\n')
    from safetensors import safe_open
    with safe_open(str(features), framework='numpy') as source:
        input_ids = source.get_tensor('input_ids')
        test_inputs = source.get_tensor('test_inputs')
    inference_file = out / 'test_inputs.safetensors'
    save_file({'input_ids': input_ids, 'inputs': test_inputs}, str(inference_file))
    chosen_pred, chosen_prob = predictions(infer, bundle / 'model.safetensors', inference_file,
                                           out / 'selected_test_predictions.tsv')
    shuffled_pred, shuffled_prob = predictions(infer, control_dir / 'model.safetensors', inference_file,
                                               out / 'shuffled_test_predictions.tsv')
    summary = {'task': 'offline_three_action_steering', 'data': str(data_path),
               'settings': settings, 'validation_candidates':
               {name: {'homeostasis': h, 'baseline_validation_accuracy': m['baseline_validation_accuracy'],
                       'validation_accuracy': m['validation_accuracy'],
                       'validation_cross_entropy': m['validation_cross_entropy'], 'best_epoch': m['best_epoch']}
                for name, h, m in candidates},
               'selected': selected_name, 'selected_best_epoch': selected_metrics['best_epoch'],
               'shuffled_validation_accuracy': control_metrics['validation_accuracy'],
               'train_majority_test_accuracy': float(np.mean(data['test_labels'] == np.bincount(data['train_labels'].astype(int), minlength=3).argmax())),
               'test': summarize(data, settings, chosen_pred, chosen_prob),
               'shuffled_test': summarize(data, settings, shuffled_pred, shuffled_prob),
               'test_selection_policy': 'validation accuracy, cross-entropy tie break; one test inference after selection'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'selected': selected_name, 'test_accuracy': summary['test']['accuracy'],
                      'test_balanced_accuracy': summary['test']['balanced_accuracy'],
                      'test_steer_mae': summary['test']['probability_decoded_steer_mae'],
                      'shuffled_test_accuracy': summary['shuffled_test']['accuracy'],
                      'summary': str(out / 'summary.json')}))


if __name__ == '__main__':
    main()
