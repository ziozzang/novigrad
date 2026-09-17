#!/usr/bin/env python3
"""Replay frozen native/permutation finals and train/validation-only permutation fits."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from safetensors.numpy import load_file

import precise_native as native
import precise_permutation as permutation
import precise_pipeline as pipeline

ROOT = pipeline.ROOT
OUT = pipeline.OUT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_runtime(value):
    if isinstance(value, dict):
        return {key: without_runtime(item) for key, item in value.items() if key != 'batch_seconds'}
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def require_equal(actual, expected, name):
    if actual != expected:
        raise AssertionError(f'{name} replay differs')


def main():
    protocol_hash = pipeline.verify()
    originals = {str(path.relative_to(ROOT)): sha256(path) for path in OUT.rglob('*')
                 if path.is_file() and path.name != 'extended-replay.json'}
    original_native = json.loads((OUT / 'native-final.json').read_text())
    original_permutation = json.loads((OUT / 'permutation-final.json').read_text())
    tensor_checks = {}
    # Keep the temporary tree under ROOT because frozen scripts require relative_to(ROOT).
    with tempfile.TemporaryDirectory(prefix='.extended-precise-replay-', dir=ROOT) as directory:
        temporary = Path(directory)
        copied = temporary / 'artifacts'
        shutil.copytree(OUT, copied)
        # The native manifest records repository paths. Point ONLY the copied
        # manifest at byte-identical copied mappings/checkpoints for isolated loads.
        manifest_path = copied / 'native-development.json'
        manifest = json.loads(manifest_path.read_text())
        records = list(manifest['mappings'].values()) + [row['checkpoint'] for row in manifest['runs']]
        for record in records:
            source = (ROOT / record['path']).resolve()
            destination = copied / source.relative_to(OUT.resolve())
            if sha256(destination) != record['sha256']:
                raise AssertionError('copied native artifact hash mismatch')
            record['path'] = str(destination.relative_to(ROOT))
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        replay_native = native.run_final(copied)
        require_equal(replay_native['runs'], original_native['runs'], 'native runs')
        require_equal(replay_native['aggregate'], original_native['aggregate'], 'native aggregate')
        replay_permutation = permutation.run_final(copied)
        require_equal(without_runtime(replay_permutation['results']),
                      without_runtime(original_permutation['results']), 'permutation final results')

        refit = temporary / 'refit'
        refit.mkdir()
        for mode in permutation.MODES:
            shutil.copy2(OUT / f'ports-{mode}.safetensors', refit / f'ports-{mode}.safetensors')
        permutation.run_development(refit)
        for mode in permutation.MODES:
            for seed in permutation.SEEDS:
                name = f'perm-{mode}-{seed}.safetensors'
                before, after = load_file(str(OUT / name)), load_file(str(refit / name))
                require_equal(set(before), set(after), f'{name} tensor keys')
                for key in before:
                    if before[key].dtype != after[key].dtype or before[key].shape != after[key].shape:
                        raise AssertionError(f'{name}:{key} dtype or shape differs')
                    np.testing.assert_array_equal(before[key], after[key], err_msg=f'{name}:{key}')
                tensor_checks[name] = {'all_arrays_exact': True, 'tensor_names': sorted(before),
                                      'original_sha256': sha256(OUT / name), 'refit_sha256': sha256(refit / name)}
        if len(tensor_checks) != 15:
            raise AssertionError('expected fifteen permutation artifacts')
    require_equal(pipeline.verify(), protocol_hash, 'protocol after replay')
    for path, expected in originals.items():
        require_equal(sha256(ROOT / path), expected, f'unchanged artifact {path}')
    report = {
        'purpose': 'Replay only; no new final selection or holdout fitting.',
        'protocol_sha256': protocol_hash,
        'evaluation_lock_sha256': sha256(OUT / 'evaluation-lock.json'),
        'generator': str(Path(__file__).resolve().relative_to(ROOT)), 'generator_sha256': sha256(__file__),
        'native_runs_exact': True, 'native_aggregate_exact': True,
        'permutation_final_results_exact_excluding_batch_seconds': True,
        'permutation_refit_artifact_count': len(tensor_checks), 'permutation_refit_arrays_exact': True,
        'permutation_refit_inputs': 'Original saved ports plus original train32 and validation12 only.',
        'permutation_refit_artifacts': tensor_checks,
        'original_artifacts_unchanged': True, 'original_sha256': originals,
        'passed': True,
    }
    destination = OUT / 'extended-replay.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(destination.relative_to(ROOT)), 'sha256': sha256(destination),
                      'passed': True, 'permutation_artifacts_exact': len(tensor_checks)}))


if __name__ == '__main__':
    main()
