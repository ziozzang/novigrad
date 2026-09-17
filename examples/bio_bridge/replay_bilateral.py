#!/usr/bin/env python3
"""Replay frozen bilateral artifacts with content-verified relocated base models.

Only model-directory lookup is overridden in memory. Locked files, inference
code, precision, device defaults, comparisons and tolerances remain unchanged.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {'.safetensors', '.json', '.txt', '.model'}


def relative_name(name):
    p = PurePosixPath(name)
    if not isinstance(name, str) or not name or '\\' in name or p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')) or ':' in name:
        raise ValueError(f'unsafe relative path: {name}')
    return p


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_model_directory(path, expected):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError(f'model directory missing: {path}')
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.suffix in SUFFIXES}
    if actual != set(expected):
        raise ValueError(f'model inventory mismatch: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}')
    for name, checksum in expected.items():
        relative_name(name)
        if digest(path / name) != checksum:
            raise ValueError(f'model content mismatch: {name}')
    return path


@contextmanager
def relocated_models(study, lm, paths):
    original_check = study.check_models
    original_load = lm.Runtime.__dict__['load']
    loader = lm.Runtime.load
    def checked(models):
        if set(models) != set(paths):
            raise ValueError('unexpected model keys')
        for key, record in models.items():
            verify_model_directory(paths[key], record['files'])
    def load(cls, model_path=None, device='mps'):
        if model_path is not None and Path(model_path).resolve() not in (Path(lm.MODEL_PATH).resolve(), paths['function']):
            raise ValueError('unverified function model override')
        return loader(model_path=paths['function'], device=device)
    study.check_models = checked
    lm.Runtime.load = classmethod(load)
    lm.Runtime._cache.clear()
    try:
        yield
    finally:
        study.check_models = original_check
        lm.Runtime.load = original_load
        lm.Runtime._cache.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding-model', required=True, type=Path)
    parser.add_argument('--function-model', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true', help='verify frozen files and relocated model hashes without inference')
    args = parser.parse_args()
    import bilateral_study as study
    import bilateral_lm as lm
    lock = json.loads((study.OUT / 'protocol-lock.json').read_text())
    paths = {key: verify_model_directory(path, lock['models'][key]['files']) for key,path in
             [('embedding',args.embedding_model),('function',args.function_model)]}
    with relocated_models(study,lm,paths):
        study.checked_lock()
        if args.check_only:
            evaluation = study.OUT / 'evaluation-lock.json'
            if evaluation.exists():study.check_files(json.loads(evaluation.read_text())['hashes'])
            print('Frozen artifacts and relocated model contents verified; no inference.')
        else:
            study.verify()
    print(json.dumps({'protocol_sha256': digest(study.OUT/'protocol-lock.json'),
        'model_paths': {k:str(v) for k,v in paths.items()}, 'check_only':args.check_only,
        'boundary':'Lookup-only override; original model contents, MPS/BF16 runtime and replay tolerances preserved.'}))


if __name__ == '__main__':main()
