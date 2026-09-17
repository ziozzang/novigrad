"""Frozen, load-only architecture comparison and adversarial replay orchestration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import importlib.metadata
import numpy as np
from safetensors.numpy import load_file, save_file
from precise_bridge import ROOT, dataset
from inhibition_mechanism import checkpoint
from encode_precise_holdout import validate_cases

OUT = ROOT / 'results/architecture-bridge'
HERE = Path(__file__).resolve().parent
SETS = {'holdout': HERE / 'architecture_holdout.json',
        'adversarial': HERE / 'architecture_adversarial.json'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def hashes(paths):
    return {str(Path(p).resolve().relative_to(ROOT.resolve())): sha(p) for p in paths}


def check_hashes(expected):
    for name, digest in expected.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'changed frozen artifact: {name}')


def develop():
    import architecture_probe as probe
    import architecture_adapter as adapter
    import architecture_sites as sites
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('development output must be empty')
    OUT.mkdir(parents=True, exist_ok=True)
    data = dataset()
    x, y, _ = data['train']
    v, vy, _ = data['validation']
    # Baseline was measured once at launch; save the fixed rule, weights and result.
    w = x.T @ np.linalg.solve(x @ x.T + .1 * np.eye(len(x)), np.eye(4)[y])
    save_file({'weight': np.ascontiguousarray(w)}, str(OUT / 'direct.safetensors'))
    write(OUT / 'baseline.json', {'boundary': 'Reused validation12 development diagnostic',
          'accuracy': float(np.mean((v @ w).argmax(1) == vy)), 'ridge': .1})
    probe.save_bundle(OUT / 'probes')
    write(OUT / 'probe-development.json', probe.evaluate_saved(OUT / 'probes', v, vy))
    sites.save_sites(OUT / 'sites', x, y)
    write(OUT / 'sites-development.json', sites.evaluate_sites(OUT / 'sites', v, vy))
    write(OUT / 'mixing-sanity.json', sites.mixing_sanity())
    adapter.train_all(OUT / 'adapters')
    print('development complete', flush=True)


def freeze():
    if (OUT / 'protocol-lock.json').exists():
        raise FileExistsError('already frozen')
    for path in SETS.values():
        validate_cases(json.loads(path.read_text()))
    code = [*HERE.glob('architecture*.py'), *HERE.glob('test_architecture*.py')]
    code += [HERE / name for name in ('precise_bridge.py', 'inhibition_mechanism.py',
              'thought_embedding.py', 'delayed_credit.py', 'encode_precise_holdout.py')]
    inputs = [ROOT / 'results/gemma-bridge/dataset.json', ROOT / 'results/gemma-bridge/encoder.json',
              ROOT / 'results/gemma-bridge/embeddings.safetensors', checkpoint(601), *SETS.values()]
    artifacts = [p for p in OUT.rglob('*') if p.is_file()]
    required = [OUT / name for name in ('adapters/report.json', 'probe-development.json',
                'sites-development.json', 'sites/sites.safetensors', 'sites/sites.json',
                'probes/architecture-probes.safetensors', 'probes/architecture-probe-bundle.json',
                'direct.safetensors', 'baseline.json', 'mixing-sanity.json')]
    required += [OUT / 'adapters' / f'{kind}-seed{seed}.safetensors'
                 for kind in ('frozen', 'rank4', 'rank16', 'rank64', 'full') for seed in (11, 23, 47)]
    if not all(p.exists() for p in required):
        raise ValueError('finish development before freezing')
    import architecture_probe as probe
    import architecture_adapter as adapter
    import architecture_sites as sites
    v, vy, _ = dataset()['validation']
    probe.evaluate_saved(OUT / 'probes', v, vy)
    adapter.evaluate_all(OUT / 'adapters', v, vy)
    sites.evaluate_sites(OUT / 'sites', v, vy)
    encoder = json.loads((ROOT / 'results/gemma-bridge/encoder.json').read_text())
    model_path = Path(encoder['model'])
    model_hashes = {str(p.relative_to(model_path)): sha(p) for p in sorted(model_path.rglob('*'))
                    if p.is_file() and p.suffix in ('.safetensors', '.json', '.model', '.txt')}
    if {k: v for k, v in model_hashes.items() if k.endswith('.safetensors')} != encoder['weight_sha256']:
        raise ValueError('current encoder weights differ from training provenance')
    lock = {'version': 1, 'primary_comparisons': [
        'KC interaction versus coefficient-count-matched interaction, additive and direct embedding',
        'Readout sites and fixed landmark pooling at 32 readout features',
        'Linear bridge parameterization frozen/rank4/rank16/rank64/full, not encoder fine-tuning'],
        'acceptance': 'No quantum-jump claim without beating strong direct baseline and surviving counterexamples. Retain reproducible negative evidence.',
        'final_policy': 'Evaluate both preauthored sets once after lock; no selection or retuning. Verify only replays saved weights.',
        'statistical_unit': '32 English/Korean scenario pairs per set, grouped in 8 families; context rows and model seeds not independent subjects.',
        'frozen': hashes(code + inputs + artifacts),
        'required_inventory': [str(p.relative_to(ROOT)) for p in required],
        'encoder_model_files': model_hashes,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'scipy', 'torch', 'sentence-transformers', 'transformers', 'safetensors')},
        'selection': 'All variants retained; fixed hyperparameters; no validation winner chosen.'}
    write(OUT / 'protocol-lock.json', lock)
    print('protocol frozen', sha(OUT / 'protocol-lock.json'), flush=True)


def checked_lock():
    path = OUT / 'protocol-lock.json'
    lock = json.loads(path.read_text())
    check_hashes(lock['frozen'])
    return lock


def encode():
    checked_lock()
    for name, path in SETS.items():
        subprocess.run([sys.executable, str(HERE / 'encode_precise_holdout.py'),
                        '--cases', str(path), '--out', str(OUT / name)], check=True)


def input_set(name):
    case_path = SETS[name]
    rows = validate_cases(json.loads(case_path.read_text()))
    features = OUT / name / 'holdout-embeddings.safetensors'
    p = json.loads((OUT / name / 'provenance.json').read_text())
    if p['cases_sha256'] != sha(case_path) or p['embedding_sha256'] != sha(features):
        raise ValueError('case or embedding provenance mismatch')
    if p['row_ids'] != [r['id'] for r in rows]:
        raise ValueError('case row order mismatch')
    encoder = json.loads((ROOT / 'results/gemma-bridge/encoder.json').read_text())
    lock = json.loads((OUT / 'protocol-lock.json').read_text())
    if p['weight_sha256'] != encoder['weight_sha256'] or p['prompt_name'] != encoder['prompt_name'] or p['parameter_dtypes'] != encoder['parameter_dtypes']:
        raise ValueError('encoder training/final provenance mismatch')
    if p['model_file_sha256'] != lock['encoder_model_files']:
        raise ValueError('encoder model/config differs from frozen revision')
    tensors = load_file(str(features))
    if set(tensors) != {'embeddings'}:
        raise ValueError('unexpected feature schema')
    x = tensors['embeddings']
    if x.shape != (64, 768) or not np.isfinite(x).all():
        raise ValueError('invalid feature shape/value')
    y = np.array([('water', 'food', 'warmth', 'rest').index(r['class']) for r in rows])
    return rows, x, y


def compute(name):
    import architecture_probe as probe
    import architecture_adapter as adapter
    import architecture_sites as sites
    rows, x, y = input_set(name)
    w = load_file(str(OUT / 'direct.safetensors'))['weight']
    pred = (x @ w).argmax(1)
    return {'set': name, 'case_ids': [r['id'] for r in rows], 'labels': y.tolist(),
            'direct_embedding': {'accuracy': float(np.mean(pred == y)), 'predictions': pred.tolist()},
            'probes': probe.evaluate_saved(OUT / 'probes', x, y),
            'sites': sites.evaluate_sites(OUT / 'sites', x, y),
            'adapters': adapter.evaluate_all(OUT / 'adapters', x, y)}


def final():
    checked_lock()
    if (OUT / 'evaluation-lock.json').exists() or any((OUT / f'{n}-final.json').exists() for n in SETS):
        raise FileExistsError('final already evaluated; use verify for replay')
    # Validate both sets and row identities before either output is inspected.
    for name in SETS:
        input_set(name)
    for name in SETS:
        write(OUT / f'{name}-final.json', compute(name))
    paths = [OUT / 'protocol-lock.json']
    paths += [p for n in SETS for p in (OUT / n).iterdir() if p.is_file()]
    paths += [OUT / f'{name}-final.json' for name in SETS]
    write(OUT / 'evaluation-lock.json', {'hashes': hashes(paths)})
    print('both frozen final evaluations complete', flush=True)


def verify():
    checked_lock()
    evaluation = json.loads((OUT / 'evaluation-lock.json').read_text())
    check_hashes(evaluation['hashes'])
    exact = {}
    for name in SETS:
        expected = json.loads((OUT / f'{name}-final.json').read_text())
        exact[name] = compute(name) == expected
        if not exact[name]:
            raise AssertionError(f'saved inference replay differs: {name}')
    write(OUT / 'verification.json', {'protocol_sha256': sha(OUT / 'protocol-lock.json'),
          'evaluation_sha256': sha(OUT / 'evaluation-lock.json'), 'saved_inference_exact': exact,
          'boundary': 'Inference replay of saved models, not bitwise retraining on MPS or biological replay.'})
    print(json.dumps(exact), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('develop', 'freeze', 'encode', 'final', 'verify'))
    args = parser.parse_args()
    globals()[args.stage]()
