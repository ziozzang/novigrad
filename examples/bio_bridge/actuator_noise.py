#!/usr/bin/env python3
"""Predeclared abstract counting-noise diagnostic; no fitting or biological window claim."""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/actuator-noise'
COUNTS = (32, 128, 512, 2048, 8192)
SEEDS = (1901, 1902, 1903)
REPLICATES = 256
DESIGN = {'expected_total_counts': list(COUNTS), 'seeds': list(SEEDS), 'replicates_per_class': REPLICATES,
          'ports': 319, 'classes': ['water', 'food', 'warmth', 'rest'], 'batch_size': 256,
          'sampling': 'independent Poisson per port; lambda = total * PN / sum(PN)',
          'normalization': 'observed counts / observed total * original PN sum; float32',
          'zero_count': 'retain all-zero PN input; record explicitly; never retry',
          'selection': 'all conditions fixed before measurement; no tuning or fitting',
          'boundary': 'Abstract count observation noise on cached engineered actuator maps; not validated neural sampling, biological time windows, real fly data or semantic generalization.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_counts(pn, total, replicates, seed):
    pn = np.asarray(pn, np.float64)
    if pn.ndim != 2 or not np.isfinite(pn).all() or np.any(pn < 0) or np.any(pn.sum(1) <= 0):
        raise ValueError('PN must be finite nonnegative rows with positive sums')
    if total <= 0 or not np.isfinite(total) or replicates < 1:
        raise ValueError('positive count expectation and replicate count required')
    # Independent deterministic stream for each fixed count condition; same noise for both policies.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, int(total)])))
    rates = np.repeat(pn, replicates, axis=0)
    sums = rates.sum(1, keepdims=True)
    counts = rng.poisson(total * rates / sums).astype(np.int64)
    observed = counts.sum(1, keepdims=True)
    normalized = counts / np.maximum(observed, 1) * sums
    return counts, normalized.astype(np.float32), (observed[:, 0] == 0)


def summarize(probabilities, intended, reference):
    p = np.asarray(probabilities)
    if p.shape != (len(intended), 4) or not np.isfinite(p).all():
        raise ValueError('invalid native probability shape or values')
    pred = p.argmax(1)
    order = np.sort(p, axis=1)
    competitor = p.copy(); competitor[np.arange(len(p)), intended] = -np.inf
    margins = p[np.arange(len(p)), intended] - competitor.max(1)
    def dist(x):
        return dict(zip(('min', 'q05', 'median', 'q95', 'max'), map(float, np.quantile(x, [0, .05, .5, .95, 1]))))
    return {'n': len(p), 'intended_agreement': float(np.mean(pred == intended)),
            'unperturbed_action_agreement': float(np.mean(pred == reference)),
            'winning_probability_margin': dist(order[:, -1] - order[:, -2]),
            'intended_probability_margin': dist(margins)}


def sources():
    import novigrad
    lock_path = ROOT / 'results/bilateral-bridge/protocol-lock.json'
    original = json.loads(lock_path.read_text())
    files = dict(original['frozen'])
    for name, expected in files.items():
        if sha(ROOT / name) != expected: raise ValueError(f'frozen source mismatch: {name}')
    paths = [Path(__file__), lock_path, ROOT / 'Cargo.lock', ROOT / 'Cargo.toml']
    paths += list((ROOT / 'src').rglob('*.rs')) + list((ROOT / 'bindings/python').rglob('*.rs'))
    paths += list(Path(novigrad.__file__).parent.glob('*.so'))
    for p in paths:
        name = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
        files[name] = sha(p)
    return files


def compute():
    from novigrad import Engine
    from bilateral_temporal import SiteCodec
    from bilateral_closed_loop import prototypes_from_old_train
    from bilateral_policy_swap import BASE_POLICY, PCA_POLICY, verify_policy_slot
    verify_policy_slot()
    prototypes = prototypes_from_old_train()
    pn = SiteCodec.load(ROOT / 'results/bilateral-bridge/temporal/codec.safetensors').bridge.encode(prototypes)
    if pn.shape != (4, 319): raise ValueError('unexpected PN shape')
    engines = {'native601': Engine.load(BASE_POLICY), 'native701': Engine.load(PCA_POLICY)}
    arrays = {'prototype_embeddings': prototypes, 'prototype_pn': pn}
    baseline = {}; reference = {}
    def infer(engine, x):
        return np.concatenate([np.asarray(engine.infer_batch(x[i:i+256].tolist()), np.float64)
                               for i in range(0, len(x), 256)])
    for name, engine in engines.items():
        p = infer(engine, pn); reference[name] = p.argmax(1)
        arrays[name + '_baseline_probabilities'] = p
        baseline[name] = {'probabilities': p.tolist(), 'actions': reference[name].tolist(),
                          'metrics': summarize(p, np.arange(4), reference[name])}
    records = []; intended = np.repeat(np.arange(4), REPLICATES)
    arrays['intended_class'] = intended.astype(np.int64)
    for total in COUNTS:
        for seed in SEEDS:
            key = f'counts{total}_seed{seed}'
            counts, inputs, zeros = sample_counts(pn, total, REPLICATES, seed)
            arrays[key + '_counts'] = counts; arrays[key + '_inputs'] = inputs
            arrays[key + '_zero_count'] = zeros
            for name, engine in engines.items():
                p = infer(engine, inputs); arrays[key + '_' + name + '_probabilities'] = p
                records.append({'expected_total': total, 'seed': seed, 'policy': name,
                    'zero_count_rows': np.flatnonzero(zeros).tolist(),
                    'per_class': [summarize(p[intended == c], intended[intended == c],
                                           np.full(REPLICATES, reference[name][c])) for c in range(4)],
                    'all_classes': summarize(p, intended, reference[name][intended])})
    return arrays, {'design': DESIGN, 'baseline': baseline, 'conditions': records}


def run(out):
    if out.exists(): raise FileExistsError(f'refusing overwrite: {out}')
    files = sources()
    out.mkdir(parents=True)
    lock = {'design': DESIGN, 'files': files, 'python': platform.python_version(), 'numpy': np.__version__}
    (out / 'source-lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    arrays, report = compute()
    if sources() != files: raise ValueError('sources changed during run')
    artifact = out / 'observations.safetensors'
    save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, str(artifact))
    report['artifact_sha256'] = sha(artifact)
    report['source_lock_sha256'] = sha(out / 'source-lock.json')
    report['tensor_sha256'] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arrays.items()}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


def verify(out):
    lock = json.loads((out / 'source-lock.json').read_text())
    if lock['design'] != DESIGN or lock['files'] != sources(): raise ValueError('source/design mismatch')
    if lock['numpy'] != np.__version__ or lock['python'] != platform.python_version(): raise ValueError('runtime version mismatch')
    report = json.loads((out / 'report.json').read_text())
    if report['source_lock_sha256'] != sha(out / 'source-lock.json') or report['artifact_sha256'] != sha(out / 'observations.safetensors'):
        raise ValueError('artifact hash mismatch')
    saved = load_file(str(out / 'observations.safetensors')); arrays, replay = compute()
    if set(saved) != set(arrays): raise ValueError('tensor inventory mismatch')
    for key, value in arrays.items():
        if saved[key].dtype != value.dtype or not np.array_equal(saved[key], value): raise ValueError(f'exact replay mismatch: {key}')
        if report['tensor_sha256'][key] != hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(): raise ValueError(f'tensor hash mismatch: {key}')
    if any(report[k] != replay[k] for k in replay): raise ValueError('report replay mismatch')
    print(json.dumps({'exact_replay': True, 'tensors': len(arrays), 'artifact_sha256': report['artifact_sha256']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'verify'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    (run if args.command == 'run' else verify)(args.out)

if __name__ == '__main__': main()
