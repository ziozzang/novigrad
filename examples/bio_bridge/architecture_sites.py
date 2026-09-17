"""Read-only site probes and constrained mixing sanity checks, not neural write-in.

All supervised readouts have 32 features (+bias), fit train32 only. Random
projections are fixed, untrained; landmark pooling is not learned token attention.
The MBON checkpoint was trained previously on the same development training set.
"""
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file, load_file
from safetensors import safe_open
from precise_bridge import PortBridge
from inhibition_mechanism import ShadowEngine, checkpoint

SEEDS = (1201, 1202, 1203)
WIDTH = 32


def unit(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def features(bridge, shadow, x):
    pn = bridge.encode(x)
    raw = shadow.raw_hidden(pn)
    kc = shadow.inhibit(raw, 'native_topk_2', {})
    mbon = np.asarray(shadow.plastic_matrix @ kc.T).T
    return {'embedding': x, 'pn': pn, 'raw_kc': raw, 'kc': kc, 'mbon': mbon}


def projected(x, seed, width=WIDTH):
    rng = np.random.default_rng(seed)
    p = rng.choice([-1., 1.], (x.shape[1], width)) / np.sqrt(width)
    return unit(unit(x) @ p)


def designs(reps, seed):
    result = {name: projected(x, seed) for name, x in reps.items()}
    sites = ('pn', 'raw_kc', 'kc', 'mbon')
    result['multi_site'] = unit(np.concatenate([
        projected(reps[name], seed + i, 8) for i, name in enumerate(sites)], axis=1))
    result['repeated_kc'] = unit(np.concatenate([
        projected(reps['kc'], seed + i, 8) for i in range(4)], axis=1))
    # Fixed landmark tokens: sum activation in deterministic disjoint groups.
    # Neuron positions are software array indices, not learned anatomical tokens.
    for count in (4, 16, 64):
        order = np.random.default_rng(seed).permutation(reps['kc'].shape[1])
        groups = np.array_split(order, count)
        pooled = np.stack([reps['kc'][:, g].sum(1) for g in groups], axis=1)
        result[f'landmarks_{count}'] = projected(pooled, seed + 30)
    return result


def fit_score(train, y, ev, ey):
    a = np.c_[train, np.ones(len(train))]
    b = np.c_[ev, np.ones(len(ev))]
    w = a.T @ np.linalg.solve(a @ a.T + .1 * np.eye(len(a)), np.eye(4)[y])
    scores = b @ w
    pred = scores.argmax(1)
    return {'accuracy': float(np.mean(pred == ey)), 'predictions': pred.tolist(),
            'train_accuracy': float(np.mean((a @ w).argmax(1) == y)),
            'parameters': int(w.size), 'train_design_rank': int(np.linalg.matrix_rank(a))}


def run_sites(x, y, ex, ey):
    bridge = PortBridge.fit(x, 'pca')
    shadow = ShadowEngine(checkpoint(601))
    train = features(bridge, shadow, x)
    ev = features(bridge, shadow, ex)
    runs = []
    for seed in SEEDS:
        a, b = designs(train, seed), designs(ev, seed)
        models = {name: fit_score(a[name], y, b[name], ey) for name in a}
        # Counterfactual channel identity violation: no label access.
        shuffled = dict(ev)
        shuffled['kc'] = ev['kc'][:, np.random.default_rng(seed + 80).permutation(ev['kc'].shape[1])]
        shuffled_b = designs(shuffled, seed)
        for name in ('kc', 'multi_site', 'repeated_kc', 'landmarks_4', 'landmarks_16', 'landmarks_64'):
            models[name]['shuffled_kc_identity'] = fit_score(a[name], y, shuffled_b[name], ey)
        ablated = dict(ev)
        ablated['kc'] = np.zeros_like(ev['kc'])
        ablated_b = designs(ablated, seed)
        models['multi_site']['zero_kc_branch'] = fit_score(a['multi_site'], y, ablated_b['multi_site'], ey)
        runs.append({'seed': seed, 'models': models})
    return {'boundary': 'Read-only supervised site probes; fixed pooling is not learned tokens; no causal write-in. Multi-site raw/PN bypasses may explain gains. MBON uses prior trained checkpoint.',
            'width': WIDTH, 'ridge': .1, 'runs': runs}


def save_sites(out, x, y):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'sites.safetensors'
    if path.exists():
        raise FileExistsError(path)
    bridge = PortBridge.fit(x, 'pca')
    shadow = ShadowEngine(checkpoint(601))
    reps = features(bridge, shadow, x)
    tensors = {'mean': bridge.mean, 'projection': bridge.projection}
    details = {}
    for seed in SEEDS:
        for name, v in designs(reps, seed).items():
            a = np.c_[v, np.ones(len(v))]
            w = a.T @ np.linalg.solve(a @ a.T + .1 * np.eye(len(a)), np.eye(4)[y])
            key = f's{seed}_{name}'
            tensors[key] = np.ascontiguousarray(w)
            details[key] = {'train_accuracy': float(np.mean((a @ w).argmax(1) == y)),
                            'parameters': int(w.size), 'train_design_rank': int(np.linalg.matrix_rank(a))}
    save_file(tensors, str(path), metadata={'format': 'novigrad.site_probes', 'version': '1'})
    (out / 'sites.json').write_text(json.dumps(details, indent=2) + '\n')


def evaluate_sites(out, ex, ey):
    out = Path(out)
    with safe_open(str(out / 'sites.safetensors'), framework='numpy') as handle:
        if handle.metadata() != {'format': 'novigrad.site_probes', 'version': '1'}:
            raise ValueError('invalid site schema metadata')
    tensors = load_file(str(out / 'sites.safetensors'))
    details = json.loads((out / 'sites.json').read_text())
    names = ('embedding', 'pn', 'raw_kc', 'kc', 'mbon', 'multi_site', 'repeated_kc',
             'landmarks_4', 'landmarks_16', 'landmarks_64')
    expected = {f's{seed}_{name}' for seed in SEEDS for name in names}
    if set(tensors) != expected | {'mean', 'projection'} or set(details) != expected:
        raise ValueError('invalid site tensor inventory')
    if tensors['mean'].shape != (768,) or tensors['projection'].shape != (768, 128):
        raise ValueError('invalid site bridge shape')
    if any(tensors[k].shape != (WIDTH + 1, 4) for k in expected) or any(not np.isfinite(v).all() for v in tensors.values()):
        raise ValueError('invalid site weights')
    bridge = PortBridge(tensors['mean'], tensors['projection'], 'pca')
    reps = features(bridge, ShadowEngine(checkpoint(601)), ex)
    runs = []
    for seed in SEEDS:
        clean = designs(reps, seed)
        changed = dict(reps)
        changed['kc'] = reps['kc'][:, np.random.default_rng(seed + 80).permutation(reps['kc'].shape[1])]
        shuffled = designs(changed, seed)
        changed['kc'] = np.zeros_like(reps['kc'])
        ablated = designs(changed, seed)
        models = {}
        for name, v in clean.items():
            key = f's{seed}_{name}'
            def score(rows):
                pred = (np.c_[rows, np.ones(len(rows))] @ tensors[key]).argmax(1)
                return {'accuracy': float(np.mean(pred == ey)), 'predictions': pred.tolist()}
            models[name] = {**details[key], **score(v)}
            if name in ('kc', 'multi_site', 'repeated_kc', 'landmarks_4', 'landmarks_16', 'landmarks_64'):
                models[name]['shuffled_kc_identity'] = score(shuffled[name])
            if name == 'multi_site':
                models[name]['zero_kc_branch'] = score(ablated[name])
        runs.append({'seed': seed, 'models': models})
    return {'boundary': 'Saved read-only supervised probes, 32 features plus bias each; no learned tokens or causal write-in. Multi-site has raw/PN bypasses and previously trained MBON.',
            'width': WIDTH, 'ridge': .1, 'runs': runs}


def sinkhorn(logits, iterations=100):
    """Positive doubly stochastic mixing proxy; not a full mHC implementation."""
    z = np.asarray(logits, np.float64)
    if z.ndim != 2 or z.shape[0] != z.shape[1] or not np.isfinite(z).all():
        raise ValueError('finite square logits required')
    p = np.exp(z - z.max())
    for _ in range(iterations):
        p /= np.maximum(p.sum(1, keepdims=True), 1e-300)
        p /= np.maximum(p.sum(0, keepdims=True), 1e-300)
    return p


def mixing_sanity():
    rng = np.random.default_rng(1221)
    p = sinkhorn(rng.normal(size=(4, 4)))
    state = rng.normal(size=(4, 32))
    initial = np.linalg.norm(state)
    mixed = state.copy()
    for _ in range(100):
        mixed = p @ mixed
    permutation = np.eye(4)[[1, 2, 3, 0]]
    # Identity information can collapse despite stable signal magnitude.
    uniform = np.full((4, 4), .25)
    return {'boundary': 'Algebraic stress test only; not trained mHC, no semantic accuracy claim.',
            'row_sum_error': float(abs(p.sum(1) - 1).max()),
            'column_sum_error': float(abs(p.sum(0) - 1).max()),
            'norm_ratio_after_100': float(np.linalg.norm(mixed) / initial),
            'uniform_stream_rank': int(np.linalg.matrix_rank(uniform)),
            'identity_stream_rank': 4,
            'permutation_is_doubly_stochastic': bool(np.all(permutation.sum(0) == 1) and np.all(permutation.sum(1) == 1)),
            'counterexample': 'Uniform mixing is stable but rank 1; permutation mixing is stable but changes stream identity. Stability does not establish semantic correspondence.'}
