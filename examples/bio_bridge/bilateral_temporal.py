"""Typed site/time resampling of frozen simulated features; supervised, not native memory."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from safetensors.numpy import save_file as save_np, load_file as load_np
from safetensors.torch import save_file, load_file
from precise_bridge import PortBridge, dataset, ROOT
from inhibition_mechanism import ShadowEngine, checkpoint
from neural_resampler import NeuralResampler, save_resampler, load_resampler

OUT = ROOT / 'results/bilateral-bridge'
SEEDS = (1801, 1802, 1803)
MODES = ('learned', 'fixed', 'pooled_mlp')
STEPS = 300


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class SiteCodec:
    def __init__(self, tensors):
        self.tensors = tensors
        self.bridge = PortBridge(tensors['mean'], tensors['projection'], 'pca')
        self.shadow = ShadowEngine(checkpoint(601))

    @classmethod
    def fit(cls, x):
        bridge = PortBridge.fit(x, 'pca')
        rng = np.random.default_rng(1781)
        tensors = {'mean': bridge.mean, 'projection': bridge.projection}
        for name, size in [('pn', 319), ('kc', 5177), ('mbon', 96), ('embedding0', 768),
                           ('embedding1', 768), ('embedding2', 768)]:
            tensors[name] = (rng.choice([-1., 1.], size=(size, 32)) / np.sqrt(32)).astype(np.float32)
        return cls(tensors)

    def encode(self, x, source='circuit'):
        if source == 'embedding':
            return np.stack([unit(unit(x) @ self.tensors[f'embedding{i}']) for i in range(3)], axis=1).astype(np.float32)
        if source != 'circuit':
            raise ValueError('unknown feature source')
        pn = self.bridge.encode(x)
        kc = self.shadow.inhibit(self.shadow.raw_hidden(pn), 'native_topk_2', {})
        mbon = np.asarray(self.shadow.plastic_matrix @ kc.T).T
        return np.stack([unit(unit(v) @ self.tensors[k]) for k, v in [('pn', pn), ('kc', kc), ('mbon', mbon)]], axis=1).astype(np.float32)

    def save(self, path):
        save_np({k: np.ascontiguousarray(v) for k, v in self.tensors.items()}, str(path),
                metadata={'format': 'novigrad.site_codec', 'version': '1'})

    @classmethod
    def load(cls, path):
        tensors = load_np(str(path))
        expected = {'mean': (768,), 'projection': (768, 128), 'pn': (319, 32), 'kc': (5177, 32),
                    'mbon': (96, 32), **{f'embedding{i}': (768, 32) for i in range(3)}}
        if set(tensors) != set(expected) or any(tensors[k].shape != shape or not np.isfinite(tensors[k]).all() for k, shape in expected.items()):
            raise ValueError('invalid saved site codec')
        return cls(tensors)


def episodes(current, labels, past, past_labels, repeats, seed):
    """Past examples are independent draws from training only; current labels never condition draws."""
    rng = np.random.default_rng(seed)
    base = np.repeat(np.arange(len(current)), repeats)
    previous = rng.integers(len(past), size=(len(base), 2))
    frames = np.concatenate([past[previous], current[base, None]], axis=1)
    count = len(base)
    packet = {'features': torch.from_numpy(frames.reshape(count, 9, 32)),
              'valid_mask': torch.ones((count, 9), dtype=torch.bool),
              'site_ids': torch.tensor(np.tile([0, 1, 2], (count, 3)), dtype=torch.long),
              'timestamps': torch.tensor(np.tile(np.repeat([-2., -1., 0.], 3), (count, 1)), dtype=torch.float32),
              'observation_time': torch.zeros(count)}
    return packet, np.asarray(labels)[base], {'base_case_indices': base.tolist(),
             'past_indices': previous.tolist(), 'oldest_labels': np.asarray(past_labels)[previous[:, 0]].tolist()}


def select(packet, ix):
    return {k: v[ix] for k, v in packet.items()}


def logits(model, head, packet):
    return head(model(**packet).flatten(1))


def prediction(model, head, packet):
    with torch.no_grad():
        return logits(model, head, packet).argmax(1).numpy()


def score(model, head, packet, labels, identity):
    pred = prediction(model, head, packet)
    swapped = {k: v.clone() for k, v in packet.items()}
    swapped['timestamps'] = -2 - swapped['timestamps']
    swap_pred = prediction(model, head, swapped)
    zero = {k: v.clone() for k, v in packet.items()}
    zero['features'][:, 6:] = 0
    absent = {k: v.clone() for k, v in packet.items()}
    absent['valid_mask'][:, 6:] = False
    shuffled = {k: v.clone() for k, v in packet.items()}
    shuffled['features'] = shuffled['features'][torch.arange(len(labels) - 1, -1, -1)]
    metadata_only = {k: v.clone() for k, v in packet.items()}
    metadata_only['features'].zero_()
    accuracy = lambda p, y: float(np.mean(p == np.asarray(y)))
    return {'accuracy': accuracy(pred, labels), 'predictions': pred.tolist(),
            'time_swap_current_target_accuracy': accuracy(swap_pred, labels),
            'time_swap_oldest_target_accuracy': accuracy(swap_pred, identity['oldest_labels']),
            'time_swap_predictions': swap_pred.tolist(),
            'zero_current_accuracy': accuracy(prediction(model, head, zero), labels),
            'absent_current_accuracy': accuracy(prediction(model, head, absent), labels),
            'row_shuffle_accuracy': accuracy(prediction(model, head, shuffled), labels),
            'metadata_only_accuracy': accuracy(prediction(model, head, metadata_only), labels),
            'row_shuffle_permutation': list(range(len(labels) - 1, -1, -1))}


def train_temporal(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    d = dataset()
    x, y, _ = d['train']; v, vy, _ = d['validation']
    codec = SiteCodec.fit(x)
    codec.save(out / 'codec.safetensors')
    report = {'scope': 'Frozen simulated sites at externally assigned times; train32, reused validation12; offline supervised readout, not learned biological recurrence.',
              'steps': STEPS, 'batch_size': 32, 'learning_rate': .003, 'seeds': list(SEEDS), 'device': 'cpu',
              'fixed_mode_meaning': 'Only query vectors fixed; attention and metadata embeddings trained.', 'models': {}}
    for source in ('circuit', 'embedding'):
        train, val = codec.encode(x, source), codec.encode(v, source)
        tp, ty, ti = episodes(train, y, train, y, 12, 1791)
        vp, vlabels, vi = episodes(val, vy, train, y, 4, 1792)
        for mode in MODES:
            runs = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                model = NeuralResampler(mode=mode)
                torch.manual_seed(seed + 13)
                head = nn.Linear(128, 4)
                optimizer = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=.003)
                rng = np.random.default_rng(seed + 10)
                target = torch.tensor(ty, dtype=torch.long)
                started = time.perf_counter()
                for step in range(STEPS):
                    ix = rng.integers(len(ty), size=32)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(logits(model, head, select(tp, ix)), target[ix])
                    loss.backward()
                    nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], 1.)
                    optimizer.step()
                prefix = f'{source}-{mode}-s{seed}'
                model.eval(); head.eval()
                save_resampler(model, out / f'{prefix}.safetensors')
                save_file({k: t.detach().contiguous() for k, t in head.state_dict().items()}, str(out / f'{prefix}-head.safetensors'))
                restored, restored_head = load_pair(out, prefix)
                with torch.no_grad():
                    if not torch.equal(logits(model, head, vp), logits(restored, restored_head, vp)):
                        raise AssertionError('saved temporal logits differ')
                runs.append({'seed': seed, 'parameters': model.trainable_parameters + 516,
                             'seconds': time.perf_counter() - started, 'final_batch_loss': float(loss.detach()),
                             'train': score(model, head, tp, ty, ti),
                             'validation': score(model, head, vp, vlabels, vi), 'saved_logits_exact': True})
            report['models'][f'{source}/{mode}'] = runs
            print(source, mode, [r['validation']['accuracy'] for r in runs], flush=True)
    (out / 'development.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def load_pair(out, prefix):
    model = load_resampler(Path(out) / f'{prefix}.safetensors').eval()
    state = load_file(str(Path(out) / f'{prefix}-head.safetensors'))
    if set(state) != {'weight', 'bias'} or state['weight'].shape != (4, 128) or state['bias'].shape != (4,) or any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError('invalid head')
    head = nn.Linear(128, 4); head.load_state_dict(state); head.eval()
    return model, head


def evaluate_temporal(out, x, y):
    torch.set_num_threads(1)
    codec = SiteCodec.load(Path(out) / 'codec.safetensors')
    tx, ty, _ = dataset()['train']
    report = {}
    for source in ('circuit', 'embedding'):
        packet, labels, identity = episodes(codec.encode(x, source), y, codec.encode(tx, source), ty, 4, 1792)
        for mode in MODES:
            report[f'{source}/{mode}'] = [{'seed': seed, **score(*load_pair(out, f'{source}-{mode}-s{seed}'), packet, labels, identity)} for seed in SEEDS]
    return {'models': report, 'labels': labels.tolist(), 'identity': identity,
            'unit': 'four temporal contexts per underlying case; not independent replicates'}


if __name__ == '__main__':
    train_temporal(OUT / 'temporal')
