"""Foreground study of temporal embeddings and real frozen-LM soft-prefix interfaces."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from safetensors.numpy import load_file
from precise_bridge import ROOT, dataset
from inhibition_mechanism import checkpoint
from bilateral_temporal import SiteCodec, evaluate_temporal, load_pair, MODES, SEEDS
from encode_precise_holdout import validate_cases

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/bilateral-bridge'
VARIANTS = {'circuit_learned': ('circuit', 'learned'), 'circuit_fixed_queries': ('circuit', 'fixed'),
            'circuit_pooled_mlp': ('circuit', 'pooled_mlp'), 'embedding_learned': ('embedding', 'learned'),
            'label_oracle': ('oracle', 'learned')}
VARIANTS.update({name + '_long': pair for name, pair in list(VARIANTS.items()) if name != 'label_oracle'})


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def file_hashes(paths):
    return {str(p.resolve().relative_to(ROOT.resolve())): sha(p) for p in paths}


def check_files(records):
    for name, digest in records.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'frozen artifact changed: {name}')


def check_models(models):
    for record in models.values():
        for name, digest in record['files'].items():
            if sha(Path(record['path']) / name) != digest:
                raise ValueError(f'frozen model changed: {name}')


def required_inventory():
    result = [OUT / 'temporal/development.json', OUT / 'temporal/codec.safetensors', OUT / 'lm-baseline.json']
    result += [OUT / 'temporal' / f'{source}-{mode}-s{seed}{suffix}' for source in ('circuit', 'embedding')
               for mode in MODES for seed in SEEDS for suffix in ('.safetensors', '-head.safetensors')]
    result += [OUT / 'lm' / f'{name}{suffix}' for name in VARIANTS
               for suffix in ('.safetensors', '.safetensors.manifest.json', '-development.json')]
    return result


def packet(x, y, source, codec):
    from bilateral_lm import load_train
    if source == 'oracle':
        features = np.zeros((len(x), 3, 32), np.float32)
        features[:, :, :4] = np.eye(4, dtype=np.float32)[y, None]
    else:
        features = codec.encode(x, source)
    return load_train(features, np.ones((len(x), 3), bool), np.tile(np.arange(3), (len(x), 1)),
                      np.zeros((len(x), 3), np.float32), np.asarray(y, np.int64))


def mean_prefix_control(path, source, codec, y, runtime, indices):
    """Mean over old TRAIN prefixes only; constant syntax/content control, not test fitting."""
    import bilateral_lm as lm
    tx, ty, _ = dataset()['train']
    adapter, _ = lm.load_adapter(path, runtime.device)
    adapter.eval()
    with torch.inference_mode():
        mean = adapter(lm._to(packet(tx, ty, source, codec), runtime.device)).mean(0, keepdim=True)
    return lm._condition_report(runtime, mean.expand(len(y), -1, -1), y, set(indices),
                                 {}, np.zeros(len(y), dtype=int), True)


def train_lm():
    import bilateral_lm as lm
    torch.set_num_threads(1)
    codec = SiteCodec.load(OUT / 'temporal/codec.safetensors')
    x, y, _ = dataset()['train']; v, vy, _ = dataset()['validation']
    runtime = lm.Runtime.load()
    if not (OUT / 'lm-baseline.json').exists():
        write(OUT / 'lm-baseline.json', lm.evaluate_no_input(packet(v, vy, 'circuit', codec), vy,
               generation_indices=range(len(v)), runtime=runtime))
    for name, (source, mode) in VARIANTS.items():
        path = OUT / 'lm' / f'{name}.safetensors'
        report_path = OUT / 'lm' / f'{name}-development.json'
        if path.exists() and report_path.exists():
            print('already measured', name, flush=True)
            continue
        if not path.exists():
            print('training', name, flush=True)
            lm.train_adapter(path, mode, packet(x, y, source, codec), y,
                             steps=480 if name.endswith('_long') else 120, runtime=runtime)
        result = lm.evaluate(path, packet(v, vy, source, codec), vy,
                             generation_indices=range(len(v)), runtime=runtime)
        write(report_path, result)
        print(name, {k: {'rank': result[k]['rank_accuracy'], 'generated_correct': result[k]['correct_count'],
                        'generated_n': result[k]['generated_count']} for k in ('actual', 'zero_prefix', 'row_shuffled_prefix')}, flush=True)


def freeze():
    if (OUT / 'protocol-lock.json').exists():
        raise FileExistsError('already frozen')
    cases = HERE / 'bilateral_holdout.json'
    validate_cases(json.loads(cases.read_text()))
    required = required_inventory()
    if not all(p.exists() for p in required):
        raise ValueError('development inventory incomplete')
    import bilateral_lm as lm
    for name in VARIANTS:
        lm.load_adapter(OUT / 'lm' / f'{name}.safetensors')
    for source in ('circuit', 'embedding'):
        for mode in MODES:
            for seed in SEEDS:
                load_pair(OUT / 'temporal', f'{source}-{mode}-s{seed}')
    v, vy, _ = dataset()['validation']
    evaluate_temporal(OUT / 'temporal', v, vy)
    paths = [*HERE.glob('bilateral*.py'), HERE / 'neural_resampler.py', *HERE.glob('test_bilateral*.py'),
             HERE / 'test_neural_resampler.py', cases, checkpoint(601)]
    paths += [HERE / name for name in ('precise_bridge.py', 'inhibition_mechanism.py', 'thought_embedding.py',
                                     'delayed_credit.py', 'encode_precise_holdout.py')]
    paths += [HERE.parent / 'function_bridge/protocol.py']
    paths += [ROOT / 'results/precise-bridge/native-checkpoints/pca-seed-701.safetensors']
    paths += [ROOT / 'results/gemma-bridge' / name for name in ('dataset.json', 'encoder.json', 'embeddings.safetensors')]
    paths += [p for p in OUT.rglob('*') if p.is_file()]
    encoder = json.loads((ROOT / 'results/gemma-bridge/encoder.json').read_text())
    models = {}
    for key, model in [('embedding', Path(encoder['model'])), ('function', lm.MODEL_PATH)]:
        models[key] = {'path': str(model), 'files': {str(p.relative_to(model)): sha(p)
            for p in sorted(model.rglob('*')) if p.is_file() and p.suffix in ('.safetensors', '.json', '.txt', '.model')}}
    if {k: v for k, v in models['embedding']['files'].items() if k.endswith('.safetensors')} != encoder['weight_sha256']:
        raise ValueError('embedding revision changed from training features')
    write(OUT / 'protocol-lock.json', {'version': 1, 'variants': VARIANTS, 'frozen': file_hashes(paths), 'models': models,
          'required_inventory': [str(p.relative_to(ROOT)) for p in required],
          'final': 'new authored64texts/32bilingualpairs; all variants, no selection; actual generated call primary, four-candidate ranking secondary',
          'boundaries': ['temporal task selects current frame; not recurrent memory', 'only adapter trained, actual BF16 FunctionGemma frozen',
                         'oracle leaks class by design and is only an upper control', 'local procedural one-shot, not cryptographic blinding'],
          'versions': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'transformers', 'sentence-transformers', 'safetensors')}})
    print('frozen', sha(OUT / 'protocol-lock.json'), flush=True)


def checked_lock():
    record = json.loads((OUT / 'protocol-lock.json').read_text())
    if record.get('version') != 1 or record.get('variants') != {k: list(v) for k, v in VARIANTS.items()}:
        raise ValueError('unsupported lock schema or variants')
    expected = {str(p.relative_to(ROOT)) for p in required_inventory()}
    if set(record.get('required_inventory', [])) != expected or not expected <= set(record['frozen']):
        raise ValueError('incomplete frozen inventory')
    check_files(record['frozen'])
    check_models(record['models'])
    return record


def encode():
    checked_lock()
    subprocess.run([sys.executable, str(HERE / 'encode_precise_holdout.py'), '--cases', str(HERE / 'bilateral_holdout.json'),
                    '--out', str(OUT / 'holdout')], check=True)


def final_data():
    lock = checked_lock()
    case_path = HERE / 'bilateral_holdout.json'
    cases = validate_cases(json.loads(case_path.read_text()))
    feature = OUT / 'holdout/holdout-embeddings.safetensors'
    provenance = json.loads((OUT / 'holdout/provenance.json').read_text())
    if provenance['cases_sha256'] != sha(case_path) or provenance['embedding_sha256'] != sha(feature) or provenance['row_ids'] != [r['id'] for r in cases]:
        raise ValueError('case/feature/order mismatch')
    if provenance['model_file_sha256'] != lock['models']['embedding']['files'] or provenance['prompt_name'] != 'Classification':
        raise ValueError('wrong encoder revision or prompt')
    tensors = load_file(str(feature))
    if set(tensors) != {'embeddings'}:
        raise ValueError('unexpected feature schema')
    x = tensors['embeddings']
    if x.shape != (64, 768) or x.dtype != np.float32 or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1., atol=2e-5):
        raise ValueError('invalid final features')
    y = np.array([('water', 'food', 'warmth', 'rest').index(r['class']) for r in cases])
    return cases, x, y


def final():
    import bilateral_lm as lm
    if (OUT / 'evaluation-lock.json').exists() or (OUT / 'final-started.json').exists():
        raise FileExistsError('final already begun; replay requires saved outputs')
    cases, x, y = final_data()
    write(OUT / 'final-started.json', {'protocol_sha256': sha(OUT / 'protocol-lock.json')})
    write(OUT / 'temporal-final.json', evaluate_temporal(OUT / 'temporal', x, y))
    codec = SiteCodec.load(OUT / 'temporal/codec.safetensors')
    lm.Runtime._cache.clear()
    runtime = lm.Runtime.load()
    base_identity = runtime.memory_fingerprint()
    for name in VARIANTS:
        manifest = json.loads((OUT / 'lm' / f'{name}.safetensors.manifest.json').read_text())
        if name.endswith('_long') and manifest['base_memory_sha256_before'] != base_identity:
            raise ValueError('loaded base memory differs from training base')
    for name, (source, _) in VARIANTS.items():
        result = lm.evaluate(OUT / 'lm' / f'{name}.safetensors', packet(x, y, source, codec), y, runtime=runtime)
        result['response_token_counts'] = [len(ids) for ids in runtime.response_ids]
        result['candidate_ranking_boundary'] = 'Sum of full-response token log-likelihoods; lengths differ. Actual parsed generation is primary, not closed-set ranking.'
        result['training_mean_prefix'] = mean_prefix_control(OUT / 'lm' / f'{name}.safetensors', source, codec,
                                                             y, runtime, range(len(y)))
        result['case_ids'] = [r['id'] for r in cases]
        write(OUT / f'{name}-final.json', result)
        print('final', name, result['actual']['rank_accuracy'], result['actual']['correct_count'], flush=True)
    from bilateral_closed_loop import run_loop, DEFAULT_INDICES
    loops = {name: run_loop(OUT / 'lm' / f'{name}.safetensors', OUT / 'temporal/codec.safetensors',
                            x, y, DEFAULT_INDICES, runtime)
             for name in ('circuit_fixed_queries_long', 'circuit_learned_long')}
    write(OUT / 'closed-loop-final.json', loops)
    from bilateral_policy_swap import run_policy_swap
    swapped = {name: run_policy_swap(OUT / 'lm' / f'{name}.safetensors', OUT / 'temporal/codec.safetensors',
                                     x, y, DEFAULT_INDICES, runtime)
               for name in ('circuit_fixed_queries_long', 'circuit_learned_long')}
    write(OUT / 'policy-swap-final.json', swapped)
    if runtime.memory_fingerprint() != base_identity:
        raise AssertionError('base memory changed during evaluation')
    files = [OUT / 'protocol-lock.json', OUT / 'final-started.json', OUT / 'temporal-final.json']
    files += [OUT / f'{name}-final.json' for name in VARIANTS] + [OUT / 'closed-loop-final.json', OUT / 'policy-swap-final.json']
    files += [p for p in (OUT / 'holdout').iterdir() if p.is_file()]
    write(OUT / 'evaluation-lock.json', {'hashes': file_hashes(files)})


def verify():
    checked_lock()
    check_files(json.loads((OUT / 'evaluation-lock.json').read_text())['hashes'])
    _, x, y = final_data()
    expected = json.loads((OUT / 'temporal-final.json').read_text())
    if evaluate_temporal(OUT / 'temporal', x, y) != expected:
        raise AssertionError('temporal saved replay differs')
    import bilateral_lm as lm
    from bilateral_closed_loop import run_loop, DEFAULT_INDICES
    codec = SiteCodec.load(OUT / 'temporal/codec.safetensors')
    lm.Runtime._cache.clear()
    runtime = lm.Runtime.load()
    checks = {}
    for name, (source, _) in VARIANTS.items():
        expected = json.loads((OUT / f'{name}-final.json').read_text())
        replayed = lm.evaluate(OUT / 'lm' / f'{name}.safetensors', packet(x, y, source, codec), y,
                               generation_indices=DEFAULT_INDICES, runtime=runtime)
        replayed['training_mean_prefix'] = mean_prefix_control(OUT / 'lm' / f'{name}.safetensors', source, codec,
                                                               y, runtime, DEFAULT_INDICES)
        checks[name] = compare_lm(expected, replayed)
        print('replayed', name, checks[name], flush=True)
    loops = json.loads((OUT / 'closed-loop-final.json').read_text())
    for name, expected_loop in loops.items():
        if run_loop(OUT / 'lm' / f'{name}.safetensors', OUT / 'temporal/codec.safetensors', x, y,
                    DEFAULT_INDICES, runtime) != expected_loop:
            raise AssertionError('closed-loop saved replay changed')
    from bilateral_policy_swap import run_policy_swap
    for name, expected_loop in json.loads((OUT / 'policy-swap-final.json').read_text()).items():
        if run_policy_swap(OUT / 'lm' / f'{name}.safetensors', OUT / 'temporal/codec.safetensors', x, y,
                           DEFAULT_INDICES, runtime) != expected_loop:
            raise AssertionError('policy-swap saved replay changed')
    write(OUT / 'verification.json', {'all_frozen_hashes_pass': True, 'temporal_full_saved_replay_exact': True,
          'lm_replay': checks, 'closed_loop_full_replay_exact': True, 'policy_swap_full_replay_exact': True,
          'generation_replay_indices_per_condition': list(DEFAULT_INDICES),
          'boundary': 'All candidate likelihood rows and ranks replayed; free generation replayed on eight predeclared indices per condition. Timings excluded; no bitwise training claim.'})


def compare_lm(expected, replayed):
    maximum = 0.
    generated = 0
    for condition in ('actual', 'zero_prefix', 'row_shuffled_prefix', 'training_mean_prefix'):
        a, b = expected[condition]['rows'], replayed[condition]['rows']
        if len(a) != len(b):
            raise AssertionError('different replay row count')
        for old, new in zip(a, b):
            for key in ('index', 'target', 'rank_prediction'):
                if old[key] != new[key]:
                    raise AssertionError(f'replay changed {condition}/{key}')
            delta = float(np.max(abs(np.asarray(old['candidate_loglikelihoods']) - new['candidate_loglikelihoods'])))
            maximum = max(maximum, delta)
            if delta > 1e-5:
                raise AssertionError(f'same-batch likelihood replay difference: {delta}')
            if 'free_output' in new:
                for key in ('free_output', 'parsed_goal', 'valid', 'correct'):
                    if old[key] != new[key]:
                        raise AssertionError(f'free generation replay changed {key}')
                generated += 1
    if expected['shuffle_order'] != replayed['shuffle_order']:
        raise AssertionError('shuffle protocol changed')
    return {'max_absolute_candidate_loglikelihood_error': maximum, 'likelihood_tolerance': 1e-5,
            'all_rank_predictions_exact': True, 'checked_free_generation_rows': generated,
            'checked_free_generation_exact': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train_lm', 'freeze', 'encode', 'final', 'verify'))
    globals()[parser.parse_args().stage]()
