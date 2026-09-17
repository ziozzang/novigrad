"""Post-hoc action-feedback diagnosis; explicit host inference, frozen LM bridge."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from novigrad import Engine
import bilateral_lm as lm
import bilateral_study as previous
from bilateral_closed_loop import prototypes_from_old_train, DEFAULT_INDICES
from bilateral_temporal import SiteCodec
from bilateral_policy_swap import BASE_POLICY, PCA_POLICY, verify_policy_slot
from feedback_environment import HiddenGoalEnvironment, FeedbackConfig, posterior

ROOT = previous.ROOT
OUT = ROOT / 'results/causal-feedback'
MODELS = ('circuit_pooled_mlp_long', 'circuit_learned_long', 'circuit_fixed_queries_long')
MODES = ('original', 'self_writeback', 'belief_half', 'belief_full', 'static_prior_prefix',
         'flipped_feedback', 'permuted_prototypes', 'uniform_prefix', 'analytic_fixed_model', 'prior_ranked_no_repeat')
POLICIES = {'native601': BASE_POLICY, 'native701': PCA_POLICY}
SCENARIOS = {'immediate': FeedbackConfig(), 'delayed': FeedbackConfig(delay=1),
             'noisy': FeedbackConfig(reward_flip_probability=.25)}


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return previous.sha(path)


def normalized_prior(loglikelihoods, lengths):
    values = np.asarray(loglikelihoods, np.float64) / np.asarray(lengths, np.float64)
    values -= values.max(axis=-1, keepdims=True)
    masses = np.exp(values)
    return masses / masses.sum(axis=-1, keepdims=True)


def next_prefix(initial, prototypes, belief, mode, last_action):
    """The host creates semantic stimulation. The LM does not compute the posterior."""
    if mode == 'original':
        return initial
    if mode == 'self_writeback':
        mixed = prototypes[last_action]
        gain = 1.
    else:
        weights = np.asarray(belief, np.float64)
        if mode == 'uniform_prefix':
            weights = np.full(4, .25)
        if mode == 'permuted_prototypes':
            weights = np.roll(weights, 1)
        mixed = torch.einsum('c,ckh->kh', torch.tensor(weights, dtype=prototypes.dtype,
                                                     device=prototypes.device), prototypes)
        gain = .5 if mode == 'belief_half' else 1.
    combined = (1 - gain) * initial + gain * mixed
    norm = combined.norm()
    if not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError('degenerate stimulation prefix')
    return combined * (initial.norm() / norm)


def prefix_key(prefix):
    return hashlib.sha256(prefix.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare():
    if OUT.exists():
        raise FileExistsError('use a new output directory for another protocol')
    previous.checked_lock()
    cases, x, y = previous.final_data()
    OUT.mkdir(parents=True)
    codec = SiteCodec.load(previous.OUT / 'temporal/codec.safetensors')
    prototypes = prototypes_from_old_train()
    maps = {}
    verify_policy_slot()
    for name, path in POLICIES.items():
        probabilities = np.asarray(Engine.load(path).infer_batch(codec.bridge.encode(prototypes).tolist()))
        maps[name] = {'actions': probabilities.argmax(1).tolist(), 'probabilities': probabilities.tolist(),
                      'checkpoint_sha256': sha(path)}
    tensors = {}
    for name in MODELS:
        adapter, _ = lm.load_adapter(previous.OUT / 'lm' / f'{name}.safetensors', 'cpu')
        adapter.eval()
        # Prefix preparation uses MPS exactly as the prior study; CPU copies are archival only.
        adapter = adapter.to('mps')
        with torch.inference_mode():
            tensors[name + '/initial'] = adapter(lm._to(previous.packet(x, y, 'circuit', codec), 'mps')).cpu().contiguous()
            tensors[name + '/prototypes'] = adapter(lm._to(previous.packet(prototypes, np.arange(4), 'circuit', codec), 'mps')).cpu().contiguous()
        old = json.loads((previous.OUT / f'{name}-final.json').read_text())
        lengths = old['response_token_counts']
        if isinstance(lengths, dict):
            lengths = [lengths[k] for k in lm.GOALS]
        prior = normalized_prior([r['candidate_loglikelihoods'] for r in old['actual']['rows']], lengths)
        tensors[name + '/prior'] = torch.tensor(prior, dtype=torch.float64)
    save_file(tensors, str(OUT / 'inputs.safetensors'))
    write(OUT / 'cases.json', cases)
    files = [Path(__file__), Path(__file__).with_name('feedback_environment.py'),
             Path(__file__).with_name('test_causal_feedback.py'), Path(__file__).with_name('test_feedback_environment.py'),
             OUT / 'inputs.safetensors', OUT / 'cases.json', previous.OUT / 'protocol-lock.json',
             previous.OUT / 'evaluation-lock.json', previous.OUT / 'verification.json', *POLICIES.values()]
    files += [Path(__file__).with_name(name) for name in ('bilateral_lm.py', 'bilateral_study.py', 'bilateral_temporal.py', 'bilateral_closed_loop.py', 'bilateral_policy_swap.py')]
    files += [previous.OUT / f'{name}-final.json' for name in MODELS]
    write(OUT / 'protocol-lock.json', {
        'version': 1, 'models': list(MODELS), 'modes': list(MODES), 'maps': maps,
        'scenarios': {k: vars(v) for k, v in SCENARIOS.items()},
        'frozen': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'prior': 'softmax(sum response loglikelihood / response token count), temperature1; not calibrated',
        'prior_study_chain': {name: sha(previous.OUT / name) for name in ('protocol-lock.json', 'evaluation-lock.json', 'verification.json')},
        'runtime_versions': json.loads((previous.OUT / 'protocol-lock.json').read_text())['versions'],
        'belief_reliability': .9, 'max_decisions': 4, 'action_cost': .05,
        'base_generation_replay_cases': list(DEFAULT_INDICES),
        'scope': 'Reused64 authored cases, post-hoc mechanistic diagnosis; no training or fresh generalization claim',
        'factorial': 'All modes on immediate feedback and both native executors; belief_full and analytic_fixed_model additionally delayed/noisy',
        'boundaries': ['Host computes Bayesian belief and labeled prototype mixtures; LM learns nothing',
                       'Codec601 fixed; only native command-to-executed-action map varies',
                       'Native maps measured on four old-train prototypes and cached, not online dynamics',
                       'Real success terminates even when scalar reward is flipped; termination is informative',
                       'Prefix interventions are norm-matched; no gain selection',
                       'Common host reliability.9 is deliberately misspecified for noise.25; not an optimal Bayesian policy',
                       'Labeled old-train prototypes and candidate functions provide supervised host calibration',
                       'Four-choice deterministic search admits a no-feedback no-repeat solution']})
    print('prepared', sha(OUT / 'protocol-lock.json'), maps, flush=True)


def checked():
    previous.checked_lock()
    lock = json.loads((OUT / 'protocol-lock.json').read_text())
    previous.check_files(lock['frozen'])
    tensors = load_file(str(OUT / 'inputs.safetensors'))
    expected = {name + '/' + key: (shape, dtype) for name in MODELS for key, shape, dtype in (('initial', (64, 4, 640), torch.float32), ('prototypes', (4, 4, 640), torch.float32), ('prior', (64, 4), torch.float64))}
    if set(tensors) != set(expected):
        raise ValueError('input tensor inventory changed')
    for key, (shape, dtype) in expected.items():
        value = tensors[key]
        if value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError('invalid input tensor ' + key)
        if key.endswith('/prior') and (torch.any(value <= 0) or not torch.allclose(value.sum(1), torch.ones(64, dtype=dtype))):
            raise ValueError('invalid prior masses')
    cases = json.loads((OUT / 'cases.json').read_text())
    old_cases = json.loads((Path(__file__).with_name('bilateral_holdout.json')).read_text())
    if cases != old_cases or len(cases) != 64:
        raise ValueError('case row identity mismatch')
    return lock


def rollout(initial, prototypes, prior, target, mapping, mode, config, case_index, generate):
    """Target is passed only to physical environment and evaluator, never the controller."""
    env = HiddenGoalEnvironment([int(target)], seed=2901 + case_index, config=config)
    env.reset()
    history, trace = [], []
    belief = np.asarray(prior, np.float64).copy()
    last_action = None
    order = np.argsort(-belief, kind='stable')
    first_wrong = True
    for step in range(4):
        key, raw, norm = None, None, None
        if mode == 'analytic_fixed_model':
            action = int(belief.argmax())
        elif mode == 'prior_ranked_no_repeat':
            action = int(order[step])
        else:
            prefix = initial if step == 0 else next_prefix(initial, prototypes, belief, mode, last_action)
            key = prefix_key(prefix)
            norm = float(prefix.float().square().mean().sqrt().cpu())
            raw = generate(prefix, key, case_index)
            goal = lm.parse_goal(raw)
            action = lm.GOALS.index(goal) if goal is not None else None
        if action is None:
            trace.append({'step': step + 1, 'prefix_key': key, 'prefix_rms': norm, 'raw': raw,
                          'invalid': True, 'belief': belief.tolist()})
            break
        executed = int(mapping[action])
        observation = env.step(executed)
        actual_success = executed == int(target)
        if step == 0:
            first_wrong = not actual_success
        delivered = dict(observation)
        if mode == 'flipped_feedback' and delivered['reward'] is not None:
            delivered['reward'] = 1 - (delivered['reward'] + config.action_cost) - config.action_cost
        if mode != 'static_prior_prefix':
            history.append(delivered)
        updated = posterior(prior, history, reliability=.9, action_cost=config.action_cost)
        trace.append({'step': step + 1, 'prefix_key': key, 'prefix_rms': norm, 'raw': raw,
                      'invalid': False, 'command': action, 'executed_action': executed,
                      'belief': belief.tolist(), 'next_belief': updated.tolist(),
                      'observation': observation, 'delivered_observation': delivered,
                      'physical_success': actual_success})
        belief, last_action = updated, action
        if observation['done']:
            break
    success = any(r.get('physical_success', False) for r in trace)
    actions = [r['executed_action'] for r in trace if not r['invalid']]
    return {'case_index': case_index, 'target': int(target), 'trace': trace, 'success': success,
            'decisions': len(trace), 'physical_actions': len(actions),
            'utility': int(success) - .05 * len(actions),
            'first_wrong': first_wrong, 'rescued': first_wrong and success,
            'invalid': any(r['invalid'] for r in trace),
            'repeated_executed_actions': len(actions) - len(set(actions)),
            'success_by_step': [any(r.get('physical_success', False) and r['step'] <= k for r in trace) for k in range(1, 5)]}


def summarize(cases):
    n = len(cases)
    return {'n': n, 'successes': sum(c['success'] for c in cases),
            'success_rate': float(np.mean([c['success'] for c in cases])),
            'success_by_step': np.mean([c['success_by_step'] for c in cases], axis=0).tolist(),
            'mean_decisions': float(np.mean([c['decisions'] for c in cases])),
            'mean_utility': float(np.mean([c['utility'] for c in cases])),
            'initial_failures': sum(c['first_wrong'] for c in cases),
            'rescues': sum(c['rescued'] for c in cases), 'invalid_episodes': sum(c['invalid'] for c in cases),
            'repeated_executed_actions': sum(c['repeated_executed_actions'] for c in cases)}


def execute(lock, tensors, generate):
    cases = json.loads((OUT / 'cases.json').read_text())
    results = {}
    for name in MODELS:
        initial, prototypes = (tensors[name + '/' + k].to('mps') for k in ('initial', 'prototypes'))
        priors = tensors[name + '/prior'].numpy()
        for policy, record in lock['maps'].items():
            for scenario, config in SCENARIOS.items():
                modes = MODES if scenario == 'immediate' else ('belief_full', 'analytic_fixed_model')
                for mode in modes:
                    key = '/'.join((name, policy, scenario, mode))
                    rows = [rollout(initial[i], prototypes, priors[i], lm.GOALS.index(case['class']),
                                    record['actions'], mode, config, i,
                                    lambda prefix, digest, case_index: generate(name, prefix, digest, case_index))
                            for i, case in enumerate(cases)]
                    results[key] = {'metrics': summarize(rows), 'cases': rows}
                    print(key, results[key]['metrics']['successes'], flush=True)
    return results


def run():
    lock = checked()
    if (OUT / 'started.json').exists():
        raise FileExistsError('refusing repeated run; use verify')
    write(OUT / 'started.json', {'protocol_sha256': sha(OUT / 'protocol-lock.json')})
    runtime = lm.Runtime.load()
    before = runtime.memory_fingerprint()
    cache, replay_keys = {name: {} for name in MODELS}, {name: set() for name in MODELS}
    def generate(name, prefix, key, case_index):
        if key not in cache[name]:
            cache[name][key] = {'raw': runtime.generate(prefix), 'prefix': prefix.detach().cpu()}
        if case_index in DEFAULT_INDICES:
            replay_keys[name].add(key)
        return cache[name][key]['raw']
    results = execute(lock, load_file(str(OUT / 'inputs.safetensors')), generate)
    after = runtime.memory_fingerprint()
    if after != before:
        raise AssertionError('frozen base changed')
    payload = {name + '/' + key: value['prefix'].contiguous() for name, rows in cache.items() for key, value in rows.items()}
    save_file(payload, str(OUT / 'generation-prefixes.safetensors'))
    write(OUT / 'generation-cache.json', {name: {key: value['raw'] for key, value in rows.items()} for name, rows in cache.items()})
    write(OUT / 'results.json', results)
    write(OUT / 'run-manifest.json', {'base_memory_before_sha256': before, 'base_memory_after_sha256': after,
          'generation': {'do_sample': False, 'max_new_tokens': 48, 'dtype': 'BF16 base/FP32 prefix', 'device': 'mps'},
          'actual_unique_generations': {name: len(rows) for name, rows in cache.items()},
          'replay_keys': {name: sorted(keys) for name, keys in replay_keys.items()},
          'boundary': 'Caching identical prefixes is exact deterministic reuse, not independent trials'})
    files = ['protocol-lock.json', 'started.json', 'results.json', 'generation-cache.json',
             'generation-prefixes.safetensors', 'run-manifest.json']
    write(OUT / 'evaluation-lock.json', {name: sha(OUT / name) for name in files})


def verify():
    lock = checked()
    for name, digest in json.loads((OUT / 'evaluation-lock.json').read_text()).items():
        if sha(OUT / name) != digest:
            raise ValueError('changed evaluation artifact ' + name)
    cache = json.loads((OUT / 'generation-cache.json').read_text())
    original = json.loads((OUT / 'results.json').read_text())
    replayed = execute(lock, load_file(str(OUT / 'inputs.safetensors')),
                       lambda name, prefix, key, case_index: cache[name][key])
    if replayed != original:
        raise AssertionError('complete environment/host/prefix replay differs')
    runtime = lm.Runtime.load()
    manifest = json.loads((OUT / 'run-manifest.json').read_text())
    if runtime.memory_fingerprint() != manifest['base_memory_before_sha256']:
        raise AssertionError('base memory identity differs')
    prefixes = load_file(str(OUT / 'generation-prefixes.safetensors'))
    counts = {}
    for name, keys in manifest['replay_keys'].items():
        for key in keys:
            prefix = prefixes[name + '/' + key]
            if prefix_key(prefix) != key or runtime.generate(prefix.to('mps')) != cache[name][key]:
                raise AssertionError('actual generation replay differs')
        counts[name] = len(keys)
    write(OUT / 'verification.json', {'full_environment_host_prefix_replay_exact': True,
          'selected_actual_generation_exact': True, 'unique_actual_generation_replayed': counts,
          'case_indices': list(DEFAULT_INDICES), 'all_hashes_pass': True,
          'boundary': 'Full dynamics replay uses saved LM outputs; actual LM rerun covers all prefixes encountered by eight prespecified cases across every condition, not all64 cases'})
    print('verified', counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', 'verify'))
    globals()[parser.parse_args().stage]()
