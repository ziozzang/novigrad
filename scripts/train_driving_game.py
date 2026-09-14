#!/usr/bin/env python3
"""Closed-loop HighwayEnv reward learning through the existing Novigrad HTTP API.

The policy sees kinematics and receives scalar environment reward. There are no
expert actions. A fixed, state-independent baseline reduces return variance.
"""
import argparse
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.parse

os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import numpy as np
from PIL import Image
from driving_environment import make_env, encode_observation, CONFIG, ACTION_NAMES

ROOT = Path(__file__).resolve().parents[1]
GAMMA = .97
EPISODES = 128
BATCH_EPISODES = 4
VALIDATION_SEEDS = list(range(10000, 10008))
TEST_SEEDS = list(range(20000, 20020))


class Client:
    def __init__(self, origin):
        url = urllib.parse.urlsplit(origin)
        self.connection = http.client.HTTPConnection(url.hostname, url.port, timeout=30)
        metadata = self.request('GET', '/v1/models/policy')
        self.ids = metadata['input_ids']

    def request(self, method, path, body=None):
        self.connection.request(method, path, None if body is None else json.dumps(body), {'Content-Type': 'application/json'})
        response = self.connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {payload[:300]!r}')
        return json.loads(payload)

    def infer(self, rates):
        result = self.request('POST', '/v1/models/policy/infer', {'inputs': [rates.tolist()], 'input_ids': self.ids})
        p = np.asarray(result['probabilities'][0], dtype=np.float64)
        if p.shape != (5,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-6):
            raise ValueError('invalid five-action distribution')
        return p / p.sum()

    def learn(self, samples):
        if not 0 < len(samples) <= 256:
            raise ValueError('one on-policy batch must contain1..256 transitions')
        return self.request('POST', '/v1/models/policy/learn', {'samples': samples, 'input_ids': self.ids})


@contextlib.contextmanager
def server(model, directory):
    directory.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([str(ROOT/'target/release/novi_api'), '--model', f'policy={model}',
                                '--bind', '127.0.0.1:0', '--allow-training', '--checkpoint-dir', str(directory)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    client = None
    try:
        startup = process.stdout.readline().strip()
        if not startup.startswith('listening='):
            raise RuntimeError(f'API failed to start: {startup}')
        client = Client(startup.split('=', 1)[1])
        yield client
    finally:
        if client is not None:
            client.connection.close()
        process.terminate()
        process.wait(timeout=10)


def episode(client, seed, rng=None, policy='learned', render=False):
    env = make_env(seed, render_mode='rgb_array' if render else None)
    frames, trace, rewards, speeds = [], [], [], []
    try:
        observation, _ = env.reset(seed=seed)
        start_x = float(env.unwrapped.vehicle.position[0])
        for _ in range(40):
            rates = encode_observation(observation, len(client.ids))
            if policy == 'random':
                action = int(rng.integers(5))
            elif policy in ('idle', 'slow'):
                action = 1 if policy == 'idle' else 4
            else:
                probabilities = client.infer(rates)
                action = int(rng.choice(5, p=probabilities)) if rng is not None else int(probabilities.argmax())
            observation, reward, terminated, truncated, _ = env.step(action)
            trace.append((rates, action))
            rewards.append(float(reward))
            speeds.append(float(env.unwrapped.vehicle.speed))
            if render:
                frames.append(Image.fromarray(env.render()))
            if terminated or truncated:
                break
        result = {'seed': seed, 'return': float(sum(rewards)), 'steps': len(rewards),
                  'collision': bool(env.unwrapped.vehicle.crashed),
                  'distance_m': float(env.unwrapped.vehicle.position[0]) - start_x,
                  'mean_speed_m_s': float(np.mean(speeds)),
                  'action_counts': np.bincount([a for _, a in trace], minlength=5).tolist()}
        return result, trace, rewards, frames
    finally:
        env.close()


def aggregate(rows):
    return {'episodes': len(rows), 'mean_return': float(np.mean([r['return'] for r in rows])),
            'collision_rate': float(np.mean([r['collision'] for r in rows])),
            'mean_distance_m': float(np.mean([r['distance_m'] for r in rows])),
            'mean_speed_m_s': float(np.mean([r['mean_speed_m_s'] for r in rows])),
            'mean_steps': float(np.mean([r['steps'] for r in rows]))}


def evaluate(client, seeds, policy='learned'):
    rows = [episode(client, seed, np.random.default_rng(seed + 70000) if policy == 'random' else None, policy)[0] for seed in seeds]
    return {'aggregate': aggregate(rows), 'episodes': rows}


def advantages(rewards, baseline):
    returns = np.zeros(40, dtype=np.float64)
    value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        value = rewards[t] + GAMMA * value
        returns[t] = value
    # Fixed scale, not a per-episode centering that discards survival differences.
    return np.clip((returns[:len(rewards)] - baseline[:len(rewards)]) / 10, -1, 1), returns


def train(initial, out, feedback='reward'):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    feedback_rng = np.random.default_rng(9001)
    history, validation = [], []
    baseline = np.zeros(40)
    started = time.perf_counter()
    with server(initial, out/'live') as client:
        initial_validation = evaluate(client, VALIDATION_SEEDS)
        best = initial_validation['aggregate']['mean_return']
        shutil.copy2(initial, out/'best.safetensors')
        best_episode = 0
        for batch_start in range(0, EPISODES, BATCH_EPISODES):
            samples, batch_returns = [], []
            for index in range(batch_start, batch_start + BATCH_EPISODES):
                row, trace, rewards, _ = episode(client, 1000 + index, rng)
                history.append(row)
                advantage, returns = advantages(rewards, baseline)
                batch_returns.append(returns)
                samples.extend({'inputs': rates.tolist(), 'action': action, 'reward': float(a)}
                               for (rates, action), a in zip(trace, advantage))
            if feedback == 'unrelated':
                # Same reward scale, but feedback is assigned to unrelated observations/actions.
                shuffled = feedback_rng.permutation([s['reward'] for s in samples])
                for sample, value in zip(samples, shuffled):
                    sample['reward'] = float(value)
            client.learn(samples)
            baseline = .9 * baseline + .1 * np.mean(batch_returns, axis=0)
            if (batch_start + BATCH_EPISODES) % 32 == 0:
                measured = evaluate(client, VALIDATION_SEEDS)
                validation.append({'episode': batch_start + BATCH_EPISODES, **measured})
                score = measured['aggregate']['mean_return']
                if score > best:
                    best = score
                    best_episode = batch_start + BATCH_EPISODES
                    client.request('POST', '/v1/models/policy/checkpoint', {})
                    shutil.copy2(out/'live/policy.safetensors', out/'best.safetensors')
                print(json.dumps({'run': out.name, 'episodes': batch_start + BATCH_EPISODES,
                                  'validation': measured['aggregate']}), flush=True)
    result = {'feedback': feedback, 'best_episode': best_episode, 'best_validation_return': best,
              'initial_validation': initial_validation, 'validation': validation, 'training': history,
              'elapsed_seconds': time.perf_counter() - started}
    (out/'training.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/driving-game')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use a new output directory')
    out.mkdir(parents=True)
    protocol = {'environment': 'highway-fast-v0', 'config': CONFIG, 'actions': ACTION_NAMES,
                'episodes': EPISODES, 'gamma': GAMMA, 'batch_episodes': BATCH_EPISODES,
                'learning_rates': [.01, .05], 'validation_seeds': VALIDATION_SEEDS, 'test_seeds': TEST_SEEDS,
                'training_seeds': [1000, 1000 + EPISODES - 1],
                'goal_source': 'environment scalar reward; no expert action labels or semantic goal inference',
                'algorithm': 'sampled-action REINFORCE, discounted return minus lagged time-indexed EMA baseline; batch weights frozen until update'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    candidates = []
    for lr in protocol['learning_rates']:
        folder = out/f'lr{lr}'
        folder.mkdir()
        initial = folder/'initial.safetensors'
        command = [str(ROOT/'target/release/novi_engine'), 'init', str(ROOT/'data/pn_kc.tsv'),
                   str(ROOT/'data/kc_mbon.tsv'), str(initial), '5', str(lr), '6', '.2', 'false', 'opponent']
        subprocess.run(command, check=True, capture_output=True)
        (folder/'init-command.json').write_text(json.dumps(command, indent=2) + '\n')
        result = train(initial, folder)
        candidates.append((result['best_validation_return'], lr, folder))
    _, lr, selected = max(candidates, key=lambda item: item[0])
    (out/'selection.json').write_text(json.dumps({'learning_rate': lr, 'folder': selected.name,
                                                 'criterion': 'validation mean episode return; initial model eligible'}, indent=2) + '\n')
    train(selected/'initial.safetensors', out/'unrelated', feedback='unrelated')
    results = {}
    for name, path in [('learned', selected/'best.safetensors'), ('frozen', selected/'initial.safetensors'),
                       ('unrelated', out/'unrelated/best.safetensors')]:
        with server(path, out/f'eval-{name}') as client:
            results[name] = evaluate(client, TEST_SEEDS)
            if name == 'frozen':
                results['random'] = evaluate(client, TEST_SEEDS, 'random')
                results['idle'] = evaluate(client, TEST_SEEDS, 'idle')
                results['slow'] = evaluate(client, TEST_SEEDS, 'slow')
            if name == 'learned':
                _, _, _, frames = episode(client, TEST_SEEDS[0], render=True)
                if frames:
                    frames[0].save(out/'driving.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)
    shutil.copy2(selected/'best.safetensors', out/'model.safetensors')
    summary = {'selected_learning_rate': lr, 'test': results, 'platform': platform.platform(),
               'versions': {name: __import__('importlib.metadata', fromlist=['version']).version(name)
                            for name in ['highway-env', 'gymnasium', 'numpy', 'pygame-ce']},
               'sha256': {os.path.relpath(path, ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), ROOT/'scripts/driving_environment.py', out/'model.safetensors']}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({name: item['aggregate'] for name, item in results.items()}, indent=2))


if __name__ == '__main__':
    main()
