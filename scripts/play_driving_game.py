#!/usr/bin/env python3
"""Play a saved Novigrad policy in HighwayEnv; no learning or vehicle hardware."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
from driving_environment import CONFIG, make_env, encode_observation
from train_driving_game import server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--human', action='store_true', help='open a local pygame game window')
    parser.add_argument('--gif', type=Path, help='write an offscreen replay GIF')
    args = parser.parse_args()
    if args.gif and args.gif.exists():
        parser.error('GIF output already exists')
    env = make_env(args.seed, 'human' if args.human else 'rgb_array')
    frames, latencies = [], []
    total, steps = 0., 0
    try:
        with tempfile.TemporaryDirectory(prefix='novi-play-') as directory:
            with server(args.model.resolve(), Path(directory)) as client:
                observation, _ = env.reset(seed=args.seed)
                for _ in range(40):
                    started = time.perf_counter()
                    p = client.infer(encode_observation(observation, len(client.ids)))
                    action = int(p.argmax())
                    observation, reward, terminated, truncated, _ = env.step(action)
                    latencies.append((time.perf_counter() - started) * 1000)
                    total += reward
                    steps += 1
                    if args.gif:
                        frames.append(Image.fromarray(env.render()))
                    if terminated or truncated:
                        break
        if frames:
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                           duration=round(1000 / CONFIG['policy_frequency']), loop=0)
        print(json.dumps({'seed': args.seed, 'return': total, 'steps': steps,
                          'collision': bool(env.unwrapped.vehicle.crashed),
                          'decision_frequency_hz': CONFIG['policy_frequency'],
                          'observation_api_simulator_step_mean_ms': float(np.mean(latencies)),
                          'observation_api_simulator_step_max_ms': float(np.max(latencies)),
                          'timing_note': 'includes simulator stepping; human mode may pace rendering, GIF encoding excluded'}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
