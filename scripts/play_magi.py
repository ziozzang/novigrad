#!/usr/bin/env python3
"""Load a three-policy Safetensors bundle and watch its closed-loop vote."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile

from PIL import Image
from driving_environment import ACTION_NAMES, make_env, encode_observation
from ensemble_policy import EnsembleClient
from train_driving_game import server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--mode', choices=['majority', 'mean'])
    parser.add_argument('--seed', type=int, default=40000)
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--gif', type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    if bundle.get('format') != 'novi.ensemble.v1' or bundle.get('actions') != ACTION_NAMES or bundle.get('encoding') != 'driving_environment.encode_observation' or len(bundle.get('members', [])) != 3:
        parser.error('expected compatible three-member driving ensemble bundle')
    if args.gif and args.gif.exists():
        parser.error('GIF output already exists')
    paths = [(args.bundle.parent/path).resolve() for path in bundle['members']]
    frames, total = [], 0.
    env = make_env(args.seed, 'human' if args.human else 'rgb_array')
    try:
        with tempfile.TemporaryDirectory(prefix='novi-magi-') as directory, ExitStack() as stack:
            members = [stack.enter_context(server(path, Path(directory)/str(i))) for i, path in enumerate(paths)]
            ensemble = EnsembleClient(members, args.mode or bundle['default_mode'])
            observation, _ = env.reset(seed=args.seed)
            for _ in range(40):
                p = ensemble.infer(encode_observation(observation, len(ensemble.ids)))
                observation, reward, terminated, truncated, _ = env.step(int(p.argmax()))
                total += reward
                if args.gif:
                    frames.append(Image.fromarray(env.render()))
                if terminated or truncated:
                    break
            if frames:
                args.gif.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=500, loop=0)
            print(json.dumps({'return': total, 'collision': bool(env.unwrapped.vehicle.crashed),
                              'voting': ensemble.statistics()}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
