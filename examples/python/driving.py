#!/usr/bin/env python3
"""Reuse the driving environment through a native Engine, without an API process."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from novigrad import Engine
from train_driving_game import episode


class NativeClient:
    """Minimal adapter for the existing game/ensemble client protocol."""
    def __init__(self, checkpoint):
        self.engine = Engine.load(checkpoint)
        self.ids = [str(value) for value in self.engine.input_ids]

    def infer(self, rates):
        probabilities = np.asarray(self.engine.infer(rates.tolist()), dtype=np.float64)
        return probabilities / probabilities.sum()

    def learn(self, samples):
        self.engine.learn([s['inputs'] for s in samples], [s['action'] for s in samples],
                          [s['reward'] for s in samples])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--seed', type=int, default=20000)
    args = parser.parse_args()
    client = NativeClient(args.model)
    started = time.perf_counter()
    result, _, _, _ = episode(client, args.seed)
    result['wall_seconds'] = time.perf_counter() - started
    result['backend'] = 'native PyO3 Rust engine; no HTTP/subprocess'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
