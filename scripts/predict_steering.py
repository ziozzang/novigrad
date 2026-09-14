#!/usr/bin/env python3
"""Decode an offline steering bundle through local Rust or the existing HTTP API."""
import argparse
import json
from pathlib import Path

import numpy as np
from predict_image import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/release/classify_features'))
    parser.add_argument('--api-url')
    parser.add_argument('--model')
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    means = np.asarray(bundle['train_class_steer_mean'], dtype=np.float64)
    if bundle['class_labels'] != ['left', 'straight', 'right'] or means.shape != (3,) or not np.isfinite(means).all() or np.any(np.abs(means) > 1):
        raise ValueError('expected a three-action steering bundle with normalized class means')
    result = predict(args.bundle, args.image, args.binary, api_url=args.api_url, model=args.model)
    result['normalized_steer'] = float(np.asarray(result['probabilities'][0]) @ means)
    result['decoder'] = 'probability-weighted training class means; offline imitation'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
