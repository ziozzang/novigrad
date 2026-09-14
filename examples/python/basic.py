#!/usr/bin/env python3
"""Standalone learning/save/load example: no download, NumPy, API or CLI required."""
from pathlib import Path
import json
import tempfile

from novigrad import Engine


def main():
    with tempfile.TemporaryDirectory(prefix='novigrad-python-') as directory:
        root = Path(directory)
        (root/'input.tsv').write_text('101\t201\t1\t1\n102\t202\t1\t1\n')
        (root/'plastic.tsv').write_text('201\t301\t1\t1\n201\t302\t1\t1\n202\t301\t1\t1\n202\t302\t1\t1\n')
        engine = Engine.from_edges(root/'input.tsv', root/'plastic.tsv', actions=2,
                                   learning_rate=.1, active_fraction=1., homeostasis=False)
        observations = [[1., 0.], [0., 1.]]
        before = engine.infer_batch(observations)
        # Demonstration teacher rewards; game examples obtain scalar rewards by acting.
        for _ in range(30):
            engine.learn(observations, [0, 1], [1., 1.])
        after = engine.infer_batch(observations)
        checkpoint = root/'model.safetensors'
        engine.save(checkpoint)
        restored = Engine.load(checkpoint)
        assert after == restored.infer_batch(observations)
        assert after[0][0] > .9 and after[1][1] > .9
        print(json.dumps({'input_ids': engine.input_ids, 'before': before, 'after': after,
                          'safetensors_roundtrip_exact': True}, indent=2))


if __name__ == '__main__':
    main()
