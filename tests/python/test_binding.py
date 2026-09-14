import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest

from novigrad import Engine

ROOT = Path(__file__).resolve().parents[2]


class NativeBindingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.input_id = 2**60 + 7
        (self.root/'input.tsv').write_text(f'{self.input_id}\t201\t1\t1\n{self.input_id+1}\t202\t1\t1\n')
        (self.root/'plastic.tsv').write_text('201\t301\t1\t1\n201\t302\t1\t1\n202\t301\t1\t1\n202\t302\t1\t1\n')
        self.engine = Engine.from_edges(self.root/'input.tsv', self.root/'plastic.tsv',
                                       learning_rate=.1, active_fraction=1., homeostasis=False)
        self.rows = [[1., 0.], [0., 1.]]

    def tearDown(self):
        self.directory.cleanup()

    def test_learn_save_reload_resume_and_no_clobber(self):
        self.assertEqual(self.engine.input_ids, [self.input_id, self.input_id+1])
        before = self.engine.infer_batch(self.rows)
        for _ in range(30):
            self.engine.learn(self.rows, [0, 1], [1., 1.])
        after = self.engine.infer_batch(self.rows)
        self.assertGreater(after[0][0], .9)
        self.assertGreater(after[1][1], .9)
        self.assertNotEqual(before, after)
        path = self.root/'model.safetensors'
        self.engine.save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        restored = Engine.load(path)
        self.assertEqual(restored.infer_batch(self.rows), after)
        with self.assertRaises(OSError):
            self.engine.save(path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for model in [self.engine, restored]:
            model.learn(self.rows, [1, 0], [-.3, -.2])
        self.assertEqual(self.engine.weights, restored.weights)
        restored.save(path, overwrite=True)
        self.assertEqual(Engine.load(path).weights, restored.weights)

    def test_invalid_whole_batch_is_atomic_and_recovers(self):
        weights = self.engine.weights
        for rows, actions, rewards in [
            ([[1., 0.], [float('nan'), 1.]], [0, 1], [1., 1.]),
            (self.rows, [0, 2], [1., 1.]),
            (self.rows, [0, 1], [1., float('inf')]),
            (self.rows, [0], [1.]),
        ]:
            with self.assertRaises(ValueError):
                self.engine.learn(rows, actions, rewards)
            self.assertEqual(self.engine.weights, weights)
        with self.assertRaises(ValueError):
            self.engine.infer([-1., 0.])
        self.engine.learn(self.rows, [0, 1], [1., 1.])
        self.assertNotEqual(self.engine.weights, weights)
        self.engine.save(self.root/'recovered.safetensors')

    def test_decoder_and_independent_threads(self):
        self.engine.set_output_gains([1., -1.])
        self.engine.set_output_actions([1, 0])
        self.engine.save(self.root/'decoder.safetensors')
        models = [Engine.load(self.root/'decoder.safetensors') for _ in range(2)]
        self.assertEqual(models[0].output_actions, [1, 0])
        self.assertEqual(models[0].output_gains, [1., -1.])
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda model: model.infer_batch(self.rows * 32), models))
        self.assertEqual(values[0], values[1])
        self.assertEqual(models[0].weights, models[1].weights)

    def test_rust_cli_parity(self):
        binary = ROOT/'target/release/classify_features'
        if not binary.exists():
            self.skipTest('build classify_features for cross-interface validation')
        import numpy as np
        from safetensors.numpy import save_file
        checkpoint = self.root/'cli.safetensors'
        self.engine.learn(self.rows, [0, 1], [1., 1.])
        self.engine.save(checkpoint)
        features = self.root/'features.safetensors'
        save_file({'inputs': np.asarray(self.rows, dtype=np.float32),
                   'input_ids': np.asarray(self.engine.input_ids, dtype=np.uint64)}, str(features))
        output = subprocess.check_output([str(binary), str(checkpoint), str(features)], text=True)
        rows = list(csv.DictReader(io.StringIO(output), delimiter='\t'))
        expected = self.engine.infer_batch(self.rows)
        for i, row in enumerate(rows):
            for action in range(2):
                self.assertAlmostEqual(float(row[f'p{action}']), expected[i][action], places=8)


if __name__ == '__main__':
    unittest.main()
