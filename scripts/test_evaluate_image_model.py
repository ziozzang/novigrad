"""Boundary and scoring tests using synthetic data only; no evaluation datasets."""
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file

import evaluate_image_model as evaluator

HEADER = 'index\tpredicted\t' + '\t'.join(f'p{i}' for i in range(10)) + '\n'


def backend_row(index, predicted):
    return '\t'.join([str(index), str(predicted)] + ['0.8' if i == predicted else '0.022222222' for i in range(10)]) + '\n'


class EvaluatorTests(unittest.TestCase):
    def test_malformed_backend_output_is_rejected(self):
        valid = HEADER + backend_row(0, 0)
        malformed = {
            'missing row': HEADER,
            'extra row': valid + backend_row(1, 1),
            'duplicate row index': HEADER + backend_row(1, 0),
            'missing probability': valid.rsplit('\t', 1)[0] + '\n',
            'extra probability': valid.rstrip('\n') + '\t0\n',
            'nonfinite probability': valid.replace('0.8', 'nan'),
            'negative probability': valid.replace('0.8', '-0.8'),
            'unnormalized probabilities': valid.replace('0.8', '0.5'),
            'invalid predicted class': valid.replace('0\t0\t', '0\t10\t', 1),
            'nonmaximal prediction': valid.replace('0\t0\t', '0\t1\t', 1),
        }
        for case, stdout in malformed.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                evaluator.parse_predictions(stdout, 1)

    def test_invalid_dataset_shapes_and_labels_are_rejected(self):
        valid = np.zeros((2, 28, 112), dtype=np.uint8)
        labels = np.zeros((2, 4), dtype=np.uint8)
        cases = [
            (np.array(0, dtype=np.uint8), labels),
            (np.zeros((0, 28, 112), dtype=np.uint8), labels[:0]),
            (np.zeros((2, 112, 28), dtype=np.uint8), labels),
            (valid.astype(np.float32), labels),
            (valid, labels.reshape(-1)),
            (valid, labels.astype(np.float32)),
            (valid, np.full((2, 4), -1, dtype=np.int8)),
            (valid, np.full((2, 4), 10, dtype=np.uint8)),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.npz'
            for images, answers in cases:
                with self.subTest(shape=images.shape, label_dtype=answers.dtype):
                    np.savez(path, images=images, labels=answers)
                    with self.assertRaises(ValueError):
                        evaluator.load_images(path, None, 4)

    def test_sequence_scoring_cell_order_and_label_free_backend(self):
        class StubAdapter:
            input_ids = np.array([101, 102], dtype=np.uint64)

            def transform(self, cells):
                # Distinct pixel values identify all eight positions without labels.
                np.testing.assert_array_equal(cells[:, 0, 0], np.arange(8))
                return np.column_stack((cells[:, 0, 0], np.ones(8))).astype(np.float32)

        def backend(command, **kwargs):
            inputs = load_file(command[2])
            self.assertEqual(set(inputs), {'inputs', 'input_ids'})
            np.testing.assert_array_equal(inputs['inputs'][:, 0], np.arange(8))
            # Seven correct characters, but only the first entire code is correct.
            predictions = [0, 1, 2, 3, 4, 5, 6, 0]
            return subprocess.CompletedProcess(command, 0, HEADER + ''.join(backend_row(i, p) for i, p in enumerate(predictions)), '')

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cells = np.broadcast_to(np.arange(8, dtype=np.uint8)[:, None, None], (8, 28, 28))
            images = cells.reshape(2, 4, 28, 28).transpose(0, 2, 1, 3).reshape(2, 28, 112)
            np.savez(root / 'input.npz', images=images, labels=np.arange(8, dtype=np.uint8).reshape(2, 4))
            (root / 'adapter.safetensors').write_bytes(b'adapter')
            (root / 'model.safetensors').write_bytes(b'model')
            (root / 'bundle.json').write_text(json.dumps({'format': 'nobi.image_bundle.v1',
                'cell_size': [28, 28], 'sequence_length': 4, 'class_labels': list(map(str, range(10))),
                'adapter': 'adapter.safetensors', 'model': 'model.safetensors'}))
            with patch.object(evaluator, 'ROOT', root), patch.object(evaluator.ImageRateAdapter, 'load', return_value=StubAdapter()), patch.object(evaluator.subprocess, 'run', side_effect=backend):
                metrics = evaluator.evaluate(root / 'bundle.json', root / 'input.npz', None, root / 'out', root / 'backend')
            self.assertEqual(metrics['accuracy'], 7 / 8)
            self.assertEqual(metrics['exact_sequence_accuracy'], 1 / 2)
            self.assertAlmostEqual(metrics['cross_entropy'], -(7 * math.log(.8) + math.log(.022222222)) / 8)
            self.assertAlmostEqual(metrics['confidence']['mean'], .8)
            self.assertEqual(metrics['confusion_matrix'][7][0], 1)
            self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), 8)
            self.assertEqual(metrics['bundle'], 'bundle.json')
            self.assertEqual(metrics['dataset'], 'input.npz')
            with (root / 'out' / 'predictions.tsv').open() as output:
                rows = list(csv.DictReader(output, delimiter='\t'))
            self.assertEqual((rows[-1]['sample'], rows[-1]['position'], rows[-1]['true'], rows[-1]['predicted']), ('1', '3', '7', '0'))
            self.assertEqual(json.loads((root / 'out' / 'metrics.json').read_text()), metrics)

    def test_external_provenance_path_remains_absolute(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'external.npz'
            self.assertEqual(evaluator.provenance_path(path), str(path.resolve()))


if __name__ == '__main__':
    unittest.main()
