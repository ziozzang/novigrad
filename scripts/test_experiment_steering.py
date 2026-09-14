import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_steering import summarize, validated_data


class SteeringExperimentTest(unittest.TestCase):
    def fixture(self, path, leakage=False):
        arrays = {}
        for split in ('train', 'validation', 'test'):
            arrays[f'{split}_images'] = np.zeros((3, 28, 28), dtype=np.uint8)
            arrays[f'{split}_labels'] = np.array([0, 1, 2], dtype=np.int64)
            arrays[f'{split}_steer'] = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
            arrays[f'{split}_groups'] = np.array([split + str(i) for i in range(3)], dtype='<U32')
        if leakage:
            arrays['test_groups'][0] = arrays['train_groups'][0]
        np.savez(path, **arrays)

    def test_group_leakage_and_bin_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.npz'
            self.fixture(path)
            _, settings = validated_data(path)
            self.assertEqual(settings['group_counts'], {'train': 3, 'validation': 3, 'test': 3})
            self.fixture(path, leakage=True)
            with self.assertRaisesRegex(ValueError, 'group leakage'):
                validated_data(path)
            self.fixture(path)
            arrays = dict(np.load(path, allow_pickle=False))
            arrays['test_labels'][0] = 2
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'steer bins'):
                validated_data(path)

    def test_probability_decoding_and_confusion(self):
        data = {'test_labels': np.array([0, 1, 2]),
                'test_steer': np.array([-0.5, 0.0, 0.5]),
                'test_groups': np.array(['a', 'a', 'b'])}
        settings = {'train_class_steer_mean': [-0.5, 0.0, 0.5],
                    'train_constant_median_steer': 0.0}
        probabilities = np.array([[0.8, 0.1, 0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = summarize(data, settings, probabilities.argmax(axis=1), probabilities)
        self.assertEqual(result['accuracy'], 1.0)
        self.assertEqual(result['balanced_accuracy'], 1.0)
        self.assertEqual(result['confusion_rows_actual_columns_predicted'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(result['probability_decoded_steer_mae'], 0.05)
        self.assertAlmostEqual(result['train_median_constant_steer_mae'], 1 / 3)
        self.assertEqual(result['per_group']['a']['count'], 2)


if __name__ == '__main__':
    unittest.main()
