import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from precise_permutation import (
    SEEDS,
    _save_decoder,
    _selected_trial,
    permute_ports,
    port_permutation,
)


class PrecisePermutationTests(unittest.TestCase):
    def test_permutation_is_deterministic_bijection(self):
        observed = [port_permutation(seed) for seed in SEEDS]
        for seed, order in zip(SEEDS, observed):
            np.testing.assert_array_equal(order, port_permutation(seed))
            np.testing.assert_array_equal(np.sort(order), np.arange(256))
        self.assertEqual(len({order.tobytes() for order in observed}), len(SEEDS))

    def test_only_first_256_ports_move(self):
        rows = np.arange(2 * 319, dtype=np.float32).reshape(2, 319)
        order = port_permutation(811)
        moved = permute_ports(rows, order)
        np.testing.assert_array_equal(moved[:, :256], rows[:, order])
        np.testing.assert_array_equal(moved[:, 256:], rows[:, 256:])
        np.testing.assert_array_equal(np.sort(moved[:, :256], axis=1), np.sort(rows[:, :256], axis=1))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            port_permutation(1)
        with self.assertRaises(ValueError):
            permute_ports(np.zeros((1, 318), np.float32), np.arange(256))
        bad = np.arange(256); bad[-1] = 0
        with self.assertRaises(ValueError):
            permute_ports(np.zeros((1, 319), np.float32), bad)

    def test_artifact_roundtrip_contains_decoder_and_permutation(self):
        basis = np.arange(15, dtype=np.float64).reshape(3, 5)
        coefficients = np.arange(12, dtype=np.float64).reshape(3, 4)
        order = port_permutation(812)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.safetensors"
            _save_decoder(path, basis, coefficients, order, "pca", 812, .1)
            restored = load_file(str(path))
            np.testing.assert_array_equal(restored["basis"], basis)
            np.testing.assert_array_equal(restored["coefficients"], coefficients)
            np.testing.assert_array_equal(restored["permutation"], order)

    def test_validation_ties_prefer_smaller_ridge(self):
        trials = [
            {"ridge": .1, "accuracy": .5, "cosine": .7},
            {"ridge": .01, "accuracy": .5, "cosine": .7},
            {"ridge": 1., "accuracy": .4, "cosine": .9},
        ]
        self.assertEqual(_selected_trial(trials)["ridge"], .01)


if __name__ == "__main__":
    unittest.main()
