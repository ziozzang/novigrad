import unittest

import numpy as np

from inhibition_mechanism import (
    ShadowEngine, calibrate, checkpoint, dual_ridge_scores, fit_dual_ridge, load_inputs,
    paired_distractors, predict_dual_ridge, row_cosine,
)
from novigrad import Engine


class InhibitionMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets, _ = load_inputs()
        cls.shadow = ShadowEngine(checkpoint(601))
        cls.calibration = calibrate(cls.shadow, cls.datasets["old_train"][0])

    def test_native_parity(self):
        rows = self.datasets["reused_confirmation_32"][0][:8]
        native = np.asarray(Engine.load(checkpoint(601)).infer_batch(rows.tolist()))
        shadow, hidden = self.shadow.forward(rows, "native_topk_2", self.calibration)
        np.testing.assert_allclose(shadow, native, atol=2e-6, rtol=2e-6)
        self.assertEqual(hidden.shape, (8, self.shadow.hidden_count))

    def test_training_only_calibration_matches_budget(self):
        self.assertLess(abs(self.calibration["global_train_active_fraction"] - 0.02), 0.001)
        self.assertLess(abs(self.calibration["local_train_active_fraction"] - 0.02), 0.001)

    def test_topk_counts_and_probabilities(self):
        rows = self.datasets["old_train"][0][:3]
        for variant, expected in (("native_topk_2", 104), ("topk_20", 1036)):
            probability, hidden = self.shadow.forward(rows, variant, self.calibration)
            np.testing.assert_array_equal(np.count_nonzero(hidden, axis=1), expected)
            np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)

    def test_distractors_rotate_labels(self):
        rows, labels = self.datasets["reused_confirmation_32"]
        distractors = paired_distractors(rows, labels)
        self.assertEqual(distractors.shape, rows.shape)
        self.assertTrue(np.isfinite(row_cosine(rows, distractors)).all())

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.shadow.raw_hidden(np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            self.shadow.forward(self.datasets["old_train"][0][:1], "missing", self.calibration)

    def test_dual_ridge_matches_primal_fixture(self):
        train = np.array([[1., 0.], [0., 1.], [1., 1.]])
        labels = np.array([0, 1, 0])
        test = np.array([[2., 1.], [1., 3.]])
        dual = dual_ridge_scores(train, labels, test, ridge=.1, actions=2)
        normalized_train = train / np.linalg.norm(train, axis=1, keepdims=True)
        normalized_test = test / np.linalg.norm(test, axis=1, keepdims=True)
        target = np.eye(2)[labels]
        primal = normalized_test @ np.linalg.solve(
            normalized_train.T @ normalized_train + .1 * np.eye(2), normalized_train.T @ target
        )
        np.testing.assert_allclose(dual, primal, atol=1e-12)
        basis, coefficients = fit_dual_ridge(train, labels, ridge=.1, actions=2)
        np.testing.assert_array_equal(predict_dual_ridge(basis, coefficients, test), dual)


if __name__ == "__main__":
    unittest.main()
