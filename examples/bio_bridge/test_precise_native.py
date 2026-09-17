import unittest

import numpy as np

from precise_native import evaluate, sample_actions, training_stream


class FakeEngine:
    def infer_batch(self, rows):
        return [[0.7, 0.1, 0.1, 0.1] if row[0] else [0.1, 0.7, 0.1, 0.1] for row in rows]


class PreciseNativeTests(unittest.TestCase):
    def test_stream_is_deterministic_and_balanced_by_epoch(self):
        a_indices, a_uniforms = training_stream(701, 32)
        b_indices, b_uniforms = training_stream(701, 32)
        np.testing.assert_array_equal(a_indices, b_indices)
        np.testing.assert_array_equal(a_uniforms, b_uniforms)
        self.assertEqual(len(a_indices), 2048)
        for epoch in range(64):
            np.testing.assert_array_equal(np.sort(a_indices[epoch * 32:(epoch + 1) * 32]), np.arange(32))

    def test_sample_actions_inverse_cdf(self):
        probability = np.array([[.25, .25, .25, .25], [.1, .2, .3, .4]])
        np.testing.assert_array_equal(sample_actions(probability, [.0, .61]), [0, 3])

    def test_evaluate_confusion_and_languages(self):
        rates = np.array([[1., 0.], [0., 1.], [1., 0.]], np.float32)
        labels = np.array([0, 1, 1])
        result = evaluate(FakeEngine(), rates, labels, np.array(["en", "ko", "en"]))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(sum(map(sum, result["confusion"])), 3)
        self.assertEqual(result["accuracy_by_language"], {"en": .5, "ko": 1.0})


if __name__ == "__main__":
    unittest.main()
