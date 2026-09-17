import unittest

import numpy as np

from context_memory import Config, ContextPolicy, context_labels, encode_context, signed_features


class ContextMemoryTests(unittest.TestCase):
    def test_signed_features_are_nonnegative_and_normalized(self):
        x = np.zeros((2, 64), np.float32)
        x[0, 0], x[0, 1], x[1, 2] = 3, -4, 2
        encoded = signed_features(x)
        self.assertEqual(encoded.shape, (2, 128))
        self.assertTrue(np.all(encoded >= 0))
        np.testing.assert_allclose(encoded.sum(1), [1.4, 1.0])

    def test_context_encodings(self):
        base = np.arange(128, dtype=np.float32)[None, :]
        blind_a = encode_context(base, "blind", 0)
        blind_b = encode_context(base, "blind", 1)
        np.testing.assert_array_equal(blind_a, blind_b)
        concat_a, concat_b = encode_context(base, "concat", 0), encode_context(base, "concat", 1)
        self.assertEqual(concat_a[0, 128], 1)
        self.assertEqual(concat_b[0, 129], 1)
        unit_base = base / np.linalg.norm(base, axis=1, keepdims=True)
        raw_concat = encode_context(unit_base, "concat", 0)
        normalized_a = encode_context(unit_base, "concat_normalized", 0)
        normalized_b = encode_context(unit_base, "concat_normalized", 1)
        np.testing.assert_allclose(np.linalg.norm(raw_concat, axis=1), np.sqrt(2), rtol=0, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(normalized_a, axis=1), 1.0, rtol=0, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(normalized_b, axis=1), 1.0, rtol=0, atol=1e-7)
        conjunctive_a = encode_context(base, "conjunctive", 0)
        conjunctive_b = encode_context(base, "conjunctive", 1)
        np.testing.assert_array_equal(conjunctive_a[0, :128], base[0])
        np.testing.assert_array_equal(conjunctive_b[0, 128:256], base[0])
        self.assertFalse(np.any(conjunctive_a[0, 128:256]))

    def test_mapping_is_identity_then_rotate(self):
        y = np.arange(4)
        np.testing.assert_array_equal(context_labels(y, 0), y)
        np.testing.assert_array_equal(context_labels(y, 1), [1, 2, 3, 0])

    def test_modular_engines_start_identically_and_are_separate(self):
        policy = ContextPolicy("modular", Config())
        self.assertEqual(policy.engines[0].weights, policy.engines[1].weights)
        self.assertIsNot(policy.engines[0], policy.engines[1])

    def test_validation(self):
        with self.assertRaises(ValueError):
            encode_context(np.zeros((1, 127)), "blind", 0)
        with self.assertRaises(ValueError):
            context_labels([4], 0)


if __name__ == "__main__":
    unittest.main()
