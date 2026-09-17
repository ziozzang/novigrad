import unittest

import numpy as np
from internal_state import encode, evaluate, make_cases, reference_metrics, utility


class InternalStateTests(unittest.TestCase):
    def test_encoding_is_nonnegative_319_ports_and_hunger_ablation(self):
        cases = np.array([[0.1, 0.7, 0.2], [0.9, 0.7, 0.2]], np.float32)
        full, ablated = encode(cases), encode(cases, False)
        self.assertEqual(full.shape, (2, 319))
        self.assertEqual(ablated.shape, (2, 319))
        self.assertTrue(np.isfinite(full).all() and (full >= 0).all())
        self.assertFalse(np.array_equal(full[0], full[1]))
        self.assertTrue(np.array_equal(ablated[0], ablated[1]))

    def test_utility_and_independent_continuous_cases(self):
        a = make_cases(np.random.default_rng(1), 20)
        b = make_cases(np.random.default_rng(2), 20)
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(np.allclose(
            utility(np.array([[1., 0.8, 0.3], [0., 1., 0.4]])), [0.5, -0.4]))

        class Perfect:
            def probabilities(self, cases):
                approach = (utility(cases) > 0).astype(float)
                return np.stack([approach, 1 - approach], axis=1)

        result = evaluate(Perfect(), a)
        self.assertEqual(result["regret"], 0.0)
        self.assertTrue(np.isclose(result["utility"], result["best_possible_utility"]))
        refs = reference_metrics(a)
        self.assertEqual(refs["full_information_optimal"]["regret"], 0.0)
        self.assertGreaterEqual(refs["hunger_blind_optimal"]["regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
