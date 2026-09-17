import unittest
import numpy as np
from architecture_sites import designs, sinkhorn, mixing_sanity


class SiteTests(unittest.TestCase):
    def test_capacity_and_determinism(self):
        rng = np.random.default_rng(1)
        reps = {k: rng.normal(size=(6, d)) for k, d in
                [('embedding', 768), ('pn', 319), ('raw_kc', 5177), ('kc', 5177), ('mbon', 96)]}
        a, b = designs(reps, 12), designs(reps, 12)
        for key in a:
            self.assertEqual(a[key].shape, (6, 32))
            np.testing.assert_array_equal(a[key], b[key])

    def test_stability_is_not_identity(self):
        r = mixing_sanity()
        self.assertLess(r['row_sum_error'], 1e-10)
        self.assertLessEqual(r['norm_ratio_after_100'], 1 + 1e-10)
        self.assertEqual(r['uniform_stream_rank'], 1)
        self.assertTrue(r['permutation_is_doubly_stochastic'])
        with self.assertRaises(ValueError):
            sinkhorn(np.ones((3, 4)))


if __name__ == '__main__':
    unittest.main()
