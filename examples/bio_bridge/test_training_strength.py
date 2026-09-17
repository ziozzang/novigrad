import unittest
import numpy as np

from training_strength import sample_action, weight_metrics


class TrainingStrengthTests(unittest.TestCase):
    def test_action_sampling_boundaries(self):
        p = [.1, .2, .3, .4]
        self.assertEqual(sample_action(p, 0.), 0)
        self.assertEqual(sample_action(p, .15), 1)
        self.assertEqual(sample_action(p, .999999), 3)

    def test_weight_metrics_finite_and_saturation(self):
        metrics = weight_metrics([0., .5, -1.], [.1, 1., -1.])
        self.assertTrue(metrics["all_finite"])
        self.assertAlmostEqual(metrics["saturation_fraction"], 2 / 3)
        self.assertGreater(metrics["delta_l2"], 0.)


if __name__ == "__main__":
    unittest.main()
