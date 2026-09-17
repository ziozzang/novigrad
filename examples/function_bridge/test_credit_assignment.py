import unittest
import numpy as np

from credit_assignment import LinearLearner


class CreditAssignmentTests(unittest.TestCase):
    def test_centering_uses_declared_ranges(self):
        learner = LinearLearner(centered=True)
        x = learner.features(np.array([[0., 0., 0.], [1., 1., 1.]]))
        self.assertTrue(np.array_equal(x[0], [-1., -1., -1., -1., 1.]))
        self.assertTrue(np.array_equal(x[1], [1., 1., 1., 1., 1.]))

    def test_all_updates_remain_finite(self):
        cases = np.array([[.2, .8, .1], [.9, .2, .7]], dtype=float)
        for learner in (LinearLearner(), LinearLearner(centered=True),
                        LinearLearner(centered=True, value_baseline=True),
                        LinearLearner(centered=True, value_baseline=True, entropy=.02)):
            learner.learn(cases, np.array([0, 1]), np.array([.06, 0.]))
            self.assertTrue(np.isfinite(learner.actor).all())
            self.assertTrue(np.isfinite(learner.critic).all())


if __name__ == "__main__":
    unittest.main()
