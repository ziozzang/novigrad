import unittest
import numpy as np

from eligibility_mechanism import LinearTracePolicy, gradient_event, norm_match, softmax


class EligibilityMechanismTests(unittest.TestCase):
    def test_reward_before_cue_has_zero_association(self):
        policy = LinearTracePolicy(features=3, tau=8)
        policy.update(1., policy.trace)
        self.assertTrue(np.array_equal(policy.weights, np.zeros((3, 4))))

    def test_impulse_after_cue_decays_in_correct_order(self):
        x = np.array([1., 0., 0.])
        event = gradient_event(x, 0, softmax(np.zeros(4)))
        policy = LinearTracePolicy(features=3, tau=8)
        policy.advance_trace(event)
        for _ in range(4):
            policy.advance_trace(np.zeros_like(event))
        self.assertTrue(np.allclose(policy.trace, event * np.exp(-4 / 8)))

    def test_score_primitives_use_float64_and_remain_finite(self):
        p = softmax(np.array([1000., 999., -1000., 0.]))
        self.assertEqual(p.dtype, np.float64)
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.)

    def test_norm_match_preserves_direction_and_matches_event_magnitude(self):
        trace = np.array([[3., 4.], [0., 0.]])
        event = np.array([[0., 2.], [0., 0.]])
        matched = norm_match(trace, event)
        self.assertAlmostEqual(float(np.linalg.norm(matched)), 2.)
        self.assertTrue(np.allclose(matched / np.linalg.norm(matched), trace / np.linalg.norm(trace)))


if __name__ == "__main__":
    unittest.main()
