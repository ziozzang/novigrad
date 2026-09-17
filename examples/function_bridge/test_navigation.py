import unittest

import numpy as np

from navigation import (
    AngularSteeringComparator,
    CircularHeadingEstimator,
    build_report,
    circular_error,
    run_episode,
    wrap_degrees,
)


class NavigationTests(unittest.TestCase):
    def test_angular_wrap(self):
        np.testing.assert_allclose(wrap_degrees([180, 181, -181, 540]), [-180, -179, 179, -180])
        self.assertAlmostEqual(float(circular_error(-179, 179)), 2.0)
        controller = AngularSteeringComparator()
        controller.set_target(-179)
        self.assertAlmostEqual(controller.choose_turn(179), 1.1)

    def test_truth_cannot_affect_estimator(self):
        # Two hypothetical worlds with different truth but identical host inputs
        # must produce the same state: update exposes no truth parameter.
        inputs = [(10.0, None, 1.0), (None, 7.0, 1.0), (40.0, 5.0, 0.2)]
        a = CircularHeadingEstimator()
        b = CircularHeadingEstimator()
        sequence_a = [a.update(*values) for values in inputs]
        sequence_b = [b.update(*values) for values in inputs]
        np.testing.assert_allclose(sequence_a, sequence_b)
        self.assertNotIn("true_heading", CircularHeadingEstimator.update.__code__.co_varnames)

    def test_reset_and_instances_are_isolated(self):
        a = CircularHeadingEstimator()
        b = CircularHeadingEstimator()
        a.update(75.0)
        self.assertIsNone(b.heading)
        a.reset()
        self.assertIsNone(a.heading)
        with self.assertRaises(RuntimeError):
            AngularSteeringComparator().choose_turn(a.heading)

    def test_deterministic_seed(self):
        first = run_episode("cue_dropout_odometry", 17)
        second = run_episode("cue_dropout_odometry", 17)
        self.assertEqual(first, second)

    def test_requested_controls_have_expected_separation(self):
        report = build_report()
        aggregate = report["aggregate"]
        with_odom = aggregate["cue_dropout_odometry"]["dropout_heading_rmse_degrees"]["mean"]
        without_odom = aggregate["cue_dropout_no_odometry"]["dropout_heading_rmse_degrees"]["mean"]
        adaptive = aggregate["conflicting_unreliable_cue"]["dropout_heading_rmse_degrees"]["mean"]
        fixed = aggregate["conflicting_fixed_trust"]["dropout_heading_rmse_degrees"]["mean"]
        self.assertLess(with_odom, without_odom)
        self.assertLess(adaptive, fixed)
        self.assertGreater(
            aggregate["goal_switch"]["target_alignment_within_15deg_fraction"]["mean"], 0.6
        )

    def test_validation_rejects_nonfinite_inputs(self):
        estimator = CircularHeadingEstimator()
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                estimator.update(visual_heading=value)
        with self.assertRaises(ValueError):
            estimator.update(visual_heading=0.0, visual_reliability=1.1)


if __name__ == "__main__":
    unittest.main()
