import unittest

from reliability import (
    ALGORITHMS, SCENARIOS, SEEDS, SensorInputs, build_report,
    estimate_headings, generate_trial, run_one,
)


class ReliabilityTests(unittest.TestCase):
    def test_deterministic_and_matched_stream(self):
        self.assertEqual(generate_trial("abrupt_visual_bias", 1001), generate_trial("abrupt_visual_bias", 1001))
        run = run_one("clean", 1001)
        self.assertEqual(set(run["algorithms"]), set(ALGORITHMS))

    def test_truth_is_evaluator_only(self):
        self.assertNotIn("truth", estimate_headings.__code__.co_varnames)
        sensors = SensorInputs((10.0, 12.0, 15.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0))
        self.assertEqual(estimate_headings("innovation_gate", sensors), estimate_headings("innovation_gate", sensors))

    def test_gate_rejects_large_innovation(self):
        sensors = SensorInputs((0.0, 100.0), (0.0, 2.0), (1.0, 0.0))
        self.assertAlmostEqual(estimate_headings("innovation_gate", sensors)[1], 2.0)
        self.assertGreater(abs(estimate_headings("fixed_trust", sensors)[1]), 20.0)

    def test_full_frozen_report_and_expected_controls(self):
        report = build_report()
        self.assertEqual(report["seeds"], list(SEEDS))
        self.assertEqual(len(report["runs"]), len(SEEDS) * len(SCENARIOS))
        self.assertIn("cluster bootstrap", report["bootstrap_units"]["all_scenarios"])
        self.assertIn("not learned", report["algorithm_notes"]["innovation_gate"])
        agg = report["aggregate"]
        self.assertLess(agg["abrupt_visual_bias"]["innovation_gate"]["mae_degrees"]["mean"],
                        agg["abrupt_visual_bias"]["fixed_trust"]["mae_degrees"]["mean"])
        self.assertLess(agg["dropout"]["innovation_gate"]["mae_degrees"]["mean"], 10.0)
        self.assertLess(agg["odometry_bias"]["fixed_trust"]["mae_degrees"]["mean"],
                        agg["odometry_bias"]["no_vision"]["mae_degrees"]["mean"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            generate_trial("missing", 1001)
        with self.assertRaises(ValueError):
            estimate_headings("missing", SensorInputs((0.0,), (0.0,), (1.0,)))
        with self.assertRaises(ValueError):
            estimate_headings("fixed_trust", SensorInputs((None,), (0.0,), (1.0,)))


if __name__ == "__main__":
    unittest.main()
