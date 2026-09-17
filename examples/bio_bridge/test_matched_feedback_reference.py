import unittest

import numpy as np

from bilateral_lm import target_text
from matched_feedback_reference import matched_rollout, parsed_command


class MatchedFeedbackReferenceTests(unittest.TestCase):
    def test_first_command_is_exact_even_when_prior_prefers_another(self):
        raw = target_text(0)
        prior = np.array([.01, .01, .01, .97])
        for mode in ("forced_first_bayes", "forced_first_no_repeat"):
            result = matched_rollout(prior, 1, [0, 1, 2, 3], raw, mode)
            self.assertEqual(parsed_command(raw), 0)
            self.assertEqual(result["trace"][0]["command"], 0)
            self.assertEqual(result["trace"][0]["raw"], raw)

    def test_invalid_first_generation_stops_without_action(self):
        result = matched_rollout(np.full(4, .25), 0, [0, 1, 2, 3], "not a call",
                                 "forced_first_bayes")
        self.assertTrue(result["invalid"])
        self.assertEqual(result["physical_actions"], 0)
        self.assertEqual(result["decisions"], 1)
        self.assertFalse(result["success"])
        self.assertNotIn("executed_action", result["trace"][0])

    def test_no_repeat_uses_every_command_once_when_goal_unreachable(self):
        mapping = [0, 3, 3, 3]
        result = matched_rollout(np.array([.4, .3, .2, .1]), 2, mapping, target_text(1),
                                 "forced_first_no_repeat")
        commands = [row["command"] for row in result["trace"]]
        self.assertEqual(commands[0], 1)
        self.assertEqual(len(commands), 4)
        self.assertEqual(set(commands), {0, 1, 2, 3})
        self.assertEqual(set(row["executed_action"] for row in result["trace"]), {0, 3})
        self.assertFalse(result["success"])

    def test_controller_choice_has_no_target_parameter(self):
        raw = target_text(2)
        prior = np.array([.6, .2, .1, .1])
        a = matched_rollout(prior, 0, [0, 1, 2, 3], raw, "forced_first_bayes")
        b = matched_rollout(prior, 1, [0, 1, 2, 3], raw, "forced_first_bayes")
        self.assertEqual(a["trace"][0]["command"], b["trace"][0]["command"])
        self.assertEqual(a["trace"][0]["belief"], b["trace"][0]["belief"])


if __name__ == "__main__":
    unittest.main()
