import unittest

import numpy as np
import torch

import causal_feedback as cf
from bilateral_lm import target_text
from hard_feedback_routing import hard_next_prefix, hard_rows, temporary_hard_routing


class HardFeedbackRoutingTests(unittest.TestCase):
    def test_argmax_routes_exact_prototype_and_preserves_initial_norm(self):
        initial = torch.tensor([[3., 4.], [0., 0.]])
        prototypes = torch.zeros((4, 2, 2))
        prototypes[0, 0, 0] = 2.
        prototypes[1, 0, 1] = 3.
        prototypes[2, 1, 0] = 7.
        prototypes[3, 1, 1] = 11.
        value = hard_next_prefix(initial, prototypes, [.1, .6, .2, .1], "belief_full", None)
        self.assertAlmostEqual(float(value.norm()), float(initial.norm()), places=6)
        self.assertEqual(int(torch.argmax(value).item()), int(torch.argmax(prototypes[1]).item()))
        self.assertTrue(torch.allclose(value / value.norm(), prototypes[1] / prototypes[1].norm()))

    def test_override_is_scoped_and_restored_after_success(self):
        original = cf.next_prefix
        with temporary_hard_routing():
            self.assertIs(cf.next_prefix, hard_next_prefix)
        self.assertIs(cf.next_prefix, original)

    def test_override_is_restored_after_exception(self):
        original = cf.next_prefix
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with temporary_hard_routing():
                raise RuntimeError("boom")
        self.assertIs(cf.next_prefix, original)

    def test_invalid_belief_and_wrong_mode_are_rejected(self):
        initial = torch.ones((2, 2))
        prototypes = torch.ones((4, 2, 2))
        with self.assertRaises(ValueError):
            hard_next_prefix(initial, prototypes, [1., np.nan, 0., 0.], "belief_full", None)
        with self.assertRaises(ValueError):
            hard_next_prefix(initial, prototypes, [.25] * 4, "original", None)

    def test_complete_cached_rollout_replays_exactly(self):
        initial = torch.zeros((2, 4, 3))
        initial[:, 0, 0] = 1.
        initial[1, 0, 0] = 0.
        initial[1, 0, 1] = 1.
        prototypes = torch.zeros((4, 4, 3))
        for index in range(4):
            prototypes[index, index, 0] = 1.
        priors = np.asarray([[.7, .1, .1, .1], [.1, .7, .1, .1]])
        cases = [{"class": "water"}, {"class": "food"}]
        generated = {}
        def first_pass(prefix, digest, case_index):
            generated[digest] = target_text(case_index)
            return generated[digest]
        first = hard_rows(initial, prototypes, priors, cases, [0, 1, 2, 3], first_pass)
        replay = hard_rows(initial, prototypes, priors, cases, [0, 1, 2, 3],
                           lambda prefix, digest, case_index: generated[digest])
        self.assertEqual(first, replay)
        self.assertEqual([row["trace"][0]["command"] for row in replay], [0, 1])


if __name__ == "__main__":
    unittest.main()
