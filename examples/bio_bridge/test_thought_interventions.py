import unittest

import numpy as np

from thought_interventions import (
    decoder_scores, intervene, random_l1_matched_intervention, selected_units,
)


class ThoughtInterventionTests(unittest.TestCase):
    def setUp(self):
        self.hidden = np.array([1.0, 0.5, 0.25, 0.0])
        self.saliency = np.array([[2., 0.], [1., 0.], [-1., 0.], [9., 0.]])

    def test_selection_uses_active_units_and_matches_count(self):
        top = selected_units(self.hidden, self.saliency, 0, "top_contribution", .5, 7)
        bottom = selected_units(self.hidden, self.saliency, 0, "bottom_contribution", .5, 7)
        random = selected_units(self.hidden, self.saliency, 0, "random_active", .5, 7)
        self.assertEqual(len(top), len(bottom))
        self.assertEqual(len(top), len(random))
        self.assertEqual(len(top), 2)
        self.assertNotIn(3, np.concatenate((top, bottom, random)))
        np.testing.assert_array_equal(top, [0, 1])
        np.testing.assert_array_equal(bottom, [2, 1])

    def test_intervention_does_not_renormalize(self):
        spec = {"strategy": "top_contribution", "fraction": .25}
        changed, removed, fraction = intervene(self.hidden, self.saliency, 0, spec, 9)
        self.assertEqual(len(removed), 1)
        self.assertEqual(changed[0], 0)
        self.assertEqual(changed[1], self.hidden[1])
        self.assertAlmostEqual(fraction, 1 / 1.75)

    def test_all_and_none_controls(self):
        none, removed_none, _ = intervene(self.hidden, self.saliency, 0, {"strategy": "none", "fraction": 0}, 1)
        all_zero, removed_all, fraction = intervene(self.hidden, self.saliency, 0, {"strategy": "all", "fraction": 1}, 1)
        np.testing.assert_array_equal(none, self.hidden)
        self.assertEqual(len(removed_none), 0)
        self.assertEqual(len(removed_all), 3)
        self.assertEqual(all_zero.sum(), 0)
        self.assertEqual(fraction, 1)

    def test_decoder_is_scale_invariant_except_zero(self):
        scores = decoder_scores(self.hidden[None, :], self.saliency)
        scaled = decoder_scores((10 * self.hidden)[None, :], self.saliency)
        np.testing.assert_allclose(scores, scaled)
        np.testing.assert_array_equal(decoder_scores(np.zeros((1, 4)), self.saliency), [[0., 0.]])

    def test_random_l1_match_is_exact_and_nonnegative(self):
        changed, indices, fully, partial, achieved = random_l1_matched_intervention(
            self.hidden, 0.37, 8321
        )
        self.assertAlmostEqual(achieved, 0.37, places=12)
        self.assertTrue(np.all(changed >= 0))
        self.assertTrue(np.all(changed <= self.hidden))
        self.assertEqual(len(indices), len(fully) + len(partial))
        self.assertLessEqual(len(partial), 1)
        removed = (np.abs(self.hidden).sum() - np.abs(changed).sum()) / np.abs(self.hidden).sum()
        self.assertAlmostEqual(removed, 0.37, places=12)


if __name__ == "__main__":
    unittest.main()
