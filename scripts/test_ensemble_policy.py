import unittest
import numpy as np
from ensemble_policy import vote


class EnsembleVoteTest(unittest.TestCase):
    def test_majority_over_confident_dissenter(self):
        p = [[.51, .49], [.51, .49], [0, 1]]
        majority, _ = vote(p)
        mean, _ = vote(p, 'mean')
        self.assertEqual(majority.argmax(), 0)
        self.assertEqual(mean.argmax(), 1)

    def test_tie_clones_and_invalid_probabilities(self):
        p = [[.6, .3, .1], [.2, .5, .3], [.1, .3, .6]]
        self.assertEqual(vote(p)[0].argmax(), 1)
        clone = np.array([.2, .3, .5])
        for mode in ('majority', 'mean'):
            self.assertEqual(vote([clone]*3, mode)[0].argmax(), clone.argmax())
        with self.assertRaises(ValueError):
            vote([[.5, .6], [.4, .6]])
