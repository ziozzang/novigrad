import unittest

import numpy as np

try:
    from driving_environment import encode_observation
    from train_driving_game import advantages
except ImportError:
    encode_observation = None


@unittest.skipIf(encode_observation is None, 'optional simulation dependencies are not installed')
class DrivingLearningTest(unittest.TestCase):
    def test_return_credit_and_terminal_padding(self):
        advantage, returns = advantages([1., 0., 1.], np.zeros(40))
        np.testing.assert_allclose(returns[:3], [1 + .97 ** 2, .97, 1.])
        np.testing.assert_array_equal(returns[3:], np.zeros(37))
        np.testing.assert_allclose(advantage, returns[:3] / 10)
        negative, _ = advantages([0.], np.ones(40))
        self.assertAlmostEqual(negative[0], -.1)

    def test_observation_signs_and_invalid_input(self):
        observation = np.zeros((5, 5), dtype=np.float32)
        observation[0, 0] = -.5
        rates = encode_observation(observation, 319)
        self.assertEqual(rates[0], 0.)
        self.assertEqual(rates[25], .5)
        self.assertEqual(rates[-1], 1.)
        self.assertTrue(np.all((rates >= 0) & (rates <= 1)))
        observation[0, 0] = np.nan
        with self.assertRaises(ValueError):
            encode_observation(observation, 319)
