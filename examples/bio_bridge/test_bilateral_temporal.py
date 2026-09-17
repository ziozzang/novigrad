import unittest
import numpy as np
import torch
from bilateral_temporal import episodes, select


class EpisodeTests(unittest.TestCase):
    def test_labels_never_change_packet(self):
        rng = np.random.default_rng(31)
        current = rng.normal(size=(8, 3, 32)).astype(np.float32)
        past = rng.normal(size=(16, 3, 32)).astype(np.float32)
        a, y, identity = episodes(current, np.arange(8) % 4, past, np.arange(16) % 4, 2, 4)
        b, _, _ = episodes(current, (np.arange(8) + 1) % 4, past, (np.arange(16) + 1) % 4, 2, 4)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        np.testing.assert_array_equal(y, np.repeat(np.arange(8) % 4, 2))
        self.assertEqual(len(identity['past_indices']), 16)
        self.assertTrue((a['timestamps'] <= a['observation_time'][:, None]).all())

    def test_current_frame_is_last_and_selection_preserves_metadata(self):
        current = np.full((4, 3, 32), 9., np.float32)
        past = np.zeros((8, 3, 32), np.float32)
        p, _, _ = episodes(current, np.arange(4), past, np.arange(8) % 4, 1, 5)
        self.assertTrue((p['features'][:, 6:] == 9).all())
        self.assertTrue((p['features'][:, :6] == 0).all())
        s = select(p, [3, 1])
        for k in p:
            self.assertTrue(torch.equal(s[k], p[k][[3, 1]]))


if __name__ == '__main__':
    unittest.main()
