import unittest

from delayed_credit import TaggedDelayQueue, transformed_reward


class TaggedDelayQueueTests(unittest.TestCase):
    def test_fifo_delay_duplicate_and_stale_ids(self):
        queue = TaggedDelayQueue(4)
        queue.schedule(10, "a")
        queue.schedule(11, "b")
        self.assertEqual(queue.pop_due(13), [])
        self.assertEqual(queue.pop_due(14), [(10, "a")])
        self.assertEqual(queue.pop_due(15), [(11, "b")])
        self.assertEqual(queue.pop_due(99), [])  # cannot double-deliver
        with self.assertRaises(ValueError):
            queue.schedule(10, "duplicate")
        with self.assertRaises(KeyError):
            queue.claim(10)

    def test_claim_removes_exact_pending_id(self):
        queue = TaggedDelayQueue(16)
        queue.schedule(1, "first")
        queue.schedule(2, "second")
        self.assertEqual(queue.claim(2), "second")
        self.assertEqual(queue.pop_due(17), [(1, "first")])

    def test_immediate_amplitude_control_matches_decay_constant(self):
        scale = transformed_reward("amplitude_control", 1.0, 0, reward_scale=__import__("math").exp(-.5))
        decayed = transformed_reward("decayed_queued", 1.0, 4)
        self.assertAlmostEqual(scale, decayed)


if __name__ == "__main__":
    unittest.main()
