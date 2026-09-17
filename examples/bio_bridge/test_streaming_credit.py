import unittest

from streaming_credit import BoundedDelayQueue


class BoundedDelayQueueTests(unittest.TestCase):
    def test_storage_bound_and_event_order(self):
        queue = BoundedDelayQueue(3)
        delivered = []
        for logical_time in range(8):
            queue.schedule(logical_time, str(logical_time))
            delivered.extend(queue.pop_due(logical_time))
        for logical_time in range(8, 11):
            delivered.extend(queue.pop_due(logical_time))
        self.assertEqual([trial_id for trial_id, _ in delivered], list(range(8)))
        self.assertLessEqual(queue.max_pending, 4)
        self.assertFalse(queue.pending)

    def test_duplicate_stale_and_nonmonotonic_rejected_without_history(self):
        queue = BoundedDelayQueue(2)
        queue.schedule(0, "zero")
        with self.assertRaises(ValueError):
            queue.schedule(0, "duplicate")
        with self.assertRaises(ValueError):
            queue.schedule(2, "gap")
        queue.schedule(1, "one")
        queue.pop_due(3)
        self.assertEqual(queue.next_id, 2)
        self.assertFalse(queue.pending)

    def test_capacity_failure_is_atomic_and_retryable(self):
        queue = BoundedDelayQueue(2)
        for trial_id in range(3):
            queue.schedule(trial_id, trial_id)
        with self.assertRaises(OverflowError):
            queue.schedule(3, "overflow")
        self.assertEqual(queue.next_id, 3)
        self.assertEqual(len(queue.pending), 3)
        queue.pop_due(2)
        queue.schedule(3, "retry")
        self.assertEqual(queue.next_id, 4)

    def test_integer_type_validation(self):
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                BoundedDelayQueue(invalid)
        queue = BoundedDelayQueue(1)
        for invalid in (-1, True, 0.0):
            with self.assertRaises(ValueError):
                queue.schedule(invalid, "x")
            with self.assertRaises(ValueError):
                queue.pop_due(invalid)


if __name__ == "__main__":
    unittest.main()
