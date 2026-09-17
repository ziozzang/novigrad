import unittest
import tempfile
from pathlib import Path
import numpy as np
from novigrad import Engine
from signal_recall import trace, sequences


class SignalRecallTests(unittest.TestCase):
    def test_direct_forgets_hold_persists_and_episode_resets(self):
        rows = np.zeros((12, 3), np.float32); rows[0, 1] = 1
        self.assertEqual(trace(rows, "direct")[1:].sum(), 0)
        np.testing.assert_array_equal(trace(rows, "hold")[:, 1], np.ones(12))
        self.assertEqual(trace(np.zeros_like(rows), "hold").sum(), 0)

    def test_ttl_expires_and_new_input_overwrites(self):
        rows = np.zeros((12, 3), np.float32); rows[0, 1] = 1; rows[10, 2] = 1
        out = trace(rows, "ttl8")
        self.assertEqual(out[7, 1], 1)
        self.assertEqual(out[8:10].sum(), 0)
        np.testing.assert_array_equal(out[11], [0, 0, 1])

    def test_cutoff_gates_emission_but_does_not_erase_accumulation(self):
        rows = np.zeros((3, 2), np.float32); rows[:, 0] = .03
        out = trace(rows, "leaky16_cutoff")
        self.assertEqual(out[0].sum(), 0)
        self.assertGreater(out[1, 0], .05)
        for invalid in [np.array([[np.nan]]), np.array([[-1.]])]:
            with self.assertRaises(ValueError): trace(invalid, "direct")

    def test_equal_total_schedule_dose_and_switch(self):
        x = np.eye(4, dtype=np.float32); labels = np.arange(4)
        for schedule in ["continuous", "one_shot", "periodic4"]:
            rows, targets, _, pulse_mask = sequences(x, labels, schedule, "equal_total", 0)
            np.testing.assert_allclose(rows[:, :12].sum((1,2)), 1, atol=1e-6)
            np.testing.assert_allclose(rows[:, 12:].sum((1,2)), 1, atol=1e-6)
            np.testing.assert_array_equal(targets[:, 12], (labels+1)%4)
            self.assertEqual(int(pulse_mask.sum()), {"continuous":24,"one_shot":2,"periodic4":6}[schedule])

    def test_background_refreshes_hold_and_invalid_schedule_rejected(self):
        rows = np.zeros((12, 2), np.float32); rows[0, 0]=1; rows[1:,1]=.001
        for mode in ["hold", "ttl8"]:
            np.testing.assert_array_equal(trace(rows, mode)[-1], rows[-1])
        for dose, background in [("typo",0),("equal_total",-1),("per_pulse",float('nan'))]:
            with self.assertRaises(ValueError):
                sequences(np.eye(4),np.arange(4),"one_shot",dose,background)

    def test_native_scale_invariance_and_no_inference_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'input.tsv').write_text('1\t11\t1\t1\n2\t12\t1\t1\n')
            (root/'plastic.tsv').write_text('11\t21\t1\t1\n12\t22\t1\t1\n')
            engine = Engine.from_edges(root/'input.tsv', root/'plastic.tsv', actions=2,
                                       active_fraction=1., homeostasis=False)
            before = engine.weights
            reference = engine.infer([1., .1])
            for gain in [1e-6, .1, 1., 100., 1e6]:
                np.testing.assert_allclose(engine.infer([gain, .1*gain]), reference, atol=1e-7)
            np.testing.assert_array_equal(engine.infer([0., 0.]), [.5, .5])
            np.testing.assert_array_equal(engine.infer([1., .1]), reference)
            self.assertEqual(before, engine.weights)


if __name__ == "__main__":
    unittest.main()
