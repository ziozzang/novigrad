import tempfile
import unittest
from pathlib import Path
import numpy as np
from actuator_noise import sample_counts, summarize, run, COUNTS, SEEDS, REPLICATES

class CountingNoiseTests(unittest.TestCase):
    def test_predeclared_design(self):
        self.assertEqual(COUNTS, (32, 128, 512, 2048, 8192))
        self.assertEqual(SEEDS, (1901, 1902, 1903))
        self.assertEqual(REPLICATES, 256)

    def test_mass_support_and_exact_repeat(self):
        pn = np.array([[0., 2., 3.], [7., 0., 0.]])
        a = sample_counts(pn, 128, 64, 19)
        b = sample_counts(pn, 128, 64, 19)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        counts, inputs, zeros = a
        np.testing.assert_allclose(inputs.sum(1), np.repeat([5.,7.],64), rtol=1e-6)
        self.assertFalse(zeros.any())
        self.assertTrue(np.all(counts[:64,0] == 0))
        self.assertTrue(np.all(inputs[64:,1:] == 0))
        self.assertFalse(np.array_equal(counts, sample_counts(pn,128,64,20)[0]))

    def test_zero_observations_retained(self):
        counts, inputs, zeros = sample_counts([[1., 1.]], 1e-12, 20, 5)
        self.assertEqual(len(inputs),20)
        self.assertTrue(zeros.all())
        self.assertEqual(counts.sum(),0)
        self.assertEqual(inputs.sum(),0)

    def test_poisson_total_not_fixed_multinomial(self):
        counts, _, _ = sample_counts([[1.,3.]],32,10000,31)
        totals=counts.sum(1)
        self.assertAlmostEqual(totals.mean(),32,delta=.3)
        self.assertAlmostEqual(totals.var(),32,delta=2)
        self.assertAlmostEqual(counts[:,1].mean()/totals.mean(),.75,delta=.01)

    def test_two_agreements_and_signed_margin(self):
        p=np.array([[.1,.7,.1,.1],[.6,.2,.1,.1]])
        m=summarize(p,np.array([0,0]),np.array([1,0]))
        self.assertEqual(m['intended_agreement'],.5)
        self.assertEqual(m['unperturbed_action_agreement'],1.)
        self.assertAlmostEqual(m['intended_probability_margin']['min'],-.6)

    def test_invalid_inputs_and_no_overwrite(self):
        for pn in ([[0.,0.]], [[-1.,2.]], [[np.nan,1.]]):
            with self.assertRaises(ValueError): sample_counts(pn,32,2,1)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): run(Path(directory))

if __name__ == '__main__': unittest.main()
