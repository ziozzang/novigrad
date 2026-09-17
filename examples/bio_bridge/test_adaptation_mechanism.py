import unittest
import numpy as np
from adaptation_mechanism import adapt,build_episodes


class AdaptationTests(unittest.TestCase):
    def test_causal_subtraction_and_recovery(self):
        rows=np.ones((40,2),np.float32)
        out=adapt(rows,'subtractive_ema',np.ones(2))
        np.testing.assert_array_equal(out[0],rows[0])
        self.assertLess(out[-1].sum(),out[0].sum()*.01)
        pulse=rows.copy();pulse[20,1]+=1
        changed=adapt(pulse,'subtractive_ema',np.ones(2))
        np.testing.assert_array_equal(changed[:20],out[:20])
        self.assertGreater(changed[20,1],.99)

    def test_fixed_normalizer_history_free(self):
        rows=np.array([[1,0],[0,1]],np.float32)
        a=adapt(rows,'fixed_divisive',np.ones(2))
        b=adapt(rows[::-1],'fixed_divisive',np.ones(2))[::-1]
        np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):adapt(rows,'divisive_ema',np.ones(2),tau=0)

    def test_oracle_only_removes_background_and_masks(self):
        x=np.eye(4,dtype=np.float32);labels=np.arange(4)
        for schedule,n in [('pulse1',2),('sustained8',16)]:
            rows,bg,mask=build_episodes(x,labels,1.,schedule,True)
            self.assertEqual(mask.sum(),n)
            z=adapt(rows[0],'oracle_background',np.ones(4),bg[0])
            self.assertEqual(z[~mask].sum(),0)
            np.testing.assert_array_equal(z[mask],np.repeat(x[:1],n,axis=0))


if __name__=='__main__':unittest.main()
