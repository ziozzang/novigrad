import tempfile
import unittest
from pathlib import Path
import numpy as np
from image_adapter import ImageRateAdapter


class AdapterTests(unittest.TestCase):
    def test_fit_roundtrip_and_batch_stability(self):
        rng=np.random.default_rng(123)
        train=rng.integers(0,256,(32,28,28),dtype=np.uint8)
        unseen=rng.integers(0,256,(4,28,28),dtype=np.uint8)
        adapter=ImageRateAdapter.fit(train,np.arange(1,20,dtype=np.uint64),components=8,whitening=.5)
        before=adapter.projection.copy()
        whole=adapter.transform(unseen)
        np.testing.assert_allclose(whole,np.concatenate([adapter.transform(unseen[i:i+1]) for i in range(4)]),atol=1e-6)
        self.assertTrue(np.isfinite(whole).all())
        self.assertTrue(((whole>=0)&(whole<=1)).all())
        np.testing.assert_array_equal(adapter.projection,before)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'adapter.safetensors'
            adapter.save(path)
            np.testing.assert_array_equal(whole,ImageRateAdapter.load(path).transform(unseen))

    def test_bad_image_shape_rejected(self):
        with self.assertRaises(ValueError):
            ImageRateAdapter.features(np.zeros((2,28,29),dtype=np.uint8))
        with self.assertRaises(ValueError):
            ImageRateAdapter.features(np.zeros((2,28,28),dtype=np.float32))


if __name__=='__main__':
    unittest.main()
