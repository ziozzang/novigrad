import tempfile
import unittest
from pathlib import Path
import numpy as np
from precise_bridge import PortBridge

class PreciseBridgeTests(unittest.TestCase):
    def setUp(self):self.x=np.random.default_rng(12).normal(size=(8,768)).astype(np.float32)
    def test_fixed_matches_original_ports(self):
        z=self.x[:,:128].copy();z/=np.linalg.norm(z,axis=1,keepdims=True)
        expected=np.zeros((len(z),319),np.float32);expected[:,:128]=np.maximum(z,0);expected[:,128:256]=np.maximum(-z,0)
        np.testing.assert_allclose(PortBridge.fit(self.x,'fixed').encode(self.x),expected,atol=1e-7)
    def test_fitted_transform_is_batch_independent(self):
        for mode in ('centered','pca'):
            b=PortBridge.fit(self.x,mode)
            single=b.encode(self.x[:1]);batch=b.encode(np.vstack([self.x[:1],self.x[1:]*100]))[:1]
            np.testing.assert_allclose(single,batch,atol=1e-6)
    def test_pca_rank_and_nonnegative_budget(self):
        b=PortBridge.fit(self.x,'pca');p=b.encode(self.x)
        self.assertEqual(np.linalg.matrix_rank(b.projection),7)
        self.assertTrue(np.all(p>=0));self.assertTrue(np.all(p[:,256:]==0))
        np.testing.assert_allclose(np.linalg.norm(p,axis=1),1,atol=1e-6)
    def test_safetensors_exact_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            for mode in ('fixed','centered','pca'):
                b=PortBridge.fit(self.x,mode);p=Path(d)/'bridge.safetensors';b.save(p)
                np.testing.assert_array_equal(b.encode(self.x),PortBridge.load(p,mode).encode(self.x))
    def test_invalid_and_zero(self):
        with self.assertRaises(ValueError):PortBridge.fit(self.x,'bad')
        b=PortBridge.fit(self.x,'centered')
        with self.assertRaises(ValueError):b.encode(np.full((1,768),np.nan))
        self.assertTrue(np.all(b.encode(b.mean[None,:])==0))

if __name__=='__main__':unittest.main()
