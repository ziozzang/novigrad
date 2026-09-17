import unittest
import numpy as np
from bridge import SemanticPorts,LinearPolicy

class BridgeTests(unittest.TestCase):
    def test_signed_ports(self):
        x=np.array([[1.,-2.,3.]])
        y=SemanticPorts(8,3).encode(x)
        np.testing.assert_allclose(y[:,:3]-y[:,3:6], x/np.linalg.norm(x),rtol=1e-6)
        self.assertTrue((y>=0).all()); self.assertTrue((y[:,6:]==0).all())
        self.assertTrue((SemanticPorts(8,3).encode(np.zeros((1,3)))==0).all())
        with self.assertRaises(ValueError):SemanticPorts(3,2)
        with self.assertRaises(ValueError):SemanticPorts(8,3).encode([[float('nan'),0,0]])
    def test_reward_gradient(self):
        p=LinearPolicy(2,.1); x=np.array([[1.,0.]],np.float32)
        p.learn(x,[2],[1.]); self.assertGreater(p.predict(x)[0,2],.25)
        p.learn(x,[2],[-1.]); self.assertLess(p.predict(x)[0,2],.27)

if __name__=='__main__':unittest.main()
