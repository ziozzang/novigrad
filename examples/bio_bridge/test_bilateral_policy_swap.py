import tempfile,unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from bilateral_policy_swap import EXPECTED_KEYS,verify_policy_slot

def tensors():
 return {'input_ids':np.array([1,2],np.uint64),'hidden_ids':np.array([3,4],np.uint64),'output_ids':np.array([5],np.uint64),'input_pre':np.array([0,1],np.uint64),'input_post':np.array([0,1],np.uint64),'input_count':np.array([1,1],np.uint64),'input_sign':np.array([1,-1],np.int8),'plastic_pre':np.array([0,1],np.uint64),'plastic_post':np.array([0,0],np.uint64),'plastic_count':np.array([1,1],np.uint64),'plastic_sign':np.array([1,1],np.int8),'output_actions':np.array([0],np.uint64),'output_gains':np.array([1.],np.float32),'plastic_weight':np.array([.1,.2],np.float32)}
META={'format':'nobi.plastic','version':'4','actions':'4','active_fraction':'0.02','logit_gain':'1','homeostasis':'false','weight_limit':'1'}
class PolicySwapTests(unittest.TestCase):
 def pair(self,d,mutate):
  a,b=tensors(),tensors();mutate(b);p=Path(d)/'a.safe';q=Path(d)/'b.safe';save_file(a,str(p),metadata=META);save_file(b,str(q),metadata=META);return p,q
 def test_only_plastic_weights_may_change(self):
  with tempfile.TemporaryDirectory() as d:
   p,q=self.pair(d,lambda b:b.__setitem__('plastic_weight',np.array([.3,.4],np.float32)));r=verify_policy_slot(p,q);self.assertTrue(r['all_frozen_tensors_exact']);self.assertEqual(r['plastic_weight_changed_count'],2)
 def test_input_topology_or_order_change_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,q=self.pair(d,lambda b:b.__setitem__('input_ids',b['input_ids'][::-1].copy()))
   with self.assertRaises(ValueError):verify_policy_slot(p,q)
 def test_nonweight_plastic_slot_change_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,q=self.pair(d,lambda b:b.__setitem__('plastic_pre',b['plastic_pre'][::-1].copy()))
   with self.assertRaises(ValueError):verify_policy_slot(p,q)
 def test_identical_policy_is_not_a_swap(self):
  with tempfile.TemporaryDirectory() as d:
   p,q=self.pair(d,lambda b:None)
   with self.assertRaises(ValueError):verify_policy_slot(p,q)
if __name__=='__main__':unittest.main()
