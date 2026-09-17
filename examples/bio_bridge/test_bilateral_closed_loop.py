import types,unittest
import numpy as np
import torch
from bilateral_closed_loop import run_loop_components
from bilateral_lm import target_text

class Adapter(torch.nn.Module):
 def forward(self,packet):
  out=torch.zeros((len(packet),4,8));out[:,0,0]=packet.features[:,0,0];return out
class Bridge:
 def encode(self,x):return np.asarray(x[:,:2],np.float32)
class Shadow:
 def forward(self,pn,*args):
  p=np.array([[.7,.1,.1,.1]],np.float32);h=np.array([[1.,0.,0.]])
  if pn[0,0]>=15:p=np.array([[.1,.7,.1,.1]],np.float32)
  return p,h
class Codec:
 def __init__(self):self.bridge=Bridge();self.shadow=Shadow()
 def encode(self,x,source='circuit'):
  out=np.zeros((len(x),3,32),np.float32);out[:,:,0]=np.asarray(x)[:,0,None];return out
class Engine:
 def __init__(self):self.calls=0
 def infer_batch(self,pn):self.calls+=1;return Shadow().forward(np.asarray(pn))[0].tolist()
class Runtime:
 device=torch.device('cpu')
 def __init__(self,mapping):self.mapping=mapping;self.calls=[]
 def generate(self,prefix):
  marker=int(round(float(prefix[0,0])));self.calls.append(marker);value=self.mapping.get(marker,'invalid')
  return '<not-a-call>' if value=='invalid' else target_text(('water','food','warmth','rest').index(value))

def fixtures(marker):
 x=np.zeros((1,768),np.float32);x[0,0]=marker;y=np.array([0],np.int64);prototypes=np.zeros((4,768),np.float32);prototypes[:,0]=[10,20,30,40]
 return x,y,prototypes
class ClosedLoopTests(unittest.TestCase):
 def test_valid_call_writes_and_uses_actual_generated_goal(self):
  x,y,p=fixtures(0);engine=Engine();runtime=Runtime({0:'water',10:'water'})
  report=run_loop_components(Adapter(),Codec(),engine,p,x,y,[0],runtime)
  case=report['cases'][0];self.assertTrue(case['state_write']['accepted']);self.assertEqual(case['state_write']['prototype_goal'],'water');self.assertTrue(case['dynamics']['correct_stable']);self.assertEqual(engine.calls,1);self.assertLessEqual(case['native']['max_abs_parity_error'],2e-6)
 def test_invalid_call_rejects_without_action_or_state_change(self):
  x,y,p=fixtures(99);engine=Engine();runtime=Runtime({99:'invalid'})
  case=run_loop_components(Adapter(),Codec(),engine,p,x,y,[0],runtime)['cases'][0]
  self.assertFalse(case['state_write']['accepted']);self.assertIsNone(case['native']);self.assertEqual(engine.calls,0);self.assertEqual(runtime.calls,[99,99])
 def test_wrong_first_goal_can_be_rescued_but_is_not_teacher_corrected(self):
  x,y,p=fixtures(2);engine=Engine();runtime=Runtime({2:'food',20:'water'})
  case=run_loop_components(Adapter(),Codec(),engine,p,x,y,[0],runtime)['cases'][0]
  self.assertEqual(case['tick1']['parsed_goal'],'food');self.assertEqual(case['state_write']['prototype_goal'],'food');self.assertTrue(case['dynamics']['rescue']);self.assertFalse(case['dynamics']['wrong_fixed_point'])
 def test_validation(self):
  x,y,p=fixtures(0)
  with self.assertRaises(ValueError):run_loop_components(Adapter(),Codec(),Engine(),p,x,y.astype(float),[0],Runtime({}))
  with self.assertRaises(ValueError):run_loop_components(Adapter(),Codec(),Engine(),p,x,y,[0,0],Runtime({}))
if __name__=='__main__':unittest.main()
