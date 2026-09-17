import tempfile,types,unittest
from pathlib import Path
import numpy as np
import torch
from bilateral_lm import (GOALS,Runtime,SET_GOAL_TOOL,_init_adapter,evaluate,
 load_adapter,load_train,parse_goal,save_adapter,target_text,train_adapter)

class TinyTokenizer:
 pad_token_id=0
 def __init__(self):self.last=None
 def apply_chat_template(self,messages,tools,add_generation_prompt,tokenize):
  self.last=(messages,tools,add_generation_prompt,tokenize);return [1,2,3]
 def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
 def decode(self,ids,skip_special_tokens=False):return ''.join(chr(int(i)) for i in ids)
class TinyLM(torch.nn.Module):
 def __init__(self,generated):
  super().__init__();self.config=types.SimpleNamespace(hidden_size=8);self.emb=torch.nn.Embedding(256,8);self.head=torch.nn.Linear(8,256);self.generated=torch.tensor([[ord(c) for c in generated]])
 def get_input_embeddings(self):return self.emb
 def forward(self,inputs_embeds,attention_mask,use_cache,logits_to_keep):
  return types.SimpleNamespace(logits=self.head(inputs_embeds[:,-logits_to_keep:]))
 def generate(self,**kwargs):return self.generated.to(kwargs['inputs_embeds'].device)

def packet(n=4):
 f=np.arange(n*3*32,dtype=np.float32).reshape(n,3,32)/100;m=np.array([[1,1,0]]*n,bool);sites=np.array([[0,1,0]]*n,np.int64);times=np.array([[-1.,0.,0.]]*n,np.float32);return load_train(f,m,sites,times,np.arange(n)%4)

class BilateralLMTests(unittest.TestCase):
 def runtime(self,goal='water'):
  tok=TinyTokenizer();raw=target_text(GOALS.index(goal));return Runtime(TinyLM(raw),tok,'cpu'),tok
 def test_packet_validation(self):
  p=packet();self.assertEqual(p.features.shape,(4,3,32))
  with self.assertRaises(ValueError):load_train(p.features,p.valid_mask.float(),p.site_ids,p.times,p.targets)
  with self.assertRaises(ValueError):load_train(p.features,p.valid_mask,p.site_ids+9,p.times,p.targets)
  with self.assertRaises(ValueError):load_train(p.features,p.valid_mask,p.site_ids,p.times,p.targets.float())
  with self.assertRaises(ValueError):load_train(p.features,p.valid_mask,p.site_ids,p.times+1,p.targets)
 def test_constant_prompt_and_set_goal_only(self):
  runtime,tok=self.runtime();messages,tools,_,_=tok.last
  self.assertEqual(len(tools),1);self.assertEqual(tools[0]['function']['name'],'set_goal');self.assertNotIn('water',messages[-1]['content']);self.assertEqual(parse_goal(target_text(2)),'warmth')
 def test_generation_decodes_entire_continuation(self):
  runtime,_=self.runtime('food');raw=runtime.generate(torch.zeros(4,8));self.assertEqual(raw,target_text(1));self.assertEqual(parse_goal(raw),'food')
 def test_batched_likelihood_matches_single(self):
  runtime,_=self.runtime();prefix=torch.randn(4,8);single=runtime.loglikelihood(prefix,runtime.response_ids[0]);batch=runtime.loglikelihood_batch(prefix.unsqueeze(0).expand(2,-1,-1),runtime.response_ids[0]);torch.testing.assert_close(batch, single.expand(2),rtol=0,atol=1e-6)
 def test_shared_projection_initialization(self):
  a=_init_adapter('learned','cpu',8);b=_init_adapter('fixed','cpu',8);torch.testing.assert_close(a.output.weight,b.output.weight,rtol=0,atol=0);torch.testing.assert_close(a.output.bias,b.output.bias,rtol=0,atol=0)
 def test_save_load_exact_prefix(self):
  p=packet();a=_init_adapter('learned','cpu',8).eval()
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'a.safetensors';save_adapter(a,path,{'test':True});b,meta=load_adapter(path)
   self.assertEqual(meta,{'test':True});torch.testing.assert_close(a(p),b(p),rtol=0,atol=0)
 def test_mock_train_and_controls(self):
  p=packet();runtime,_=self.runtime('water')
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'a.safetensors';report=train_adapter(path,'pooled_mlp',p,p.targets,steps=1,runtime=runtime,batch_size=2)
   self.assertTrue(report['base_all_parameters_frozen']);result=evaluate(path,p,p.targets,generation_indices=[0],runtime=runtime)
   self.assertEqual(result['actual']['generated_count'],1);self.assertEqual(len(result['actual']['rows'][0]['candidate_probabilities']),4);self.assertEqual(set(('actual','zero_prefix','row_shuffled_prefix'))-set(result),set())
   zero=[row['candidate_loglikelihoods'] for row in result['zero_prefix']['rows']];self.assertTrue(all(row==zero[0] for row in zero))
   path.write_bytes(path.read_bytes()+b'tamper')
   with self.assertRaises(ValueError):evaluate(path,p,p.targets,generation_indices=[],runtime=runtime)
if __name__=='__main__':unittest.main()
