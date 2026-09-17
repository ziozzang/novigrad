#!/usr/bin/env python3
"""Frozen FunctionGemma conditioned by simulated site/time feature soft prefixes.

Only the resampler and 32->640 projection are trainable. This is an engineered
language-model interface, not a neural recording decoder or biological model.
"""
from __future__ import annotations
import hashlib,json,time,sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from neural_resampler import MODES,NeuralResampler
FUNCTION_DIR=Path(__file__).resolve().parents[1]/'function_bridge'
if str(FUNCTION_DIR) not in sys.path:sys.path.insert(0,str(FUNCTION_DIR))
from protocol import tools as protocol_tools

MODEL_PATH=Path('/Users/a405394/models/google_functiongemma-270m-it');SEED=1701
GOALS=('water','food','warmth','rest');DIM=32;LATENTS=4;HIDDEN=640;HEADS=4;SITES=4
CONSTANT_USER='Use the latent state to set the current goal.'
BASE='You are a model that can do function calling with the following functions'
SET_GOAL_TOOL=[tool for tool in protocol_tools() if tool['function']['name']=='set_goal']

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def target_text(label):
 goal=GOALS[int(label)];return f'<start_function_call>call:set_goal{{goal:<escape>{goal}<escape>}}<end_function_call><start_function_response>'
def parse_goal(raw):
 from protocol import parse
 try:name,args=parse(raw);return args['goal'] if name=='set_goal' else None
 except (ValueError,KeyError):return None

@dataclass(frozen=True)
class Packet:
 features:torch.Tensor;valid_mask:torch.Tensor;site_ids:torch.Tensor;times:torch.Tensor;targets:torch.Tensor
 def __len__(self):return len(self.features)
 def take(self,index):return Packet(*(x[index] for x in (self.features,self.valid_mask,self.site_ids,self.times,self.targets)))

def load_train(features,mask,site_ids,times,targets):
 f=torch.as_tensor(features);m=torch.as_tensor(mask);s=torch.as_tensor(site_ids);t=torch.as_tensor(times);y=torch.as_tensor(targets)
 if f.ndim!=3 or f.shape[2]!=DIM or not torch.is_floating_point(f) or not torch.isfinite(f).all():raise ValueError('features must be finite floating [B,T,32]')
 shape=f.shape[:2]
 if m.shape!=shape or m.dtype!=torch.bool or not m.any(1).all():raise ValueError('mask must be bool [B,T] with one valid token per row')
 if s.shape!=shape or s.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8) or (s[m]<0).any() or (s[m]>=SITES).any():raise ValueError('site_ids must be integer [B,T] in [0,4) on valid tokens')
 if t.shape!=shape or not torch.is_floating_point(t) or not torch.isfinite(t).all() or (t>0).any():raise ValueError('times must be finite floating [B,T] at or before fixed observation time 0')
 if y.ndim!=1 or len(y)!=len(f) or y.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8) or (y<0).any() or (y>=len(GOALS)).any():raise ValueError('targets must be integer class IDs [B]')
 return Packet(f.float().contiguous(),m.contiguous(),s.long().contiguous(),t.float().contiguous(),y.long().contiguous())

class SoftPrefix(nn.Module):
 def __init__(self,mode,hidden=HIDDEN):
  super().__init__();self.mode=mode;self.hidden=hidden;self.resampler=NeuralResampler(mode=mode,dim=DIM,latents=LATENTS,heads=HEADS,sites=SITES);self.output=nn.Linear(DIM,hidden)
 def forward(self,packet):
  obs=torch.zeros(len(packet),dtype=packet.times.dtype,device=packet.times.device)
  z=self.resampler(packet.features,packet.valid_mask,packet.site_ids,packet.times,obs);return self.output(z)
 def config(self):return {'mode':self.mode,'dim':DIM,'latents':LATENTS,'heads':HEADS,'sites':SITES,'hidden':self.hidden}

class Runtime:
 _cache={}
 def __init__(self,model,tokenizer,device,model_path=None):
  self.model,self.tokenizer,self.device,self.model_path=model,tokenizer,torch.device(device),None if model_path is None else Path(model_path)
  self.model.eval()
  for p in self.model.parameters():p.requires_grad_(False)
  prompt=tokenizer.apply_chat_template([{'role':'developer','content':BASE},{'role':'user','content':CONSTANT_USER}],tools=SET_GOAL_TOOL,add_generation_prompt=True,tokenize=True)
  if hasattr(prompt,'keys') and 'input_ids' in prompt:prompt=prompt['input_ids']
  if isinstance(prompt,str):prompt=tokenizer.encode(prompt,add_special_tokens=False)
  if isinstance(prompt,torch.Tensor):prompt=prompt.flatten().tolist()
  self.prompt_ids=torch.tensor(prompt,dtype=torch.long,device=self.device)
  self.response_ids=[torch.tensor(tokenizer.encode(target_text(i),add_special_tokens=False),dtype=torch.long,device=self.device) for i in range(4)]
  if any(len(x)==0 for x in self.response_ids):raise ValueError('empty target serialization')
 @classmethod
 def load(cls,model_path=MODEL_PATH,device='mps'):
  key=(str(Path(model_path).resolve()),device)
  if key not in cls._cache:
   from transformers import AutoModelForCausalLM,AutoTokenizer
   tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.bfloat16).to(device)
   cls._cache[key]=cls(model,tok,device,model_path)
  return cls._cache[key]
 @property
 def checkpoint_sha256(self):return sha256(self.model_path/'model.safetensors') if self.model_path else None
 def memory_fingerprint(self):
  h=hashlib.sha256()
  for name,value in self.model.state_dict().items():
   h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 def loglikelihood(self,prefix,response_ids):
  return self.loglikelihood_batch(prefix.unsqueeze(0),response_ids)[0]
 def loglikelihood_batch(self,prefixes,response_ids):
  if prefixes.ndim!=3 or prefixes.shape[2]!=getattr(self.model.config,'hidden_size',prefixes.shape[2]) or not torch.isfinite(prefixes).all():raise ValueError('prefixes must be finite [B,K,H]')
  ids=torch.cat((self.prompt_ids,response_ids));inputs=ids[:-1];token_emb=self.model.get_input_embeddings()(inputs.unsqueeze(0)).detach().expand(len(prefixes),-1,-1);joined=torch.cat((prefixes.to(token_emb.dtype),token_emb),1);mask=torch.ones(joined.shape[:2],dtype=torch.long,device=self.device)
  kwargs={'inputs_embeds':joined,'attention_mask':mask,'use_cache':False,'logits_to_keep':len(response_ids)}
  out=self.model(**kwargs);logits=out.logits[:,-len(response_ids):].float();targets=response_ids[None,:].expand(len(prefixes),-1);return torch.log_softmax(logits,2).gather(2,targets.unsqueeze(-1)).squeeze(-1).sum(1)
 def generate(self,prefix,max_new_tokens=48):
  token_emb=self.model.get_input_embeddings()(self.prompt_ids.unsqueeze(0)).detach();joined=torch.cat((prefix.unsqueeze(0).to(token_emb.dtype),token_emb),1);mask=torch.ones(joined.shape[:2],dtype=torch.long,device=self.device)
  with torch.inference_mode():out=self.model.generate(inputs_embeds=joined,attention_mask=mask,do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=self.tokenizer.pad_token_id)
  # inputs_embeds generation returns continuation-only IDs in the supported runtime.
  return self.tokenizer.decode(out[0],skip_special_tokens=False)

def _to(packet,device):return Packet(*(x.to(device) for x in (packet.features,packet.valid_mask,packet.site_ids,packet.times,packet.targets)))
def _packet_sha(packet):
 h=hashlib.sha256()
 for x in (packet.features,packet.valid_mask,packet.site_ids,packet.times,packet.targets):h.update(x.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def _init_adapter(mode,device,hidden=HIDDEN):
 # Re-seeding makes the shared output projection identical across modes.
 torch.manual_seed(SEED);model=SoftPrefix(mode,hidden);torch.manual_seed(SEED+99);model.output.reset_parameters();return model.to(device)

def save_adapter(adapter,path,metadata):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);state={k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()}
 if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('nonfinite adapter state')
 meta={'format':'novigrad.bilateral-soft-prefix','version':'1','config':json.dumps(adapter.config(),sort_keys=True,separators=(',',':')),'manifest':json.dumps(metadata,sort_keys=True,separators=(',',':'))};save_file(state,str(path),metadata=meta)
def load_adapter(path,device='cpu'):
 sidecar=Path(str(path)+'.manifest.json')
 if sidecar.exists():
  record=json.loads(sidecar.read_text())
  if record.get('adapter_sha256')!=sha256(path):raise ValueError('adapter hash does not match manifest')
 with safe_open(str(path),framework='pt',device='cpu') as f:meta=f.metadata() or {}
 if meta.get('format')!='novigrad.bilateral-soft-prefix' or meta.get('version')!='1':raise ValueError('invalid soft-prefix metadata')
 try:config=json.loads(meta['config']);manifest=json.loads(meta['manifest']);model=SoftPrefix(config['mode'],config['hidden'])
 except (KeyError,ValueError,TypeError,json.JSONDecodeError) as e:raise ValueError('invalid soft-prefix config') from e
 if config!=model.config():raise ValueError('unsupported soft-prefix config')
 state=load_file(str(path),device='cpu')
 if set(state)!=set(model.state_dict()):raise ValueError('invalid soft-prefix tensor schema')
 for k,v in state.items():
  if v.shape!=model.state_dict()[k].shape or v.dtype!=model.state_dict()[k].dtype or not torch.isfinite(v).all():raise ValueError('invalid soft-prefix tensor')
 model.load_state_dict(state,strict=True);return model.to(device),manifest

def train_adapter(out,mode,trainpacket,labels,steps=120,runtime=None,batch_size=4,learning_rate=.003):
 if mode not in MODES or steps<=0 or batch_size<=0 or not np.isfinite(learning_rate) or learning_rate<=0:raise ValueError('invalid training configuration')
 artifact=Path(out);sidecar=Path(str(artifact)+'.manifest.json')
 if artifact.exists() or sidecar.exists():raise FileExistsError('refusing to overwrite adapter or manifest')
 labels=torch.as_tensor(labels)
 if labels.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8) or not torch.equal(labels.long().cpu(),trainpacket.targets.cpu()):raise ValueError('labels must exactly match packet integer targets')
 runtime=runtime or Runtime.load();packet=_to(trainpacket,runtime.device);adapter=_init_adapter(mode,runtime.device,getattr(runtime.model.config,'hidden_size',HIDDEN));adapter.train();optimizer=torch.optim.AdamW(adapter.parameters(),lr=learning_rate,weight_decay=0.0);rng=np.random.default_rng(SEED+1);stream=rng.integers(0,len(packet),size=(steps,batch_size));losses=[];start=time.perf_counter();initial_output_sha=hashlib.sha256(adapter.output.weight.detach().cpu().numpy().tobytes()+adapter.output.bias.detach().cpu().numpy().tobytes()).hexdigest();initial_resampler_sha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in adapter.resampler.state_dict().values())).hexdigest()
 with torch.inference_mode():
  initial_prefix=adapter(packet);prompt_embedding=runtime.model.get_input_embeddings()(runtime.prompt_ids)
  initial_prefix_rms=float(initial_prefix.float().square().mean().sqrt().cpu());prompt_embedding_rms=float(prompt_embedding.float().square().mean().sqrt().cpu())
 checkpoint_before=runtime.checkpoint_sha256;assert all(not p.requires_grad for p in runtime.model.parameters())
 memory_before=runtime.memory_fingerprint()
 for step,chosen in enumerate(stream):
  ix=torch.tensor(chosen,device=runtime.device);batch=packet.take(ix);prefix=adapter(batch);loss=prefix.sum()*0
  for target in batch.targets.unique():
   selected=batch.targets==target;loss-=runtime.loglikelihood_batch(prefix[selected],runtime.response_ids[int(target)]).sum()/batch_size
  optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
  if step%60==0:print('soft-prefix',mode,step,losses[-1],flush=True)
 artifact=Path(out);manifest={'mode':mode,'seed':SEED,'steps':steps,'batch_size':batch_size,'learning_rate':learning_rate,'optimizer':'AdamW','weight_decay':0.0,'gradient_clip':1.0,'train_packet_sha256':_packet_sha(trainpacket),'paired_stream_sha256':hashlib.sha256(stream.tobytes()).hexdigest(),'initial_shared_output_sha256':initial_output_sha,'initial_resampler_sha256':initial_resampler_sha,'initial_prefix_rms':initial_prefix_rms,'constant_prompt_token_embedding_rms':prompt_embedding_rms,'prefix_output_scale_policy':'shared seeded Linear(32,H) initialization across modes; no tuned or final-dependent scaling','trainable_parameters':sum(p.numel() for p in adapter.parameters() if p.requires_grad),'base_checkpoint_sha256_before':checkpoint_before,'base_checkpoint_sha256_after':runtime.checkpoint_sha256,'base_all_parameters_frozen':all(not p.requires_grad for p in runtime.model.parameters()),'source_sha256':sha256(__file__),'constant_user':CONSTANT_USER,'tool_schema':SET_GOAL_TOOL,'loss_scope':'teacher-forced response tokens only; prompt tokens excluded','losses':losses,'seconds':time.perf_counter()-start,'boundary':'soft-prefix adapter only; frozen FunctionGemma; no biological or thought-decoding claim'}
 memory_after=runtime.memory_fingerprint()
 if memory_before!=memory_after or any(p.grad is not None for p in runtime.model.parameters()):raise AssertionError('frozen base memory or gradients changed')
 manifest.update(base_memory_sha256_before=memory_before,base_memory_sha256_after=memory_after,base_gradients_absent=True,response_token_counts=[len(x) for x in runtime.response_ids])
 save_adapter(adapter,artifact,manifest);manifest['adapter_sha256']=sha256(artifact);Path(str(artifact)+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def _likelihood_matrix(runtime,prefixes,batch_size=8):
 columns=[]
 with torch.inference_mode():
  for response in runtime.response_ids:
   columns.append(torch.cat([runtime.loglikelihood_batch(prefixes[i:i+batch_size],response) for i in range(0,len(prefixes),batch_size)]))
 return torch.stack(columns,1)
def _condition_report(runtime,prefixes,labels,generation_indices,generation_cache=None,source_indices=None,identical_prefix=False):
 start=time.perf_counter();source_indices=np.arange(len(prefixes)) if source_indices is None else np.asarray(source_indices);generation_cache={} if generation_cache is None else generation_cache
 scores=_likelihood_matrix(runtime,prefixes[:1] if identical_prefix else prefixes);scores=scores.expand(len(prefixes),-1) if identical_prefix else scores
 rows=[]
 with torch.inference_mode():
  for i,label in enumerate(labels):
   ll=scores[i];prob=torch.softmax(ll,0);row={'index':i,'target':int(label),'candidate_loglikelihoods':ll.cpu().tolist(),'candidate_probabilities':prob.cpu().tolist(),'rank_prediction':int(ll.argmax().cpu())}
   if i in generation_indices:
    source=int(source_indices[i])
    if source not in generation_cache:generation_cache[source]=runtime.generate(prefixes[0] if identical_prefix else prefixes[i])
    raw=generation_cache[source];parsed=parse_goal(raw);row.update({'free_output':raw,'parsed_goal':parsed,'valid':parsed is not None,'correct':parsed==GOALS[int(label)]})
   rows.append(row)
 generated=[r for r in rows if 'free_output' in r];return {'rows':rows,'rank_accuracy':float(np.mean([r['rank_prediction']==r['target'] for r in rows])),'generated_count':len(generated),'valid_count':sum(r['valid'] for r in generated),'correct_count':sum(r['correct'] for r in generated),'seconds':time.perf_counter()-start,'likelihood_batch_size':8,'generation_cache':'shared by source-prefix identity; cached outputs are not independent generation passes'}

def _validated_labels(labels,packet):
 labels=torch.as_tensor(labels)
 if labels.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8) or not torch.equal(labels.long().cpu(),packet.targets.cpu()):raise ValueError('integer labels must match packet targets')
 return labels.long().cpu()

def evaluate(adapter,packet,labels,generation_indices=None,runtime=None,shuffle_seed=SEED+2):
 runtime=runtime or Runtime.load();labels=_validated_labels(labels,packet)
 generation_indices=set(range(len(packet))) if generation_indices is None else set(map(int,generation_indices))
 if any(i<0 or i>=len(packet) for i in generation_indices):raise ValueError('generation index out of range')
 packet=_to(packet,runtime.device);adapter=load_adapter(adapter,runtime.device)[0] if isinstance(adapter,(str,Path)) else adapter.to(runtime.device);adapter.eval()
 with torch.inference_mode():prefix=adapter(packet)
 order=torch.as_tensor(np.random.default_rng(shuffle_seed).permutation(len(packet)),device=runtime.device);order_cpu=order.cpu().numpy();cache={}
 actual=_condition_report(runtime,prefix,labels,generation_indices,cache,np.arange(len(packet)));shuffled=_condition_report(runtime,prefix[order],labels,generation_indices,cache,order_cpu);zero=_condition_report(runtime,torch.zeros_like(prefix),labels,generation_indices,{},np.zeros(len(packet),int),True)
 return {'actual':actual,'zero_prefix':zero,'row_shuffled_prefix':shuffled,'shuffle_order':order_cpu.tolist(),'adapter_mode':adapter.mode,'base_checkpoint_sha256':runtime.checkpoint_sha256,'base_all_parameters_frozen':all(not p.requires_grad for p in runtime.model.parameters())}

def evaluate_no_input(packet,labels,generation_indices=None,runtime=None):
 runtime=runtime or Runtime.load();labels=_validated_labels(labels,packet)
 generation_indices=set(range(len(packet))) if generation_indices is None else set(map(int,generation_indices))
 if any(i<0 or i>=len(packet) for i in generation_indices):raise ValueError('generation index out of range')
 prefix=torch.empty((len(packet),0,getattr(runtime.model.config,'hidden_size',HIDDEN)),device=runtime.device)
 return _condition_report(runtime,prefix,labels,generation_indices,{},np.zeros(len(packet),int),True)
