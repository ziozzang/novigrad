#!/usr/bin/env python3
"""Two-tick generated-call feedback diagnostic over frozen engineered components.

A parsed tool call selects a train-only semantic prototype. The host encodes
that prototype and evaluates a stateless rate circuit. This is neither direct
hidden-state stimulation nor a biological recurrent loop.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from novigrad import Engine
from bilateral_lm import GOALS,load_adapter,load_train,parse_goal
from bilateral_temporal import SiteCodec
from precise_bridge import ROOT,dataset
from inhibition_mechanism import checkpoint,sha256

DEFAULT_INDICES=(0,1,16,17,32,33,48,49);PARITY_ATOL=2e-6

def prototypes_from_old_train():
 x,y,_=dataset()['train'];x=np.asarray(x,np.float32);y=np.asarray(y,np.int64)
 if x.shape!=(32,768) or any(np.sum(y==c)==0 for c in range(4)):raise ValueError('unexpected old train prototype source')
 return np.stack([x[y==c].mean(0) for c in range(4)]).astype(np.float32)
def _packet_from_encoded(encoded):
 encoded=np.asarray(encoded,np.float32)
 return load_train(encoded,np.ones(encoded.shape[:2],bool),np.tile(np.arange(encoded.shape[1]),(len(encoded),1)),np.zeros(encoded.shape[:2],np.float32),np.zeros(len(encoded),np.int64))
def _prefix(adapter,codec,embeddings,runtime):
 packet=_packet_from_encoded(codec.encode(np.asarray(embeddings,np.float32),'circuit'))
 with torch.inference_mode():return adapter(_to_device(packet,runtime.device))
def _to_device(packet,device):
 from bilateral_lm import Packet
 return Packet(*(x.to(device) for x in (packet.features,packet.valid_mask,packet.site_ids,packet.times,packet.targets)))
def _valid_inputs(embeddings,labels,indices):
 x=np.asarray(embeddings);y=np.asarray(labels);ix=np.asarray(indices)
 if x.ndim!=2 or x.shape[1]!=768 or not np.isfinite(x).all():raise ValueError('embeddings must be finite [N,768]')
 if y.ndim!=1 or y.dtype.kind not in 'iu' or len(y)!=len(x) or np.any((y<0)|(y>=4)):raise ValueError('labels must be integer [N] class IDs')
 if ix.ndim!=1 or ix.dtype.kind not in 'iu' or len(ix)==0 or len(np.unique(ix))!=len(ix) or np.any((ix<0)|(ix>=len(x))):raise ValueError('indices must be unique in-range integers')
 return x,y,ix.astype(np.int64)

def run_loop_components(adapter,codec,engine,prototypes,embeddings,labels,indices,runtime,parity_atol=PARITY_ATOL):
 """Dependency-injected core used by the real loader and CPU unit tests."""
 x,y,indices=_valid_inputs(embeddings,labels,indices);proto=np.asarray(prototypes,np.float32)
 if proto.shape!=(4,768) or not np.isfinite(proto).all() or parity_atol<0:raise ValueError('invalid prototypes or parity tolerance')
 adapter=adapter.to(runtime.device).eval();initial=_prefix(adapter,codec,x[indices],runtime);cases=[]
 for local,(index,true_label) in enumerate(zip(indices,y[indices])):
  original_prefix=initial[local];raw1=runtime.generate(original_prefix);goal1=parse_goal(raw1);record={'index':int(index),'external_initial_need':GOALS[int(true_label)],'tick1':{'raw':raw1,'parsed_goal':goal1,'correct':goal1==GOALS[int(true_label)]},'state_write':None,'native':None}
  next_prefix=original_prefix
  if goal1 is not None:
   goal_index=GOALS.index(goal1);chosen=proto[goal_index:goal_index+1];pn=codec.bridge.encode(chosen);shadow_prob,hidden=codec.shadow.forward(pn,'native_topk_2',{});native_prob=np.asarray(engine.infer_batch(pn.tolist()),np.float64);error=float(np.max(np.abs(native_prob-shadow_prob)))
   if error>parity_atol:raise AssertionError(f'native/shadow probability mismatch: {error}')
   action=int(native_prob[0].argmax());next_prefix=_prefix(adapter,codec,chosen,runtime)[0]
   record['state_write']={'accepted':True,'path':'parsed set_goal -> old-train class-mean embedding -> saved SiteCodec PN/KC/MBON rates','prototype_goal':goal1}
   record['native']={'probabilities':native_prob[0].tolist(),'action':action,'action_goal':GOALS[action],'action_matches_generated_goal':action==goal_index,'action_matches_external_initial_need':action==int(true_label),'shadow_probabilities':np.asarray(shadow_prob[0],float).tolist(),'max_abs_parity_error':error,'hidden_active_count':int(np.count_nonzero(hidden[0]))}
  else:record['state_write']={'accepted':False,'path':None,'reason':'strict parse rejected; state and circuit unchanged'}
  raw2=runtime.generate(next_prefix);goal2=parse_goal(raw2);record['tick2']={'raw':raw2,'parsed_goal':goal2,'correct':goal2==GOALS[int(true_label)]};record['dynamics']={'goal_stable':goal1 is not None and goal2==goal1,'correct_stable':goal1==GOALS[int(true_label)] and goal2==goal1,'wrong_fixed_point':goal1 is not None and goal1!=GOALS[int(true_label)] and goal2==goal1,'rescue':goal1 is not None and goal1!=GOALS[int(true_label)] and goal2==GOALS[int(true_label)],'destabilized':goal1==GOALS[int(true_label)] and goal2!=goal1}
  cases.append(record)
 count=len(cases);valid=[c for c in cases if c['tick1']['parsed_goal'] is not None];native=[c for c in cases if c['native'] is not None]
 return {'scope':'two-tick host discrete tool-to-prototype writeback through a stateless frozen rate circuit; every case independently resets to its supplied input; no reward learning, direct hidden stimulation, environment-goal learning, biological recurrence, or thought claim','parameters':{'ticks':2,'indices':indices.tolist(),'predeclared_final_indices':list(DEFAULT_INDICES),'parity_atol':parity_atol,'prototype':'un-normalized class mean of old train32 embeddings','invalid_policy':'reject with no state/circuit write; tick2 reuses unchanged initial prefix'},'cases':cases,'metrics':{'n':count,'tick1_valid':len(valid),'tick1_correct':sum(c['tick1']['correct'] for c in cases),'tick2_correct':sum(c['tick2']['correct'] for c in cases),'goal_stable':sum(c['dynamics']['goal_stable'] for c in cases),'correct_stable':sum(c['dynamics']['correct_stable'] for c in cases),'wrong_fixed_points':sum(c['dynamics']['wrong_fixed_point'] for c in cases),'rescues':sum(c['dynamics']['rescue'] for c in cases),'destabilized':sum(c['dynamics']['destabilized'] for c in cases),'native_evaluated':len(native),'native_action_correct_external_need':sum(c['native']['action_matches_external_initial_need'] for c in native),'native_action_matches_generated_goal':sum(c['native']['action_matches_generated_goal'] for c in native),'max_native_shadow_probability_error':max((c['native']['max_abs_parity_error'] for c in native),default=None)}}

def run_loop(adapter_path,codec_path,embeddings,labels,indices,runtime):
 """Load frozen artifacts and run the predeclared diagnostic without training."""
 adapter,adapter_manifest=load_adapter(adapter_path,runtime.device);codec=SiteCodec.load(codec_path);native_path=checkpoint(601);engine=Engine.load(native_path);prototypes=prototypes_from_old_train();report=run_loop_components(adapter,codec,engine,prototypes,embeddings,labels,indices,runtime)
 sources=[Path(adapter_path),Path(codec_path),native_path,ROOT/'results/gemma-bridge/dataset.json',ROOT/'results/gemma-bridge/embeddings.safetensors',Path(__file__)]
 report['provenance']={'files':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha256(p) for p in sources},'adapter_training_manifest':adapter_manifest,'prototype_input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'results/gemma-bridge/dataset.json',ROOT/'results/gemma-bridge/embeddings.safetensors')},'native_checkpoint_seed':601,'native_checkpoint_used_for_both_engine_and_shadow':True}
 return report
