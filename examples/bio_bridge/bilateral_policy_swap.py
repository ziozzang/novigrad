#!/usr/bin/env python3
"""Frozen plastic-policy swap for the two-tick soft-prefix diagnostic.

The swap is allowed only when every saved graph/topology tensor is identical and
only plastic_weight differs. It tests checkpoint calibration, not biological
transfer, and never retrains the language adapter.
"""
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file
from novigrad import Engine
from bilateral_closed_loop import DEFAULT_INDICES,prototypes_from_old_train,run_loop_components
from bilateral_lm import load_adapter
from bilateral_temporal import SiteCodec
from inhibition_mechanism import ROOT,ShadowEngine,checkpoint,sha256

BASE_POLICY=checkpoint(601)
PCA_POLICY=ROOT/'results/precise-bridge/native-checkpoints/pca-seed-701.safetensors'
EXPECTED_KEYS={'input_ids','hidden_ids','output_ids','input_pre','input_post','input_count','input_sign','plastic_pre','plastic_post','plastic_count','plastic_sign','output_actions','output_gains','plastic_weight'}
FROZEN_KEYS=EXPECTED_KEYS-{'plastic_weight'}

def verify_policy_slot(base_path=BASE_POLICY,swap_path=PCA_POLICY):
 base_path,swap_path=Path(base_path),Path(swap_path)
 with safe_open(str(base_path),framework='numpy') as a, safe_open(str(swap_path),framework='numpy') as b:
  am,bm=a.metadata() or {},b.metadata() or {};ak,bk=set(a.keys()),set(b.keys())
  if ak!=EXPECTED_KEYS or bk!=EXPECTED_KEYS:raise ValueError('checkpoint tensor schema mismatch')
  for field in ('format','version','actions','learning_rate','active_fraction','logit_gain','homeostasis','weight_limit'):
   if am.get(field)!=bm.get(field):raise ValueError(f'checkpoint metadata mismatch: {field}')
  matched={}
  for key in sorted(FROZEN_KEYS):
   matched[key]=bool(np.array_equal(a.get_tensor(key),b.get_tensor(key)))
   if not matched[key]:raise ValueError(f'policy swap changed frozen graph slot: {key}')
  old=a.get_tensor('plastic_weight');new=b.get_tensor('plastic_weight')
  if old.shape!=new.shape or old.dtype!=new.dtype or not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('invalid plastic policy tensors')
  changed=int(np.count_nonzero(old!=new))
  if changed==0:raise ValueError('policy swap does not change plastic weights')
 return {'compatible':True,'frozen_tensor_keys':sorted(FROZEN_KEYS),'all_frozen_tensors_exact':all(matched.values()),'plastic_weight_changed_count':changed,'plastic_weight_count':old.size,'plastic_weight_max_abs_delta':float(np.max(np.abs(old-new))),'base_sha256':sha256(base_path),'swap_sha256':sha256(swap_path),'scope':'only saved plastic_weight values differ; ordered input/hidden/output IDs, PN-KC topology, plastic slots, signs, action routing, and gains are exact'}

def _swapped_components(codec_path):
 compatibility=verify_policy_slot();codec=SiteCodec.load(codec_path);codec.shadow=ShadowEngine(PCA_POLICY);engine=Engine.load(PCA_POLICY);return compatibility,codec,engine

def prototype_diagnostic(codec_path):
 """Development-only typed compatibility check on four old-train prototypes."""
 compatibility,codec,engine=_swapped_components(codec_path);prototypes=prototypes_from_old_train();pn=codec.bridge.encode(prototypes);shadow_prob,hidden=codec.shadow.forward(pn,'native_topk_2',{});native=np.asarray(engine.infer_batch(pn.tolist()),np.float64);error=float(np.max(np.abs(native-shadow_prob)))
 if error>2e-6:raise AssertionError(f'native/shadow mismatch: {error}')
 return {'stage':'development_only_old_train_prototypes','actions':native.argmax(1).tolist(),'probabilities':native.tolist(),'external_labels':[0,1,2,3],'accuracy':float(np.mean(native.argmax(1)==np.arange(4))),'active_kcs':np.count_nonzero(hidden,axis=1).tolist(),'native_shadow_max_abs_error':error,'compatibility':compatibility,'boundary':'four train-derived class means only; not final evidence or an improvement claim'}

def run_policy_swap(adapter_path,codec_path,embeddings,labels,indices=DEFAULT_INDICES,runtime=None):
 if runtime is None:
  from bilateral_lm import Runtime
  runtime=Runtime.load()
 compatibility,codec,engine=_swapped_components(codec_path);adapter,adapter_manifest=load_adapter(adapter_path,runtime.device);report=run_loop_components(adapter,codec,engine,prototypes_from_old_train(),embeddings,labels,indices,runtime)
 report['condition']='pretrained_pca_native_701_policy_swap_with_unchanged_soft_prefix_adapter'
 report['policy_swap']={'compatibility':compatibility,'adapter_retrained':False,'scope':'plastic policy checkpoint changed in the exact same graph slots; fixed PN-KC wiring and LM adapter retained','claim_boundary':'diagnostic of checkpoint calibration impact on generated-goal fixed points; no improvement or biological transfer claim'}
 paths=(Path(adapter_path),Path(codec_path),BASE_POLICY,PCA_POLICY,ROOT/'results/gemma-bridge/dataset.json',ROOT/'results/gemma-bridge/embeddings.safetensors',Path(__file__))
 report['provenance']={'files':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha256(p) for p in paths},'adapter_training_manifest':adapter_manifest,'base_policy_sha256':sha256(BASE_POLICY),'swapped_policy_sha256':sha256(PCA_POLICY)}
 return report
