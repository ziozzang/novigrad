#!/usr/bin/env python3
"""Frozen supervised probes for engineered context-conditioned KC representations.

These are offline ridge probes, not native reward updates, recurrence, thought, or
an identified biological context circuit.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file,save_file
from delayed_credit import ROOT
from inhibition_mechanism import ShadowEngine,checkpoint,sha256
from precise_bridge import PortBridge,dataset

RIDGE=.1;SEED=9901;CONTEXTS=4;ACTIONS=4
CYCLIC_RULES=np.array([[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]],np.int64)
ALTERNATE_RULES=np.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]],np.int64)
ARCHITECTURES=('blind_kc','additive_kc','interaction_kc','matched_interaction_kc','interaction_embedding','interaction_pca_z')
def frozen_spec():return {'architecture_probe_frozen':True,'architecture_probe_source_sha256':sha256(Path(__file__)),'ridge':RIDGE,'seed':SEED,'cyclic_rules':CYCLIC_RULES.tolist(),'alternate_rules':ALTERNATE_RULES.tolist(),'checkpoint_sha256':sha256(checkpoint(601)),'train_dataset_sha256':sha256(ROOT/'results/gemma-bridge/dataset.json'),'train_embeddings_sha256':sha256(ROOT/'results/gemma-bridge/embeddings.safetensors')}

def unit(x):
 x=np.asarray(x,np.float64);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def validate_rules(rules):
 r=np.asarray(rules)
 if r.shape!=(CONTEXTS,ACTIONS) or r.dtype.kind not in 'iu' or any(sorted(row.tolist())!=list(range(ACTIONS)) for row in r) or any(sorted(r[:,c].tolist())!=list(range(ACTIONS)) for c in range(ACTIONS)):raise ValueError('rules must be a balanced 4x4 integer Latin permutation table')
 return r.astype(np.int64)
def expand(x,labels,rules):
 x=np.asarray(x);labels=np.asarray(labels)
 if x.ndim!=2 or labels.ndim!=1 or len(x)!=len(labels) or labels.dtype.kind not in 'iu' or np.any((labels<0)|(labels>=ACTIONS)):raise ValueError('invalid features or labels')
 contexts=np.tile(np.arange(CONTEXTS),len(x));rows=np.repeat(np.arange(len(x)),CONTEXTS);targets=rules[contexts,labels[rows]]
 return x[rows],contexts,targets,rows

def interaction(x,contexts):
 x=np.asarray(x);g=np.asarray(contexts);out=np.zeros((len(x),CONTEXTS*x.shape[1]),np.float64)
 valid=(g>=0)&(g<CONTEXTS)
 for c in range(CONTEXTS):out[valid&(g==c),c*x.shape[1]:(c+1)*x.shape[1]]=x[valid&(g==c)]
 return out
def design(kind,reps,contexts,projection=None):
 h,z,e=reps['h'],reps['z'],reps['embedding'];g=np.asarray(contexts);one=np.zeros((len(g),CONTEXTS));valid=(g>=0)&(g<CONTEXTS);one[np.flatnonzero(valid),g[valid]]=1
 if kind=='blind_kc':base=h
 elif kind=='additive_kc':base=np.c_[h,one]
 elif kind=='interaction_kc':base=interaction(h,g)
 elif kind=='matched_interaction_kc':base=interaction(h@projection,g)
 elif kind=='interaction_embedding':base=interaction(e,g)
 elif kind=='interaction_pca_z':base=interaction(z,g)
 else:raise ValueError('unknown architecture')
 return np.c_[base,np.ones(len(base))]

class RidgeProbe:
 def __init__(self,kind,coef,projection=None):self.kind=kind;self.coef=coef;self.projection=projection
 def predict(self,reps,contexts):
  scores=design(self.kind,reps,contexts,self.projection)@self.coef;scores-=scores.max(1,keepdims=True);p=np.exp(scores);return p/p.sum(1,keepdims=True)

def fit_probes(reps,labels,rules,seed=SEED,shuffled_context=False):
 rules=validate_rules(rules);expanded={k:expand(v,labels,rules)[0] for k,v in reps.items()};_,contexts,targets,_=expand(reps['h'],labels,rules)
 if shuffled_context:contexts=np.random.default_rng(seed+71).permutation(contexts)
 d=reps['h'].shape[1];q=max(1,round((d+CONTEXTS)/CONTEXTS));rng=np.random.default_rng(seed);projection=rng.choice(np.array([-1.,1.]),size=(d,q))/np.sqrt(q)
 probes={};details={}
 for kind in ARCHITECTURES:
  proj=projection if kind=='matched_interaction_kc' else None;X=design(kind,expanded,contexts,proj);one=np.eye(ACTIONS)[targets];dual=np.linalg.solve(X@X.T+RIDGE*np.eye(len(X)),one);coef=X.T@dual;probes[kind]=RidgeProbe(kind,coef,proj)
  details[kind]={'design_width_with_intercept':X.shape[1],'coefficient_count':int(coef.size),'train_design_rank':int(np.linalg.matrix_rank(X)),'projected_kc_width':q if proj is not None else None,'capacity_matching_scope':'coefficient-count only; effective rank and representation geometry remain unequal' if proj is not None else None}
 return probes,details

def representations(bridge,shadow,embeddings):
 x=np.asarray(embeddings,np.float32);z=(x-bridge.mean)@bridge.projection;z=unit(z);ports=bridge.encode(x);h=shadow.inhibit(shadow.raw_hidden(ports),'native_topk_2',{})
 return {'h':np.asarray(h,np.float64),'z':z,'embedding':unit(x)}
def evaluate(probe,reps,labels,rules,contexts_override=None):
 _,contexts,targets,base_rows=expand(reps['h'],labels,rules);expanded={k:v[base_rows] for k,v in reps.items()};used=contexts if contexts_override is None else np.asarray(contexts_override);p=probe.predict(expanded,used);pred=p.argmax(1)
 return {'accuracy':float(np.mean(pred==targets)),'mean_correct_probability':float(p[np.arange(len(p)),targets].mean()),'predictions':pred.tolist(),'targets':targets.tolist(),'contexts_used':used.tolist(),'base_case_indices':base_rows.tolist(),'probabilities':p.tolist()}
def schedule_controls(probe,reps,labels,rules):
 n=min(len(labels),8);base=np.arange(n);true_contexts=(np.arange(n)//2)%CONTEXTS;targets=rules[true_contexts,labels[base]];sub={k:v[base] for k,v in reps.items()}
 repeated=probe.predict(sub,true_contexts);latched=[];latch=None
 for i,c in enumerate(true_contexts):
  if i==0 or c!=true_contexts[i-1]:latch=int(c)
  latched.append(latch)
 once=probe.predict(sub,np.array(latched));missing=probe.predict(sub,np.full(n,-1));stale=probe.predict(sub,np.zeros(n,dtype=int))
 reset_contexts=true_contexts.copy();reset_contexts[1::2]=-1;reset=probe.predict(sub,reset_contexts)
 metric=lambda p:float(np.mean(p.argmax(1)==targets))
 return {'schedule':true_contexts.tolist(),'context_switch_indices':np.flatnonzero(np.r_[True,true_contexts[1:]!=true_contexts[:-1]]).tolist(),'repeated_vs_cue_once_latch_exact':bool(np.array_equal(repeated,once)),'repeated_accuracy':metric(repeated),'cue_once_latch_accuracy':metric(once),'missing_context_accuracy':metric(missing),'stale_latch_accuracy':metric(stale),'reset_every_second_step_accuracy':metric(reset),'boundary':'host latch is deterministic external state, not neural recurrence'}

def run_probe(train_embeddings,train_labels,eval_embeddings,eval_labels,rules=CYCLIC_RULES,seed=SEED):
 rules=validate_rules(rules);bridge=PortBridge.fit(train_embeddings,'pca');shadow=ShadowEngine(checkpoint(601));train=representations(bridge,shadow,train_embeddings);ev=representations(bridge,shadow,eval_embeddings);probes,detail=fit_probes(train,train_labels,rules,seed);shuffled_probes,shuffled_detail=fit_probes(train,train_labels,rules,seed,True)
 rng=np.random.default_rng(seed+72);_,ctx,_,_=expand(ev['h'],eval_labels,rules);shuffled_eval=rng.permutation(ctx)
 models={}
 for kind,probe in probes.items():
  models[kind]={'architecture':detail[kind],'evaluation':evaluate(probe,ev,eval_labels,rules),'shuffled_eval_context':evaluate(probe,ev,eval_labels,rules,shuffled_eval),'missing_context':evaluate(probe,ev,eval_labels,rules,np.full(len(ctx),-1)),'trained_with_shuffled_context':evaluate(shuffled_probes[kind],ev,eval_labels,rules),'host_latch':schedule_controls(probe,ev,eval_labels,rules)}
 return {'boundary':'offline supervised ridge probes on frozen engineered representations; no native reward updates, recurrence, thought, or biological context mechanism','protocol':{'ridge':RIDGE,'seed':seed,'rules':rules.tolist(),'contexts':CONTEXTS,'actions':ACTIONS,'no_tuning':True,'expanded_rows_are_clustered_by_base_case':True,'shuffled_train_detail':shuffled_detail},'representations':{'pca_projection_rank':int(np.linalg.matrix_rank(bridge.projection)),'kc_width':train['h'].shape[1],'mean_active_kcs_train':float(np.count_nonzero(train['h'],axis=1).mean()),'embedding_width':train['embedding'].shape[1],'pca_z_width':train['z'].shape[1]},'models':models}

def _artifact_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _models_report(probes,shuffled_probes,detail,reps,labels,rules,seed=SEED):
 _,contexts,_,_=expand(reps['h'],labels,rules);shuffled_eval=np.random.default_rng(seed+72).permutation(contexts);models={}
 for kind,probe in probes.items():
  models[kind]={'architecture':detail[kind],'evaluation':evaluate(probe,reps,labels,rules),'shuffled_eval_context':evaluate(probe,reps,labels,rules,shuffled_eval),'missing_context':evaluate(probe,reps,labels,rules,np.full(len(contexts),-1)),'trained_with_shuffled_context':evaluate(shuffled_probes[kind],reps,labels,rules),'host_latch':schedule_controls(probe,reps,labels,rules)}
 return models

def save_bundle(out_dir):
 """Fit all development probes once and persist every inference tensor."""
 out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True);weights_path=out_dir/'architecture-probes.safetensors';manifest_path=out_dir/'architecture-probe-bundle.json'
 data=dataset();train_x,train_y,_=data['train'];bridge=PortBridge.fit(train_x,'pca');shadow=ShadowEngine(checkpoint(601));train=representations(bridge,shadow,train_x)
 tensors={'bridge.mean':bridge.mean,'bridge.projection':bridge.projection};details={};expected=[]
 for table_name,rules in (('cyclic',CYCLIC_RULES),('alternate',ALTERNATE_RULES)):
  normal,detail=fit_probes(train,train_y,rules,SEED);shuffled,shuffled_detail=fit_probes(train,train_y,rules,SEED,True);details[table_name]={'normal':detail,'shuffled':shuffled_detail}
  for family,probes in (('normal',normal),('shuffled',shuffled)):
   for kind,probe in probes.items():
    key=f'{table_name}.{family}.{kind}.coef';tensors[key]=np.ascontiguousarray(probe.coef,np.float64);expected.append(key)
    if probe.projection is not None:
     pkey=f'{table_name}.{family}.{kind}.projection';tensors[pkey]=np.ascontiguousarray(probe.projection,np.float64);expected.append(pkey)
 save_file(tensors,str(weights_path),metadata={'format':'novi.architecture-probe-bundle','version':'1','ridge':repr(RIDGE),'seed':str(SEED)})
 manifest={'format':'novi.architecture-probe-manifest','version':1,'frozen_spec':frozen_spec(),'weights':{'path':weights_path.name,'sha256':_artifact_sha(weights_path),'tensor_keys':sorted(tensors)},'fit':{'train_rows':len(train_x),'pca_projection_rank':int(np.linalg.matrix_rank(bridge.projection)),'kc_width':train['h'].shape[1],'mean_active_kcs':float(np.count_nonzero(train['h'],axis=1).mean()),'details':details},'boundary':'all supervised probes, including shuffled-context controls, fitted once on old train32; evaluation must load only'}
 manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');return manifest

def _load_bundle(out_dir):
 out_dir=Path(out_dir);manifest_path=out_dir/'architecture-probe-bundle.json';manifest=json.loads(manifest_path.read_text());expected=frozen_spec()
 if manifest.get('format')!='novi.architecture-probe-manifest' or manifest.get('version')!=1 or manifest.get('frozen_spec')!=expected:raise ValueError('bundle manifest source/input/protocol hashes do not match')
 weights_path=out_dir/manifest['weights']['path']
 if _artifact_sha(weights_path)!=manifest['weights']['sha256']:raise ValueError('bundle weights hash mismatch')
 with safe_open(str(weights_path),framework='numpy') as f:meta=f.metadata() or {};keys=sorted(f.keys())
 if meta!={'format':'novi.architecture-probe-bundle','version':'1','ridge':repr(RIDGE),'seed':str(SEED)} or keys!=manifest['weights']['tensor_keys']:raise ValueError('bundle tensor schema or metadata mismatch')
 tensors=load_file(str(weights_path));mean=tensors.pop('bridge.mean');projection=tensors.pop('bridge.projection')
 if mean.shape!=(768,) or projection.shape!=(768,128) or not np.isfinite(mean).all() or not np.isfinite(projection).all():raise ValueError('invalid saved PortBridge tensors')
 bridge=PortBridge(mean,projection,'pca');families={}
 for table_name in ('cyclic','alternate'):
  families[table_name]={}
  for family in ('normal','shuffled'):
   probes={}
   for kind in ARCHITECTURES:
    prefix=f'{table_name}.{family}.{kind}';coef=tensors.pop(prefix+'.coef');projection=tensors.pop(prefix+'.projection',None)
    if coef.ndim!=2 or coef.shape[1]!=ACTIONS or not np.isfinite(coef).all() or projection is not None and (projection.ndim!=2 or not np.isfinite(projection).all()):raise ValueError('invalid saved probe tensor')
    probes[kind]=RidgeProbe(kind,coef,projection)
   families[table_name][family]=probes
 if tensors:raise ValueError('unexpected saved probe tensors')
 return manifest,bridge,families

def evaluate_saved(out_dir,eval_embeddings,eval_labels):
 """Evaluate a verified fitted bundle without any fitting or mutable train access."""
 manifest,bridge,families=_load_bundle(out_dir);x=np.asarray(eval_embeddings);y=np.asarray(eval_labels)
 if x.ndim!=2 or x.shape[1]!=768 or not np.isfinite(x).all() or y.ndim!=1 or y.dtype.kind not in 'iu' or len(y)!=len(x) or np.any((y<0)|(y>=ACTIONS)):raise ValueError('invalid finite evaluation embeddings or integer class labels')
 reps=representations(bridge,ShadowEngine(checkpoint(601)),x);result={'stage':'saved_bundle_evaluation','bundle_manifest_sha256':_artifact_sha(Path(out_dir)/'architecture-probe-bundle.json'),'boundary':'load-only offline supervised probe evaluation; no refit and no native reward learning'}
 for table_name,rules in (('cyclic',CYCLIC_RULES),('alternate',ALTERNATE_RULES)):
  result[table_name]={'rules':rules.tolist(),'models':_models_report(families[table_name]['normal'],families[table_name]['shuffled'],manifest['fit']['details'][table_name]['normal'],reps,y,rules)}
 return result

def run_development(output):
 output=Path(output);bundle_dir=output.parent/'architecture-probe-bundle';save_bundle(bundle_dir);data=dataset();val_x,val_y,_=data['validation'];report=evaluate_saved(bundle_dir,val_x,val_y);report['stage']='development_validation12_load_only';report['lock_template']=frozen_spec();output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('bundle','development'),required=True);p.add_argument('--output');p.add_argument('--output-dir');a=p.parse_args()
 if a.stage=='bundle':
  if not a.output_dir:p.error('bundle requires --output-dir')
  r=save_bundle(a.output_dir);stage='bundle'
 else:
  if not a.output:p.error('development requires --output')
  r=run_development(a.output);stage=r['stage']
 print(json.dumps({'stage':stage,'output':a.output or a.output_dir},indent=2))
if __name__=='__main__':main()
