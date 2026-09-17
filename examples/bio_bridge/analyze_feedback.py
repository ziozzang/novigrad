#!/usr/bin/env python3
"""Read-only post-hoc summaries for the frozen causal-feedback diagnostic."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/causal-feedback';DRAWS=10000;SEED=3011
CONTRASTS=(('belief_full','original'),('belief_full','static_prior_prefix'),('belief_full','permuted_prototypes'),('belief_full','prior_ranked_no_repeat'),('belief_half','belief_full'))
GOALS=('water','food','warmth','rest')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _summary(rows):
 n=len(rows);success=[bool(r['success']) for r in rows];winning=[r for r in rows if r['success']]
 return {'n':n,'successes':sum(success),'success_rate':float(np.mean(success)),'mean_physical_actions':float(np.mean([r['physical_actions'] for r in rows])),'conditional_mean_decisions_among_successes':None if not winning else float(np.mean([r['decisions'] for r in winning])),'invalid_episodes':sum(bool(r['invalid']) for r in rows)}
def grouped(rows,cases,key):
 groups={}
 for row,case in zip(rows,cases):groups.setdefault(case[key],[]).append(row)
 return {name:_summary(value) for name,value in sorted(groups.items())}
def condition_summary(rows,cases):return {'overall':_summary(rows),'by_class':grouped(rows,cases,'class'),'by_language':grouped(rows,cases,'language'),'by_family':grouped(rows,cases,'family')}
def paired_cluster_bootstrap(left,right,pair_ids,draws=DRAWS,seed=SEED):
 a=np.asarray(left,bool);b=np.asarray(right,bool);pairs=np.asarray(pair_ids)
 if a.shape!=b.shape or a.ndim!=1 or len(a)!=len(pairs):raise ValueError('paired rows and IDs must align')
 names=np.unique(pairs)
 if len(names)<2 or any(np.sum(pairs==name)!=2 for name in names):raise ValueError('each bilingual pair must have exactly two aligned rows')
 delta=np.array([np.mean(a[pairs==name])-np.mean(b[pairs==name]) for name in names]);rng=np.random.default_rng(seed);boot=delta[rng.integers(0,len(delta),size=(draws,len(delta)))].mean(1)
 return {'left_success_rate':float(a.mean()),'right_success_rate':float(b.mean()),'paired_success_difference_left_minus_right':float(a.mean()-b.mean()),'clusters':len(names),'cluster_unit':'bilingual EN/KO pair','draws':draws,'seed':seed,'percentile_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
def reachable_ceiling(record):
 actions=np.asarray(record['actions'])
 if actions.shape!=(4,) or actions.dtype.kind not in 'iu' or np.any((actions<0)|(actions>=4)):raise ValueError('invalid frozen native map')
 reachable=sorted(set(actions.tolist()))
 return {'command_to_executed_action':actions.tolist(),'reachable_goal_ids':reachable,'reachable_goals':[GOALS[i] for i in reachable],'reachable_goal_count':len(reachable),'balanced_four_class_case_ceiling':len(reachable)/4,'boundary':'combinatorial actuator reachability ceiling only; it is not observed success or a learned-policy estimate'}
def prior_diagnostics(prior,labels):
 p=np.asarray(prior,np.float64);y=np.asarray(labels)
 if p.shape!=(len(y),4) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,rtol=0,atol=1e-8):raise ValueError('invalid saved prior probabilities')
 one=np.eye(4)[y];return {'n':len(y),'top1_accuracy':float(np.mean(p.argmax(1)==y)),'multiclass_brier_mean_sum_squared_error':float(np.mean(np.sum((p-one)**2,axis=1))),'mean_target_probability':float(p[np.arange(len(y)),y].mean()),'boundary':'descriptive diagnostics for length-normalized candidate likelihood softmax; these values are not claimed to be calibrated probabilities'}
def _load():
 required=[OUT/'results.json',OUT/'protocol-lock.json',OUT/'inputs.safetensors',OUT/'cases.json'];missing=[str(p) for p in required if not p.exists()]
 if missing:raise FileNotFoundError('completed run inputs required: '+', '.join(missing))
 lock=json.loads(required[1].read_text());results=json.loads(required[0].read_text());cases=json.loads(required[3].read_text());tensors=load_file(str(required[2]))
 if len(cases)!=64 or len({c['pair_id'] for c in cases})!=32:raise ValueError('expected 64 aligned cases in 32 bilingual pairs')
 expected=set()
 for model in lock['models']:
  for policy in lock['maps']:
   for scenario in lock['scenarios']:
    modes=lock['modes'] if scenario=='immediate' else ('belief_full','analytic_fixed_model')
    expected.update(f'{model}/{policy}/{scenario}/{mode}' for mode in modes)
 if set(results)!=expected:raise ValueError('results condition inventory is incomplete or unexpected')
 for key,value in results.items():
  rows=value.get('cases')
  if not isinstance(rows,list) or len(rows)!=64 or [r.get('case_index') for r in rows]!=list(range(64)):raise ValueError('misaligned cases: '+key)
 return required,lock,results,cases,tensors
def analyze(output=OUT/'posthoc-analysis.json'):
 files,lock,results,cases,tensors=_load();conditions={key:condition_summary(value['cases'],cases) for key,value in sorted(results.items())};pairs=[c['pair_id'] for c in cases];contrasts={}
 for model_index,model in enumerate(lock['models']):
  contrasts[model]={}
  for offset,(left,right) in enumerate(CONTRASTS):
   l=results[f'{model}/native701/immediate/{left}']['cases'];r=results[f'{model}/native701/immediate/{right}']['cases'];contrasts[model][f'{left}_minus_{right}']=paired_cluster_bootstrap([x['success'] for x in l],[x['success'] for x in r],pairs,seed=SEED+model_index*10+offset)
 labels=np.array([GOALS.index(c['class']) for c in cases]);priors={model:prior_diagnostics(tensors[model+'/prior'].numpy(),labels) for model in lock['models']}
 report={'status':'post-hoc descriptive analysis; no model selection, training-seed inference, calibrated-probability claim, or topology-generalization claim','conditions':conditions,'native_executor_reachability':{name:reachable_ceiling(record) for name,record in lock['maps'].items()},'paired_native701_immediate_contrasts':contrasts,'saved_prior_diagnostics':priors,'interpretation':{'efficiency_caution':'Mean physical actions includes invalid early stops. Conditional mean decisions is reported only among successful episodes and must not be read as unconditional efficiency.','uncertainty':'There is one training seed per adapter. The paired bootstrap resamples 32 authored bilingual case pairs, not training runs or independent topology samples. It does not support a population or topology-generalization claim.','prior':'The prior is a temperature-1 softmax of response log likelihood divided by response token count. Brier score and top-1 accuracy are descriptive; no calibration claim is made.','policy_boundary':'Hidden labels are used by the environment/evaluator and this post-hoc scorer, never introduced into the saved controller traces by this analysis.'},'provenance':{'source_sha256':sha(__file__),'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'bootstrap_seed':SEED,'bootstrap_draws':DRAWS}}
 output=Path(output)
 if output.exists():raise FileExistsError(f'refusing to overwrite {output}')
 output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT/'posthoc-analysis.json');a=p.parse_args();r=analyze(a.output);print(json.dumps({'conditions':len(r['conditions']),'output':str(a.output)},indent=2))
