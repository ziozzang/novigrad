#!/usr/bin/env python3
"""Post-hoc descriptive analysis of the nine frozen bilateral LM final reports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/bilateral-bridge';CASES=ROOT/'examples/bio_bridge/bilateral_holdout.json'
MODELS=('circuit_learned','circuit_fixed_queries','circuit_pooled_mlp','embedding_learned','label_oracle','circuit_learned_long','circuit_fixed_queries_long','circuit_pooled_mlp_long','embedding_learned_long')
COMPARISONS=(('circuit_learned_long','circuit_fixed_queries_long'),('circuit_learned_long','circuit_pooled_mlp_long'),('circuit_learned_long','embedding_learned_long'))
TRANSITIONS=(('learned','circuit_learned','circuit_learned_long'),('fixed_queries','circuit_fixed_queries','circuit_fixed_queries_long'),('pooled_mlp','circuit_pooled_mlp','circuit_pooled_mlp_long'),('direct_embedding','embedding_learned','embedding_learned_long'))
BOOTSTRAP_SEED=2719;BOOTSTRAP_DRAWS=10000

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _accuracy(rows):return float(np.mean([bool(r['correct']) for r in rows]))
def _groups(rows,cases,key):
 values={}
 for row,case in zip(rows,cases):values.setdefault(case[key],[]).append(row)
 return {name:{'n':len(group),'correct':sum(bool(r['correct']) for r in group),'valid':sum(bool(r.get('valid')) for r in group),'accuracy':_accuracy(group)} for name,group in sorted(values.items())}
def summarize(rows,cases):
 return {'overall':{'n':len(rows),'correct':sum(bool(r['correct']) for r in rows),'valid':sum(bool(r.get('valid')) for r in rows),'accuracy':_accuracy(rows)},'by_language':_groups(rows,cases,'language'),'by_class':_groups(rows,cases,'class'),'by_family':_groups(rows,cases,'family')}
def paired_cluster_bootstrap(left,right,clusters,draws=BOOTSTRAP_DRAWS,seed=BOOTSTRAP_SEED):
 left=np.asarray(left,bool);right=np.asarray(right,bool);clusters=np.asarray(clusters);names=np.unique(clusters)
 if left.shape!=right.shape or len(left)!=len(clusters) or len(names)<2:raise ValueError('invalid paired cluster inputs')
 deltas=[]
 for name in names:
  mask=clusters==name;deltas.append(float(np.mean(left[mask])-np.mean(right[mask])))
 deltas=np.asarray(deltas);rng=np.random.default_rng(seed);sampled=deltas[rng.integers(0,len(deltas),size=(draws,len(deltas)))].mean(1)
 return {'left_accuracy':float(left.mean()),'right_accuracy':float(right.mean()),'paired_accuracy_difference_left_minus_right':float(left.mean()-right.mean()),'cluster_count':len(names),'draws':draws,'seed':seed,'percentile_95_ci':[float(np.quantile(sampled,.025)),float(np.quantile(sampled,.975))],'cluster_unit':'EN/KO pair' if len(names)==32 else 'scenario family'}
def transitions(short,long,cases):
 a=np.asarray(short,bool);b=np.asarray(long,bool);labels=np.where(~a&b,'help',np.where(a&~b,'hurt',np.where(a&b,'both_correct','both_wrong')))
 return {'n':len(a),'help':int(np.sum(labels=='help')),'hurt':int(np.sum(labels=='hurt')),'both_correct':int(np.sum(labels=='both_correct')),'both_wrong':int(np.sum(labels=='both_wrong')),'net_help_minus_hurt':int(np.sum(labels=='help')-np.sum(labels=='hurt')),'cases':[{'id':case['id'],'pair_id':case['pair_id'],'family':case['family'],'language':case['language'],'transition':str(label)} for case,label in zip(cases,labels)]}
def _load():
 cases=json.loads(CASES.read_text())
 if not isinstance(cases,list) or len(cases)!=64 or len({c['id'] for c in cases})!=64 or len({c['pair_id'] for c in cases})!=32 or len({c['family'] for c in cases})!=8:raise ValueError('expected frozen 64 cases, 32 EN/KO pairs, and 8 families')
 reports={};paths={};shuffle={}
 for model in MODELS:
  path=OUT/f'{model}-final.json'
  if not path.exists():raise FileNotFoundError(f'all nine finals required; missing {path}')
  report=json.loads(path.read_text());rows=report.get('actual',{}).get('rows');ids=report.get('case_ids')
  if not isinstance(rows,list) or len(rows)!=64 or ids!=[c['id'] for c in cases] or [r.get('index') for r in rows]!=list(range(64)) or any('correct' not in r or 'free_output' not in r for r in rows):raise ValueError(f'invalid or misordered actual free-generation rows: {model}')
  order=report.get('shuffle_order')
  if not isinstance(order,list) or sorted(order)!=list(range(64)):raise ValueError(f'invalid shuffle permutation: {model}')
  targets=np.array([r['target'] for r in rows]);order=np.asarray(order)
  shuffle[model]={'same_label_count':int(np.sum(targets[order]==targets)),'same_label_fraction':float(np.mean(targets[order]==targets)),'fixed_point_count':int(np.sum(order==np.arange(64))),'fixed_point_fraction':float(np.mean(order==np.arange(64))),'boundary':'row shuffle is a partial disruption: same-label assignments preserve class information and fixed points preserve the exact row'}
  reports[model]=rows;paths[model]=path
 return cases,reports,paths,shuffle
def analyze(output=OUT/'posthoc-analysis.json'):
 cases,reports,paths,shuffle=_load();correct={m:np.array([bool(r['correct']) for r in rows]) for m,rows in reports.items()};pair=np.array([c['pair_id'] for c in cases]);family=np.array([c['family'] for c in cases])
 comparison={}
 for offset,(left,right) in enumerate(COMPARISONS):comparison[f'{left}_minus_{right}']={'paired_32_enko_clusters':paired_cluster_bootstrap(correct[left],correct[right],pair,seed=BOOTSTRAP_SEED+offset),'optional_8_family_clusters':paired_cluster_bootstrap(correct[left],correct[right],family,seed=BOOTSTRAP_SEED+100+offset)}
 report={'status':'post-hoc descriptive analysis of all nine frozen final reports; no model selection or superiority claim','models':{m:summarize(reports[m],cases) for m in MODELS},'shuffle_control_diagnostics':shuffle,'paired_long_comparisons':comparison,'step_120_to_480_transitions':{name:transitions(correct[short],correct[long],cases) for name,short,long in TRANSITIONS},'uncertainty':{'bootstrap':'paired accuracy differences; percentile interval; 10,000 fixed-seed draws','primary_cluster_unit':'32 bilingual EN/KO pairs','optional_family_cluster_unit':'8 authored scenario families','development_informed_design':'The 480-update follow-up was added after development results; it was not part of the original 120-update design. The 120/480 comparison reuses the same deterministic stream prefix and a single training seed, so it is a within-run duration contrast, not independent replication.','shuffle_limitation':'Row shuffle does not remove all class information: approximately one quarter of rows can retain the same class by chance, separately reported for every saved permutation; exact permutation fixed points are also reported.','limitations':'There are no repeated training seeds. The 32 bilingual pairs remain authored and the optional family analysis has only 8 clusters. Intervals describe this case set and do not establish model superiority or population-level uncertainty.'},'provenance':{'source_sha256':sha(__file__),'input_sha256':{str(CASES.relative_to(ROOT)):sha(CASES),**{str(p.relative_to(ROOT)):sha(p) for p in paths.values()}},'labels_source':'unchanged bilateral_holdout.json; analysis performs no relabeling','bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_draws':BOOTSTRAP_DRAWS}}
 output=Path(output)
 if output.exists():raise FileExistsError(f'refusing to overwrite {output}')
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT/'posthoc-analysis.json');a=p.parse_args();r=analyze(a.output);print(json.dumps({'output':str(a.output),'models':len(r['models'])},indent=2))
