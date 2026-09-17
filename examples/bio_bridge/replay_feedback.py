#!/usr/bin/env python3
"""Replay causal-feedback artifacts with content-verified relocated Gemma models."""
import argparse,json
from pathlib import Path
from replay_bilateral import digest,relocated_models,verify_model_directory

RUNTIME_DEPENDENCIES=Path(__file__).resolve().parents[2]/'results/causal-feedback/model-runtime-dependencies.json'

def verify_runtime_dependencies(paths,lock_path=RUNTIME_DEPENDENCIES):
 lock_path=Path(lock_path)
 if not lock_path.is_file():raise FileNotFoundError('supplementary model runtime dependency lock missing')
 lock=json.loads(lock_path.read_text())
 if lock.get('schema_version')!=1 or lock.get('status')!='posthoc_dependency_lock_added_after_outcomes' or set(lock.get('models',{}))!={'embedding','function'}:raise ValueError('invalid model runtime dependency lock')
 if set(paths)!={'embedding','function'}:raise ValueError('runtime model path keys differ')
 counts={}
 for model,path in paths.items():
  expected=lock['models'][model].get('jinja_files')
  if not isinstance(expected,dict):raise ValueError('invalid Jinja dependency inventory')
  if model=='function' and 'chat_template.jinja' not in expected:raise ValueError('FunctionGemma chat_template.jinja is not locked')
  root=Path(path).resolve();actual={str(p.relative_to(root)) for p in root.rglob('*.jinja') if p.is_file()}
  if actual!=set(expected):raise ValueError(f'{model} Jinja inventory mismatch: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}')
  for name,record in expected.items():
   relative=Path(name)
   if relative.is_absolute() or '..' in relative.parts or not isinstance(record,dict):raise ValueError('invalid Jinja dependency record')
   value=root/relative
   if value.stat().st_size!=record.get('bytes') or digest(value)!=record.get('sha256'):raise ValueError(f'{model} Jinja content mismatch: {name}')
  counts[model]=len(expected)
 return {'lock_sha256':digest(lock_path),'jinja_files':counts,'status':lock['status']}

def _verify_flat_lock(out,label):
 lock=out/'evaluation-lock.json'
 if not lock.exists():raise FileNotFoundError(f'{label} evaluation-lock.json missing; run is incomplete')
 record=json.loads(lock.read_text())
 if not isinstance(record,dict) or not record:raise ValueError(f'invalid {label} evaluation lock')
 for name,expected in record.items():
  path=out/name
  if not path.is_file() or digest(path)!=expected:raise ValueError(f'changed or missing {label} artifact: {name}')
 return len(record)

def verify_evaluation_complete(study,feedback,hard=None):
 old=study.OUT/'evaluation-lock.json';new=feedback.OUT/'evaluation-lock.json'
 if not old.exists():raise FileNotFoundError('bilateral evaluation-lock.json missing; partial study cannot be verified')
 old_record=json.loads(old.read_text())
 if set(old_record)!={'hashes'} or not isinstance(old_record['hashes'],dict):raise ValueError('invalid bilateral evaluation lock')
 study.check_files(old_record['hashes'])
 counts={'bilateral_files':len(old_record['hashes']),'causal_feedback_files':_verify_flat_lock(feedback.OUT,'causal-feedback')}
 if hard is not None:counts['hard_feedback_routing_files']=_verify_flat_lock(hard.OUT,'hard-feedback-routing')
 return counts

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--embedding-model',required=True,type=Path);p.add_argument('--function-model',required=True,type=Path);p.add_argument('--check-only',action='store_true');p.add_argument('--include-hard-routing',action='store_true',help='also verify and replay the completed hard-routing post-hoc diagnostic');a=p.parse_args()
 import bilateral_study as study
 import bilateral_lm as lm
 import causal_feedback as feedback
 hard=None
 if a.include_hard_routing:
  import hard_feedback_routing as hard
 lock=json.loads((study.OUT/'protocol-lock.json').read_text());paths={key:verify_model_directory(path,lock['models'][key]['files']) for key,path in [('embedding',a.embedding_model),('function',a.function_model)]}
 runtime_dependencies=verify_runtime_dependencies(paths)
 with relocated_models(study,lm,paths):
  study.checked_lock();feedback.checked()
  if hard is not None:hard.checked()
  counts=verify_evaluation_complete(study,feedback,hard)
  if not a.check_only:
   feedback.verify()
   if hard is not None:hard.verify()
 report={'bilateral_protocol_sha256':digest(study.OUT/'protocol-lock.json'),'causal_feedback_protocol_sha256':digest(feedback.OUT/'protocol-lock.json'),'causal_feedback_evaluation_sha256':digest(feedback.OUT/'evaluation-lock.json'),'model_runtime_dependencies':runtime_dependencies,'verified_locked_files':counts,'model_paths':{k:str(v) for k,v in paths.items()},'check_only':a.check_only,'include_hard_routing':a.include_hard_routing,'boundary':'Lookup-only model relocation; original model contents, frozen files, runtime precision, and replay tolerances are unchanged. The post-outcome supplementary lock discloses the formerly omitted runtime Jinja dependency.'}
 if hard is not None:report.update({'hard_feedback_protocol_sha256':digest(hard.OUT/'protocol-lock.json'),'hard_feedback_evaluation_sha256':digest(hard.OUT/'evaluation-lock.json')})
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
