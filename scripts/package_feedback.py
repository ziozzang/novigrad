#!/usr/bin/env python3
"""Pack or safely unpack the bilateral plus causal-feedback replay bundle."""
import argparse,ast,io,json
from pathlib import Path,PurePosixPath
import tarfile,tempfile
from package_bilateral import MAX_MEMBER,MAX_TOTAL,safe_name,sha,source_bytes

ROOT=Path(__file__).resolve().parents[1];OLD_MANIFEST='results/bilateral-bridge/bundle-manifest.json';MANIFEST='results/causal-feedback/bundle-manifest.json';DEFAULT_ARCHIVE=ROOT/'dist/novigrad-feedback-20260918.tar.gz'

def locked_inventory(root=ROOT):
 root=Path(root);old_path=root/OLD_MANIFEST;protocol=root/'results/causal-feedback/protocol-lock.json';evaluation=root/'results/causal-feedback/evaluation-lock.json'
 for path,label in ((old_path,'bilateral bundle manifest'),(protocol,'causal-feedback protocol lock'),(evaluation,'causal-feedback evaluation lock')):
  if not path.is_file():raise FileNotFoundError(f'{label} missing; completed frozen run required')
 old=json.loads(old_path.read_text());pl=json.loads(protocol.read_text());el=json.loads(evaluation.read_text())
 if old.get('schema_version')!=1 or not isinstance(old.get('files'),dict):raise ValueError('invalid bilateral bundle manifest')
 if not isinstance(pl.get('frozen'),dict) or not isinstance(el,dict) or not el:raise ValueError('invalid causal-feedback locks')
 expected={name:record['sha256'] for name,record in old['files'].items()}
 for records,prefix in ((pl['frozen'],''),(el,'results/causal-feedback/')):
  for raw_name,digest in records.items():
   name=prefix+raw_name
   if name in expected and expected[name]!=digest:raise ValueError(f'conflicting locked hash: {name}')
   expected[name]=digest
 hard=root/'results/hard-feedback-routing'
 if hard.exists():
  hp,he=hard/'protocol-lock.json',hard/'evaluation-lock.json'
  if not hp.is_file() or not he.is_file():raise FileNotFoundError('hard-feedback routing protocol/evaluation lock missing; completed frozen run required')
  hpl,hel=json.loads(hp.read_text()),json.loads(he.read_text())
  if not isinstance(hpl.get('frozen'),dict) or not isinstance(hel,dict) or not hel:raise ValueError('invalid hard-feedback locks')
  for records,prefix in ((hpl['frozen'],''),(hel,'results/hard-feedback-routing/')):
   for raw_name,digest in records.items():
    name=prefix+raw_name
    if name in expected and expected[name]!=digest:raise ValueError(f'conflicting locked hash: {name}')
    expected[name]=digest
 return old,pl,el,expected

def collect_names(root=ROOT):
 root=Path(root);old,protocol,evaluation,expected=locked_inventory(root);names=set(expected)|{OLD_MANIFEST,'results/causal-feedback/protocol-lock.json','results/causal-feedback/evaluation-lock.json','results/causal-feedback/model-runtime-dependencies.json','examples/bio_bridge/replay_feedback.py','examples/bio_bridge/test_replay_feedback.py','scripts/package_feedback.py','scripts/test_package_feedback.py'}
 feedback=root/'results/causal-feedback'
 names.update(str(p.relative_to(root)) for p in feedback.rglob('*') if p.is_file() and str(p.relative_to(root))!=MANIFEST)
 optional=root/'results/actuator-noise'
 if optional.exists():names.update(str(p.relative_to(root)) for p in optional.rglob('*') if p.is_file() and p.suffix not in ('.so','.dylib','.dll'))
 hard=root/'results/hard-feedback-routing'
 if hard.exists():names.update(str(p.relative_to(root)) for p in hard.rglob('*') if p.is_file())
 predecode=root/'results/hard-feedback-routing-predecode-v0'
 if predecode.exists():names.update(str(p.relative_to(root)) for p in predecode.rglob('*') if p.is_file())
 for pattern in ('examples/bio_bridge/*FEEDBACK*.md','examples/bio_bridge/*feedback*.py','examples/bio_bridge/*causal*.py','examples/bio_bridge/*actuator_noise*.py','reports/**/*feedback*','results/causal-feedback/README*'):
  names.update(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
 queue=[name for name in names if name.endswith('.py')]
 while queue:
  name=queue.pop();tree=ast.parse(source_bytes(root,name))
  for node in ast.walk(tree):
   modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else [])
   for module in modules:
    local=(PurePosixPath(name).parent/(module.split('.')[0]+'.py')).as_posix()
    candidates=(local,f'examples/bio_bridge/{module.split(".")[0]}.py',f'scripts/{module.split(".")[0]}.py')
    for candidate in candidates:
     if (root/candidate).is_file() and candidate not in names:names.add(candidate);queue.append(candidate);break
 return names,old,protocol,evaluation,expected

def make_bundle(archive=DEFAULT_ARCHIVE):
 names,old,protocol,evaluation,expected=collect_names();payload={name:source_bytes(ROOT,name) for name in sorted(names)}
 for name,digest in expected.items():
  if name not in payload or sha(payload[name])!=digest:raise ValueError(f'frozen hash mismatch: {name}')
 if sum(map(len,payload.values()))>MAX_TOTAL:raise ValueError('bundle exceeds bounded size')
 runtime=json.loads(payload['results/causal-feedback/model-runtime-dependencies.json'])
 report={'schema_version':1,'scope':'Frozen bilateral, causal-feedback and hard-routing replay, with separate actuator-noise artifacts; no base Gemma weights or platform-specific native binary','bilateral_manifest_sha256':sha(payload[OLD_MANIFEST]),'causal_protocol_sha256':sha(payload['results/causal-feedback/protocol-lock.json']),'causal_evaluation_sha256':sha(payload['results/causal-feedback/evaluation-lock.json']),'model_runtime_dependencies_sha256':sha(payload['results/causal-feedback/model-runtime-dependencies.json']),'model_runtime_dependencies':runtime['models'],'hard_feedback_protocol_sha256':sha(payload['results/hard-feedback-routing/protocol-lock.json']) if 'results/hard-feedback-routing/protocol-lock.json' in payload else None,'hard_feedback_evaluation_sha256':sha(payload['results/hard-feedback-routing/evaluation-lock.json']) if 'results/hard-feedback-routing/evaluation-lock.json' in payload else None,'files':{name:{'sha256':sha(data),'bytes':len(data)} for name,data in payload.items()},'safetensors_count':sum(name.endswith('.safetensors') for name in payload),'base_models_required':protocol.get('models_required',json.loads(payload['results/bilateral-bridge/protocol-lock.json'])['models']),'prerequisites':'Python 3.11, Rust/Cargo toolchain, complete matching local Gemma directories including locked FunctionGemma chat_template.jinja, and Apple MPS for the original runtime.','setup':['python3.11 -m venv .venv','.venv/bin/python -m pip install -r requirements-gemma.txt -r requirements-functiongemma.txt scipy==1.17.1','.venv/bin/python -m pip install .','Keep chat_template.jinja in the FunctionGemma model root; replay rejects missing, changed, or extra .jinja templates','.venv/bin/python examples/bio_bridge/replay_feedback.py --embedding-model /local/embeddinggemma --function-model /local/functiongemma --include-hard-routing --check-only','.venv/bin/python examples/bio_bridge/replay_feedback.py --embedding-model /local/embeddinggemma --function-model /local/functiongemma --include-hard-routing'],'optional_actuator_noise_limitation':'Actuator-noise artifacts and source may be included, but strict replay can require the original native shared-library fingerprint. Platform native binaries are deliberately excluded, so that separate fingerprint must be supplied externally.','boundary':'Existing bilateral manifest is included unchanged. Base model weights, platform native binaries, this manifest, and archive hash are excluded from the hashed payload to avoid relocation/circularity. The Jinja dependency lock was added after outcomes when portable replay exposed the original suffix omission.'}
 data=(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode();archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=archive.parent,delete=False) as temp:temporary=Path(temp.name)
 try:
  with tarfile.open(temporary,'w:gz') as tar:
   for name,content in {**payload,MANIFEST:data}.items():info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(content))
  temporary.replace(archive)
 finally:temporary.unlink(missing_ok=True)
 (ROOT/MANIFEST).write_bytes(data);return {'archive':str(archive),'archive_sha256':sha(archive.read_bytes()),'files':len(payload),'safetensors':report['safetensors_count'],'bytes':archive.stat().st_size}

def validate_payload(payload):
 if MANIFEST not in payload:raise ValueError('missing feedback bundle manifest')
 report=json.loads(payload[MANIFEST]);files=report.get('files',{})
 if report.get('schema_version')!=1 or set(payload)!=(set(files)|{MANIFEST}):raise ValueError('manifest inventory mismatch')
 for name,record in files.items():
  safe_name(name)
  if len(payload[name])!=record['bytes'] or sha(payload[name])!=record['sha256']:raise ValueError(f'archive content mismatch: {name}')
 old=json.loads(payload[OLD_MANIFEST])
 if sha(payload[OLD_MANIFEST])!=report['bilateral_manifest_sha256']:raise ValueError('bilateral manifest identity mismatch')
 for name,record in old['files'].items():
  if name not in files or sha(payload[name])!=record['sha256']:raise ValueError('bilateral payload no longer matches original manifest')
 runtime_name='results/causal-feedback/model-runtime-dependencies.json'
 if runtime_name not in payload or sha(payload[runtime_name])!=report.get('model_runtime_dependencies_sha256'):raise ValueError('model runtime dependency lock identity mismatch')
 for lock,key,field in [('protocol-lock.json','frozen','causal_protocol_sha256'),('evaluation-lock.json',None,'causal_evaluation_sha256')]:
  name='results/causal-feedback/'+lock
  if sha(payload[name])!=report[field]:raise ValueError('causal lock identity mismatch')
  records=json.loads(payload[name]);records=records[key] if key else records
  for path,digest in records.items():
   if key is None:path='results/causal-feedback/'+path
   if path not in files or sha(payload[path])!=digest:raise ValueError('causal frozen content mismatch')
 if report.get('hard_feedback_protocol_sha256') is not None:
  for lock,key,field in [('protocol-lock.json','frozen','hard_feedback_protocol_sha256'),('evaluation-lock.json',None,'hard_feedback_evaluation_sha256')]:
   name='results/hard-feedback-routing/'+lock
   if name not in payload or sha(payload[name])!=report[field]:raise ValueError('hard-feedback lock identity mismatch')
   records=json.loads(payload[name]);records=records[key] if key else records
   for path,digest in records.items():
    if key is None:path='results/hard-feedback-routing/'+path
    if path not in files or sha(payload[path])!=digest:raise ValueError('hard-feedback frozen content mismatch')
 return report

def unpack(archive,destination):
 destination=Path(destination).absolute()
 if destination.exists():raise FileExistsError('unpack requires a new destination directory')
 payload={};total=0
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   name=safe_name(member.name)
   if not member.isfile() or name in payload or member.size<0 or member.size>MAX_MEMBER:raise ValueError('unsupported, duplicate, or oversized member')
   total+=member.size
   if total>MAX_TOTAL:raise ValueError('archive exceeds total limit')
   data=tar.extractfile(member).read(MAX_MEMBER+1)
   if len(data)!=member.size:raise ValueError('archive size mismatch')
   payload[name]=data
 validate_payload(payload);destination.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.feedback-unpack-',dir=destination.parent) as temporary:
  tree=Path(temporary)/'tree';tree.mkdir()
  for name,data in payload.items():path=tree/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  if destination.exists():raise FileExistsError('destination appeared during unpack')
  tree.rename(destination)
 return {'destination':str(destination),'verified_files':len(payload)-1}
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);pack=sub.add_parser('pack');pack.add_argument('--out',type=Path,default=DEFAULT_ARCHIVE);extract=sub.add_parser('unpack');extract.add_argument('archive',type=Path);extract.add_argument('destination',type=Path);a=p.parse_args();print(json.dumps(make_bundle(a.out) if a.command=='pack' else unpack(a.archive,a.destination),indent=2))
if __name__=='__main__':main()
