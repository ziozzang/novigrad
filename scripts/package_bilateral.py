#!/usr/bin/env python3
"""Pack or safely unpack the frozen bilateral replay sources and small artifacts.

Base Gemma weights are excluded. Native bindings are supplied as portable Rust
and Python sources, not an architecture-specific shared library.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/bilateral-bridge'
MANIFEST = 'results/bilateral-bridge/bundle-manifest.json'
MAX_MEMBER = 64*1024**2
MAX_TOTAL = 256*1024**2


def safe_name(name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or PurePosixPath(name).is_absolute() or any(p in ('','.','..') for p in name.split('/')):
        raise ValueError(f'unsafe archive path: {name}')
    return name


def sha(data):return hashlib.sha256(data).hexdigest()


def source_bytes(root,name):
    safe_name(name)
    root=Path(root).resolve();path=root/name
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f'nonlocal or missing input: {name}')
    if path.stat().st_size>MAX_MEMBER:raise ValueError(f'oversized input: {name}')
    return path.read_bytes()


def make_bundle(archive):
    protocol=json.loads((OUT/'protocol-lock.json').read_text())
    evaluation=json.loads((OUT/'evaluation-lock.json').read_text())
    expected={}
    for records in (protocol['frozen'],evaluation['hashes']):
        for name,digest in records.items():
            if name in expected and expected[name]!=digest:raise ValueError('conflicting frozen hashes')
            expected[name]=digest
    names=set(expected)|{'results/bilateral-bridge/protocol-lock.json','results/bilateral-bridge/evaluation-lock.json',
        'examples/bio_bridge/replay_bilateral.py','scripts/package_bilateral.py','scripts/test_package_bilateral.py',
        'examples/bio_bridge/test_replay_bilateral.py','Cargo.toml','Cargo.lock','pyproject.toml','LICENSE','examples/python/README.md',
        '.cargo/config.toml','results/bilateral-bridge/README.md','results/bilateral-bridge/README.ko.md',
        'results/bilateral-bridge/posthoc-analysis.json','examples/bio_bridge/analyze_bilateral.py',
        'reports/neural-link/bilateral.html','reports/neural-link/bilateral.ko.html',
        'reports/neural-link/build_bilateral_report.py',
        'examples/bio_bridge/BILATERAL_RESEARCH.md','examples/bio_bridge/BILATERAL_RESEARCH.ko.md',
        'examples/bio_bridge/PAIRED_DATA_RESEARCH.md','examples/bio_bridge/PAIRED_DATA_RESEARCH.ko.md'}
    for pattern in ('src/**/*.rs','bindings/python/**/*.rs','bindings/python/**/*.toml','bindings/python/**/*.lock',
                    'bindings/python/python/**/*.py','bindings/python/python/**/*.pyi','bindings/python/python/**/py.typed','requirements*.txt'):
        names.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file())
    verification=OUT/'verification.json'
    if verification.exists():names.add(str(verification.relative_to(ROOT)))
    # Local Python imports transitively used by the locked helpers must ship too.
    import ast
    queue=[name for name in names if name.endswith('.py')]
    while queue:
        name=queue.pop()
        tree=ast.parse(source_bytes(ROOT,name))
        for node in ast.walk(tree):
            modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                     [alias.name for alias in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                candidate=str(PurePosixPath(name).parent/(module.split('.')[0]+'.py'))
                if (ROOT/candidate).is_file() and candidate not in names:
                    names.add(candidate);queue.append(candidate)
    payload={name:source_bytes(ROOT,name) for name in sorted(names)}
    for name,digest in expected.items():
        if sha(payload[name])!=digest:raise ValueError(f'frozen hash mismatch: {name}')
    if sum(map(len,payload.values()))>MAX_TOTAL:raise ValueError('bundle exceeds bounded size')
    report={'schema_version':1,'scope':'Frozen bilateral replay; no base Gemma weights or platform-specific native binary',
        'protocol_sha256':sha(payload['results/bilateral-bridge/protocol-lock.json']),
        'evaluation_sha256':sha(payload['results/bilateral-bridge/evaluation-lock.json']),
        'files':{name:{'sha256':sha(data),'bytes':len(data)} for name,data in payload.items()},
        'safetensors_count':sum(name.endswith('.safetensors') for name in payload),
        'base_models_required':protocol['models'],
        'prerequisites':'Python 3.11, Rust/Cargo build toolchain, Apple MPS hardware for original runtime; obtain matching base models separately.',
        'setup':['python3.11 -m venv .venv','.venv/bin/python -m pip install -r requirements-gemma.txt -r requirements-functiongemma.txt scipy==1.17.1',
                 '.venv/bin/python -m pip install .',
                 '.venv/bin/python examples/bio_bridge/replay_bilateral.py --embedding-model /local/embeddinggemma --function-model /local/functiongemma'],
        'boundary':'Model contents must match original hashes. Inference replay preserves MPS/BF16 and original tolerances; other hardware is not guaranteed. Manifest excludes itself and archive hash to avoid circularity.'}
    data=(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode()
    archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent,delete=False) as temp:temporary=Path(temp.name)
    try:
        with tarfile.open(temporary,'w:gz') as tar:
            for name,content in {**payload,MANIFEST:data}.items():
                info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(content))
        temporary.replace(archive)
    finally:temporary.unlink(missing_ok=True)
    (ROOT/MANIFEST).write_bytes(data)
    return {'archive':str(archive),'archive_sha256':sha(archive.read_bytes()),'files':len(payload),'safetensors':report['safetensors_count'],'bytes':archive.stat().st_size}


def unpack(archive,destination):
    destination=Path(destination).absolute()
    if destination.exists():raise FileExistsError('unpack requires a new destination directory')
    payload={};total=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            name=safe_name(member.name)
            if not member.isfile() or name in payload or member.size>MAX_MEMBER or member.size<0:
                raise ValueError('unsupported, duplicate, or oversized member')
            total+=member.size
            if total>MAX_TOTAL:raise ValueError('archive exceeds total limit')
            content=tar.extractfile(member).read(MAX_MEMBER+1)
            if len(content)!=member.size:raise ValueError('archive size mismatch')
            payload[name]=content
    if MANIFEST not in payload:raise ValueError('missing bundle manifest')
    report=json.loads(payload[MANIFEST]);files=report['files']
    if report.get('schema_version')!=1 or set(payload)!=(set(files)|{MANIFEST}):raise ValueError('manifest inventory mismatch')
    for name,record in files.items():
        safe_name(name)
        if len(payload[name])!=record['bytes'] or sha(payload[name])!=record['sha256']:raise ValueError(f'archive content mismatch: {name}')
    for name,field in [('protocol-lock.json','protocol_sha256'),('evaluation-lock.json','evaluation_sha256')]:
        if sha(payload['results/bilateral-bridge/'+name])!=report[field]:raise ValueError('lock identity mismatch')
    for lock,key in [('protocol-lock.json','frozen'),('evaluation-lock.json','hashes')]:
        for name,digest in json.loads(payload['results/bilateral-bridge/'+lock])[key].items():
            if name not in files or sha(payload[name])!=digest:raise ValueError('frozen lock content mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bilateral-unpack-',dir=destination.parent) as temporary:
        tree=Path(temporary)/'tree';tree.mkdir()
        for name,data in payload.items():
            path=tree/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        if destination.exists():raise FileExistsError('destination appeared during unpack')
        tree.rename(destination)
    return {'destination':str(destination),'verified_files':len(files)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    pack=sub.add_parser('pack');pack.add_argument('--out',type=Path,default=ROOT/'dist/novigrad-bilateral-20260917.tar.gz')
    extract=sub.add_parser('unpack');extract.add_argument('archive',type=Path);extract.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(json.dumps(make_bundle(args.out) if args.command=='pack' else unpack(args.archive,args.destination),indent=2))


if __name__=='__main__':main()
