"""Layered experiment contract: freeze, verify, evaluate, and replay.

The lock is an integrity record, not an adversarial security boundary. Intentional
new research must use a new output directory/protocol rather than rewrite a lock.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
import precise_bridge as bridge
from inhibition_mechanism import sha256

ROOT=bridge.ROOT
OUT=bridge.OUT


def write(path,payload):path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')

def check_hashes(hashes,root=ROOT):
    for relative,expected in hashes.items():
        path=root/relative
        if not path.is_file() or sha256(path)!=expected:raise ValueError(f'locked artifact changed: {relative}')

def compare_tensors(left,right):
    a,b=load_file(str(left)),load_file(str(right))
    if a.keys()!=b.keys():raise AssertionError('tensor keys changed')
    for key in a:
        if not np.array_equal(a[key],b[key]):raise AssertionError(f'tensor mismatch: {key}')
    return True

def freeze():
    lock=OUT/'protocol-lock.json'
    if lock.exists():raise FileExistsError('protocol already frozen; verify or replay it')
    required=[OUT/'development.json',OUT/'native-development.json',OUT/'permutation-development.json',ROOT/'examples/bio_bridge/precise_holdout.json']
    for p in required:
        if not p.exists():raise FileNotFoundError(p)
    files=required+list(OUT.rglob('*.safetensors'))+list((ROOT/'examples/bio_bridge').glob('*precise*.py'))
    files += [ROOT/'examples/bio_bridge'/p for p in ['thought_embedding.py','inhibition_mechanism.py','delayed_credit.py']]
    files += [ROOT/'results/gemma-bridge'/p for p in ['dataset.json','embeddings.safetensors']]
    files += [ROOT/'results/thought-bridge'/p for p in ['description-embeddings.safetensors']]
    files += [ROOT/'examples/bio_bridge/thought_descriptions.json',bridge.checkpoint(601)]
    payload={'schema_version':1,'layers':{'data':'train32 / validation12 / authored final64 (32 paired scenarios)','encoder':'frozen EmbeddingGemma300M FP32; Classification prompt; normalized768','ports':'fit train only; fixed MRL128 / centered MRL128 / PCA rank<=31 padded128; signed256 into319','circuit':'fixed PN-KC connectome, top2%; semantic ridge or independently reward-trained native KC-MBON','selection':'ridge .01/.1/1 on validation accuracy then cosine; all3 mapping results reported','evaluation':'single held-out pass; no refit or selection; language and pair dependency reported','replay':'fresh semantic fit tensor equality and fresh native training equality; saved model inference'},'environment':{'platform':platform.platform(),'python':platform.python_version(),'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','safetensors','torch','sentence-transformers']}},'sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))},'limitations':['Authored cases are not external data or biological measurements.','Development sets were reused across earlier research.','PCA fitting uses only32 rows; insufficient for universal alignment.','Native and semantic decoders optimize different targets.']}
    write(lock,payload);print('Protocol frozen:',sha256(lock))

def verify():
    lock=OUT/'protocol-lock.json';payload=json.loads(lock.read_text());check_hashes(payload['sha256'])
    p=OUT/'evaluation-lock.json'
    if p.exists():check_hashes(json.loads(p.read_text())['sha256'])
    return sha256(lock)

def evaluate():
    lock_hash=verify()
    if (OUT/'evaluation-lock.json').exists() or (OUT/'final.json').exists():raise FileExistsError('final evaluation already exists; use replay instead')
    provenance=json.loads((OUT/'provenance.json').read_text())
    if provenance['cases_sha256']!=sha256(ROOT/'examples/bio_bridge/precise_holdout.json'):raise ValueError('holdout cases mismatch')
    if provenance['embedding_sha256']!=sha256(OUT/'holdout-embeddings.safetensors'):raise ValueError('holdout features mismatch')
    bridge.final()
    import precise_native
    precise_native.run_final(OUT)
    import precise_permutation
    precise_permutation.run_final(OUT)
    files=[OUT/p for p in ['final.json','native-final.json','permutation-final.json','holdout-embeddings.safetensors','provenance.json']]
    write(OUT/'evaluation-lock.json',{'protocol_sha256':lock_hash,'sha256':{str(p.relative_to(ROOT)):sha256(p) for p in files}})

def replay():
    verify()
    with tempfile.TemporaryDirectory(prefix='novi-precise-replay-') as directory:
        temporary=Path(directory);old=bridge.OUT
        try:
            bridge.OUT=temporary
            bridge.development(bridge.MODES)
        finally:bridge.OUT=old
        for mode in bridge.MODES:
            for kind in ('ports','decoder'):compare_tensors(OUT/f'{kind}-{mode}.safetensors',temporary/f'{kind}-{mode}.safetensors')
        expected=json.loads((OUT/'development.json').read_text())['modes']
        actual=json.loads((temporary/'development.json').read_text())['modes']
        if expected!=actual:raise AssertionError('semantic development results differ')
    # Recompute final predictions in a temporary output folder with frozen artifacts.
    final_equal=None
    if (OUT/'final.json').exists():
        import shutil
        with tempfile.TemporaryDirectory(prefix='novi-precise-eval-') as directory:
            temporary=Path(directory)
            for p in OUT.iterdir():
                if p.suffix=='.safetensors' or p.name=='development.json':shutil.copy2(p,temporary/p.name)
            old=bridge.OUT
            try:bridge.OUT=temporary;bridge.final()
            finally:bridge.OUT=old
            expected=json.loads((OUT/'final.json').read_text())['results'];actual=json.loads((temporary/'final.json').read_text())['results']
            for collection in (expected,actual):
                for r in collection.values():r.pop('batch_bridge_seconds',None)
            if expected!=actual:raise AssertionError('semantic final predictions differ')
            final_equal=True
    report={'semantic_refit_exact':True,'six_tensor_artifacts_exact':True,'saved_final_predictions_exact':final_equal,'protocol_sha256':sha256(OUT/'protocol-lock.json')}
    write(OUT/'semantic-replay.json',report);print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','verify','evaluate','replay']);args=parser.parse_args()
    result=globals()[args.phase]()
    if result:print(result)
