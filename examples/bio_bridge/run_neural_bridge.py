"""Run isolated embedding-to-neural experiments through explicit lifecycle phases."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import precise_bridge as bridge
import precise_native as native
import precise_permutation as permutation
import precise_pipeline as pipeline
from inhibition_mechanism import sha256

@contextmanager
def experiment_directory(path):
    path=Path(path).resolve()
    if not path.is_relative_to(bridge.ROOT):raise ValueError('run directory must be inside the repository for portable manifest paths')
    original=(bridge.OUT,pipeline.OUT)
    bridge.OUT=pipeline.OUT=path
    try:yield path
    finally:bridge.OUT,pipeline.OUT=original

def validate_model(model):
    provenance=json.loads((bridge.ROOT/'results/thought-bridge/description-provenance.json').read_text())
    for relative,expected in provenance['weight_sha256'].items():
        path=Path(model)/relative
        if not path.is_file() or sha256(path)!=expected:raise ValueError(f'model revision differs from frozen training features: {relative}')

def run(phase,out,model):
    if phase in ('develop','encode','all'):validate_model(model)
    with experiment_directory(out) as directory:
        if phase in ('develop','all'):
            if directory.exists() and any(directory.iterdir()):raise FileExistsError('development needs a fresh empty run directory')
            directory.mkdir(parents=True,exist_ok=True)
            bridge.development(bridge.MODES);native.run_development(directory);permutation.run_development(directory)
            (directory/'run-config.json').write_text(json.dumps({'schema_version':1,'model':str(Path(model).resolve()),'run_directory':str(directory.relative_to(bridge.ROOT)),'protocol':'precise-bridge-v1','stages':['develop','freeze','encode','evaluate','verify','replay'],'scope':'frozen text embeddings and simulated fly-derived rates'},indent=2)+'\n')
        if phase in ('freeze','all'):pipeline.freeze()
        if phase in ('encode','all'):
            pipeline.verify()
            subprocess.run([sys.executable,str(Path(__file__).with_name('encode_precise_holdout.py')),'--out',str(directory),'--model',str(model)],check=True,stdout=subprocess.DEVNULL)
        if phase in ('evaluate','all'):pipeline.evaluate()
        if phase in ('verify','all'):pipeline.verify()
        if phase in ('replay','all'):
            pipeline.replay();native.replay(directory)
        return {'phase':phase,'run_directory':str(directory.relative_to(bridge.ROOT)),'passed':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['develop','freeze','encode','evaluate','verify','replay','all']);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model',type=Path,default=Path('/Users/a405394/models/google_embeddinggemma-300m'));a=p.parse_args()
    print(json.dumps(run(a.phase,a.run_dir,a.model)))
