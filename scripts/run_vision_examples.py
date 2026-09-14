#!/usr/bin/env python3
"""Reproduce the published fixed image experiment; test is never used to tune."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
LABELS=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


def call(args):
    print('+ '+' '.join(str(x) for x in args),flush=True)
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)


def bundle(task,features,run_dir,out):
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(features/'adapter.safetensors',out/'adapter.safetensors')
    shutil.copy2(run_dir/'model.safetensors',out/'model.safetensors')
    metadata=json.loads((features/'bundle.json').read_text())
    metadata['class_labels']=LABELS if task=='fashion' else [str(i) for i in range(10)]
    metadata['components']=64;metadata['whitening']=.5
    metadata['readout']='opponent'
    (out/'bundle.json').write_text(json.dumps(metadata,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',default='1,2,3')
    parser.add_argument('--skip-prepare',action='store_true')
    parser.add_argument('--task',choices=['fashion','captcha','all'],default='all')
    args=parser.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('seeds must be nonempty and unique')
    if not args.skip_prepare:
        if not (ROOT/'data/pn_kc.tsv').exists():
            call([sys.executable,'scripts/prepare_data.py'])
            call([sys.executable,'scripts/prepare_training.py'])
        call([sys.executable,'scripts/prepare_vision.py'])
        for task,npz in [('fashion','fashion_mnist'),('captcha','captcha')]:
            call([sys.executable,'scripts/image_adapter.py',f'data/vision/{npz}.npz','--out',f'data/vision/{task}64','--components','64','--whitening','.5'])
    call(['cargo','build','--release','--bins'])
    tasks=['fashion','captcha'] if args.task=='all' else [args.task]
    for task in tasks:
        features=ROOT/f'data/vision/{task}64'
        for mode,seed in [('supervised',seed) for seed in seeds]+[('shuffled',seeds[0])]:
            run_dir=ROOT/f'results/vision/{task}-{mode}-{seed}'
            call([ROOT/'target/release/train_classifier',features/'features.safetensors','data/pn_kc.tsv','data/kc_mbon.tsv',run_dir,'--epochs','25','--patience','5','--lr','0.0001','--gain','12','--readout','opponent','--phase','full','--mode',mode,'--seed',str(seed)])
        bundle(task,features,ROOT/f'results/vision/{task}-supervised-{seeds[0]}',ROOT/f'examples/vision/models/{task}')
    print('Finished. Models are in examples/vision/models; metrics and predictions are in results/vision.')


if __name__=='__main__':main()
