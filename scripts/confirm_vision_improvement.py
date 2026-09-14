#!/usr/bin/env python3
"""Confirm a previously selected configuration; never select by holdout scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection',type=Path,help='JSON mapping fashion/captcha to experiment names, fixed before evaluation')
    parser.add_argument('--out',type=Path,default=ROOT/'results/improvement/confirmation')
    args=parser.parse_args()
    selection=json.loads(args.selection.read_text())
    final=args.out.resolve()
    if final.exists():raise ValueError('Confirmation exists; refuse accidental repeated holdout evaluation')
    final.mkdir(parents=True)
    shutil.copy2(args.selection,final/'selection.json')
    def run(task):
        experiment=ROOT/'results/improvement'/selection[task]
        cfg=json.loads((experiment/'config.json').read_text())
        if cfg['components']==64 and cfg['whitening']==.5:
            feature=ROOT/f'data/vision/{task}64'
        else:
            feature=ROOT/f"data/vision/experiments/{task}-pca{cfg['components']}-white{cfg['whitening']}"
        for mode,seed in [('supervised',1),('supervised',2),('supervised',3),('shuffled',1)]:
            out=final/f'{task}-{mode}-{seed}'
            if mode=='supervised' and seed==1:
                out.mkdir()
                for name in ('model.safetensors','metrics.json','curves.csv'):
                    shutil.copy2(experiment/task/name,out/name)
            else:
                command=[str(ROOT/'target/release/train_classifier'),str(feature/'features.safetensors'),
                         'data/pn_kc.tsv','data/kc_mbon.tsv',str(out),'--phase','tune','--mode',mode,
                         '--seed',str(seed),'--readout','opponent','--gain',str(cfg['gain']),
                         '--lr',str(cfg['lr']),'--active-fraction',str(cfg['active']),
                         '--homeostasis',cfg['homeostasis'],'--epochs',str(cfg['epochs']),
                         '--patience',str(cfg['patience']),'--batch-size',str(cfg.get('batch_size',1))]
                with (final/f'{task}-{mode}-{seed}.log').open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            bundle=json.loads((ROOT/f'examples/vision/models/{task}/bundle.json').read_text())
            bundle.update(components=cfg['components'],whitening=cfg['whitening'],
                          training_configuration=cfg,training_seed=seed,training_mode=mode)
            shutil.copy2(feature/'adapter.safetensors',out/'adapter.safetensors')
            (out/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
        return task
    # Finish every training run before unsealing any holdout.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,['fashion','captcha']))
    for task in ('fashion','captcha'):
        bundles=[('baseline',ROOT/f'examples/vision/models/{task}/bundle.json')]
        bundles += [(f'{mode}-{seed}',final/f'{task}-{mode}-{seed}/bundle.json')
                    for mode,seed in [('supervised',1),('supervised',2),('supervised',3),('shuffled',1)]]
        for name,bundle in bundles:
            out=final/f'eval-{task}-{name}'
            command=[sys.executable,'scripts/evaluate_image_model.py',str(bundle),
                     f'data/vision/holdout_{task}.npz','--out',str(out)]
            with (final/f'eval-{task}-{name}.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics=json.loads((out/'metrics.json').read_text())
            print(json.dumps({'task':task,'run':name,'accuracy':metrics['accuracy'],
                              'exact':metrics['exact_sequence_accuracy']}),flush=True)
    print('Confirmation complete; configuration was fixed before holdout evaluation.')

if __name__=='__main__':main()
