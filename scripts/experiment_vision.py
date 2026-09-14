#!/usr/bin/env python3
"""Run a named, validation-only pair of topology-preserving image experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--gain', type=float, default=12)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--active', type=float, default=.1)
    parser.add_argument('--homeostasis', choices=['true', 'false'], default='true')
    parser.add_argument('--components', type=int, default=64)
    parser.add_argument('--whitening', type=float, default=.5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error('name must be one path component')
    out = ROOT / 'results/improvement' / args.name
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args)
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    def run(task):
        original = 'fashion_mnist' if task == 'fashion' else task
        if args.components == 64 and args.whitening == .5:
            features = ROOT / f'data/vision/{task}64'
        else:
            features = ROOT / f'data/vision/experiments/{task}-pca{args.components}-white{args.whitening}'
            if not (features/'features.safetensors').exists():
                subprocess.run([sys.executable,'scripts/image_adapter.py',f'data/vision/{original}.npz',
                                '--out',str(features),'--components',str(args.components),
                                '--whitening',str(args.whitening)],cwd=ROOT,check=True)
        taskout = out/task
        command = [str(ROOT/'target/release/train_classifier'),str(features/'features.safetensors'),
                   'data/pn_kc.tsv','data/kc_mbon.tsv',str(taskout),'--phase','tune','--seed','1',
                   '--readout','opponent','--gain',str(args.gain),'--lr',str(args.lr),
                   '--active-fraction',str(args.active),'--homeostasis',args.homeostasis,
                   '--epochs',str(args.epochs),'--patience',str(args.patience)]
        if args.batch_size != 1:
            command.extend(['--batch-size',str(args.batch_size)])
        with (out/f'{task}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((taskout/'metrics.json').read_text())
        assert metrics['test_accuracy'] is None
        assert not (taskout/'predictions.tsv').exists()
        return task,metrics
    with ThreadPoolExecutor(max_workers=2) as pool:
        metrics=dict(pool.map(run,['fashion','captcha']))
    summary={task:metrics[task]['validation_accuracy'] for task in metrics}
    summary['mean_validation_accuracy']=sum(summary.values())/2
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'experiment':args.name,**summary}),flush=True)

if __name__ == '__main__':
    main()
