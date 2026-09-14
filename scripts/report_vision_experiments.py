#!/usr/bin/env python3
"""Report validation ablations and weight mass; never load held-out datasets."""
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]

def weight_stats(path):
    with safe_open(str(path),framework='numpy') as f:
        w=f.get_tensor('plastic_weight')
        post=f.get_tensor('plastic_post').astype(np.int64)
        limit=float(f.metadata()['weight_limit'])
    mass=np.bincount(post,weights=np.abs(w))
    return dict(mean_output_l1=float(mass.mean()),max_output_l1=float(mass.max()),
                fraction_at_zero=float(np.mean(w==0)),fraction_at_limit=float(np.mean(np.abs(w)>=limit-1e-6)))

def main():
    rows=[]
    root=ROOT/'results/improvement'
    for config_path in sorted(root.glob('*/config.json')):
        directory=config_path.parent
        if not (directory/'summary.json').exists():continue
        config=json.loads(config_path.read_text())
        for task in ('fashion','captcha'):
            metrics=json.loads((directory/task/'metrics.json').read_text())
            rows.append(dict(experiment=directory.name,task=task,config=config,
                             validation_accuracy=metrics['validation_accuracy'],
                             validation_cross_entropy=metrics['validation_cross_entropy'],
                             validation_correct_probability=metrics['validation_correct_probability'],
                             best_epoch=metrics['best_epoch'],
                             **weight_stats(directory/task/'model.safetensors')))
    (root/'ablation-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Validation mechanism experiments','',
           'All rows use training seed 1 and the same training/validation splits. Test data is not evaluated by this report. Accuracy and checkpoint selection use validation only. This is a mechanism comparison, not an independent final accuracy claim.','',
           '| Experiment | Task | Accuracy | Cross entropy | True-class probability | Mean output L1 | Zero weights | Best epoch |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['experiment']} | {r['task']} | {r['validation_accuracy']:.2%} | {r['validation_cross_entropy']:.3f} | {r['validation_correct_probability']:.2%} | {r['mean_output_l1']:.2f} | {r['fraction_at_zero']:.1%} | {r['best_epoch']} |")
    lines+=['','See each named directory for configuration, complete epoch curves, and metrics; [JSON](ablation-summary.json) also includes maximum output L1 and fraction of weights at their individual limit.','']
    (root/'ABLATIONS.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
