#!/usr/bin/env python3
"""Summarize preselected models on the new holdout and package seed1 bundles."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation',type=Path,default=ROOT/'results/improvement/confirmation')
    args=parser.parse_args()
    directory=args.confirmation.resolve()
    selection=json.loads((directory/'selection.json').read_text())
    final={}
    for task in ('fashion','captcha'):
        read=lambda name:json.loads((directory/f'eval-{task}-{name}/metrics.json').read_text())
        baseline=read('baseline');runs=[read(f'supervised-{seed}') for seed in (1,2,3)]
        control=read('shuffled-1')
        accuracy=[r['accuracy'] for r in runs];exact=[r['exact_sequence_accuracy'] for r in runs]
        final[task]={'selection':selection[task],'samples':baseline['samples'],
                     'baseline_accuracy':baseline['accuracy'],'baseline_exact':baseline['exact_sequence_accuracy'],
                     'accuracy_mean':statistics.mean(accuracy),'accuracy_min':min(accuracy),'accuracy_max':max(accuracy),
                     'exact_mean':statistics.mean(exact),'exact_min':min(exact),'exact_max':max(exact),
                     'cross_entropy_mean':statistics.mean(r['cross_entropy'] for r in runs),
                     'baseline_cross_entropy':baseline['cross_entropy'],
                     'shuffled_accuracy':control['accuracy'],'shuffled_exact':control['exact_sequence_accuracy'],
                     'model_bytes':runs[0]['model_bytes'],'adapter_bytes':runs[0]['adapter_bytes'],
                     'per_seed_accuracy':accuracy,'per_seed_exact':exact}
        bundle=ROOT/f'examples/vision/models/{task}-optimized'
        bundle.mkdir(parents=True,exist_ok=True)
        for name in ('bundle.json','adapter.safetensors','model.safetensors'):
            shutil.copy2(directory/f'{task}-supervised-1'/name,bundle/name)
    (ROOT/'results/improvement/final-summary.json').write_text(json.dumps(final,indent=2)+'\n')
    f,c=final['fashion'],final['captcha']
    lines=['# Novigrad v0.1.1: learning mechanisms and fresh holdout results','',
           '[한국어 보고서](IMPROVEMENT_REPORT.ko.md)','',
           'The same anatomical input/plastic edges now support substantially stronger image recognition. No pretrained CNN/OCR or new dense layer was added. The selected configuration disables per-output L1 normalization, trains for up to 75 epochs with validation early stopping, and retains 20% of hidden neurons. Synapse signs and individual weight bounds remain enforced. The engine default remains unchanged; these are explicit task configurations.','',
           '## Final comparison on previously unused data','',
           '| Metric | Released v0.1.0 model, same new holdout | Improved mean, 3 seeds | Improved seed range | Shuffled-label control |',
           '|---|---:|---:|---:|---:|',
           f"| Clothing accuracy ({f['samples']:,} images) | {f['baseline_accuracy']:.2%} | **{f['accuracy_mean']:.2%}** | {f['accuracy_min']:.2%}–{f['accuracy_max']:.2%} | {f['shuffled_accuracy']:.2%} |",
           f"| CAPTCHA character accuracy ({c['samples']*4:,} cells) | {c['baseline_accuracy']:.2%} | **{c['accuracy_mean']:.2%}** | {c['accuracy_min']:.2%}–{c['accuracy_max']:.2%} | {c['shuffled_accuracy']:.2%} |",
           f"| Exact 4-digit success ({c['samples']:,} images) | {c['baseline_exact']:.2%} | **{c['exact_mean']:.2%}** | {c['exact_min']:.2%}–{c['exact_max']:.2%} | {c['shuffled_exact']:.2%} |",'',
           'The baseline column evaluates the packaged v0.1.0 seed-1 model on these same new images. It is not the old published three-seed average on the earlier 2,000/500-image tests. Improved models use seeds 1, 2, 3 on one fixed split; their range describes optimizer/order variability, not uncertainty across different datasets. Seed 1 is packaged regardless of test score.','',
           f"Mean test cross-entropy changes from {f['baseline_cross_entropy']:.3f} to {f['cross_entropy_mean']:.3f} for clothing, and {c['baseline_cross_entropy']:.3f} to {c['cross_entropy_mean']:.3f} for digits. Better argmax accuracy and loss do not establish calibrated confidence or robustness to distribution shifts.",'',
           '## Eight controlled configurations','',
           'Every tuning row used seed 1 and validation only. Each configuration was applied to both tasks; `--phase tune` left test metrics null and wrote no test predictions. [Full ablations](improvement/ABLATIONS.md) include cross-entropy, true-class probabilities, selected epochs, zero-weight fractions and output L1 mass. [Machine-readable configurations and statistics](improvement/ablation-summary.json) preserve every result, including failures.','',
           '| Change | Fashion validation | CAPTCHA validation | Interpretation |',
           '|---|---:|---:|---|',
           '| Original 25-epoch setting | 74.65% | 84.20% | Baseline reproduced exactly |',
           '| Disable output L1 normalization | 86.70% | 97.95% | Largest single tested improvement |',
           '| Keep normalization, gain 12→48, inverse-scale learning rate | 85.75% | 95.55% | Evidence scale and resulting gradients matter too |',
           '| No L1 normalization, up to 75 epochs | 87.85% | 99.00% | Additional passes help the undertrained model |',
           '| Retain 20% instead of 10% hidden cells, 75 epochs | 88.00% | 99.30% | Selected for both tasks |',
           '| PCA 96 instead of 64, top 10%, 75 epochs | 87.60% | 98.50% | More dimensions did not help |',
           '| Batch 16, lr 0.0016, top 10%, 75 epochs | 87.85% | 99.00% | Same accuracy as matched online training |',
           '| Batch 64, lr 0.0064, top 20%, 75 epochs | 87.95% | 99.15% | Slightly below matched online training |','',
           '![Recorded validation learning curves](improvement/learning-curves.svg)','',
           'The 25-epoch no-normalization run and its longer version share the same initial trajectory; the orange marker identifies the shorter run. These are validation curves, not fresh test curves.','',
           '## Why these changes matter','',
           'With per-output normalization, each MBON has an incoming absolute weight budget of 1. After an online update exceeds that budget, all incoming weights are divided by their total, creating competition among previously and currently useful connections. Removing that rescaling permits a larger and nonuniform evidence budget while still clipping each existing edge to its original sign and magnitude bound. On seed-1 validation checkpoints, mean output L1 grows from about 0.99 to 5.56/6.62 after 25 epochs (clothing/digits), accompanied by much stronger true-class probabilities. This is an engineering result, not proof about physiological homeostasis.','',
           'Gain alone cannot change the argmax of a fixed trained model. Here the gain control retrained the network: even with learning_rate × gain held constant, softmax changes the error term and therefore the gradient. Its improvement shows that output scale is part of the bottleneck. The experiment does not uniquely separate all effects of scale, normalization and optimization.','',
           'Extra epochs help because the no-normalization learning curves are still improving at epoch 25. Validation selects the checkpoint, so a later worse epoch is not forced into the model. Retaining more hidden cells gives a smaller additional benefit. More PCA dimensions simultaneously alter compression and how components repeat across 319 ports, so their negative result is not evidence that dimensionality is universally harmful.','',
           'Minibatch accumulation is kept as a reusable library/CLI capability, not advertised as an accuracy win. Batch size 1 keeps the original online path; larger batches average gradients and apply projection once. Learning rates in these experiments were explicitly scaled to roughly preserve update magnitude per epoch. Other batch/rate combinations remain untested.','',
           '## Data and selection discipline','',
           '- Training data remains 10,000 clothing images and 3,000 four-digit CAPTCHA images, with the original 2,000-image and 500-string validation sets.',
           '- The new Fashion holdout contains the remaining 8,000 official test indices excluded from the earlier release evaluation. CAPTCHA has 1,000 new strings excluded from all original train/validation/test strings, generated with a separate seed.',
           '- Exact train/holdout image duplicates: zero in both datasets. The CAPTCHA generator still shares fonts, noise distribution and fixed cell positions across splits.',
           '- PCA/feature scaling fit only training images. No target labels enter the adapter or backend inference payload.',
           '- The eight-setting search fixed [selection.json](improvement/selection.json) before any new holdout prediction. Confirmation then completed all three seeds and the shuffled-label controls before unsealing evaluation.',
           '- [Holdout provenance](improvement/holdout-manifest.json), [duplicate audit](improvement/image-overlap-audit.json), [final metrics](improvement/final-summary.json), and [per-sample predictions](improvement/confirmation/) make the comparison inspectable. Future tuning should treat this holdout as observed.', '',
           '## Reusable architecture and execution','',
           'The implementation adds generic mean-gradient batches, an optional [local HTTP API](../docs/api.md), a [soft real-time periodic runtime](../docs/runtime.md), and [delayed reward modulation](../docs/neuromodulation.md). They share the same port IDs, action decoder and Safetensors circuit. Current executable/crate names are `novi` and `novi_engine`; legacy serialized `nobi.*` format identifiers remain readable.','',
           'The API uses per-model locking and bounded blocking workers; IDs are decimal strings to preserve U64 precision. A real image passed through the API matches local binary inference ([smoke result](improvement/api-image-smoke.json)). The runtime has preallocated per-tick buffers and reports compute/scheduled deadline misses separately. A measured M2 Ultra 1-ms-period run of 1,000 baseline-model ticks averaged about 116 microseconds compute with zero deadline misses; all starts were technically later than their ideal slot. This one observation does not guarantee deadlines under macOS ([raw timing](improvement/runtime-benchmark.json)).','',
           'The [27-run delayed-reward report](NEUROMODULATION_REPORT.md) evaluates synthetic odor cues and virtual foreleg commands with normal, absent and unrelated rewards plus reversal. It is a separate reward-learning experiment; the image results above use supervised labels. Actual DAN compartment dynamics and a reconstructed foreleg motor path are not implemented.','',
           '## Reproduction and verification','',
           'See [mechanism commands and API semantics](../docs/optimization.md), [image examples](../examples/vision/README.md), and the saved configuration JSON files. `scripts/report_vision_experiments.py`, `scripts/report_vision_improvement.py`, and `scripts/plot_vision_experiments.py` regenerate the reports and plot from recorded data. Plotting dependencies are isolated in `requirements-reports.txt`.', '',
           'Validation covers Rust unit/CLI/HTTP tests, independent Python checkpoint inference, malformed data, minibatch remainder handling, checkpoint boundaries, delayed reward direction, scheduler arithmetic and old-model loading. Source code is MIT, authored by jioh jung <jung@jioh.net>; upstream source rights remain documented in NOTICE.', '']
    (ROOT/'results/IMPROVEMENT_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(final,indent=2))

if __name__=='__main__':main()
