#!/usr/bin/env python3
"""Aggregate fixed-seed held-out predictions, including whole CAPTCHA accuracy."""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def main():
    records = []
    for task in ('fashion', 'captcha'):
        for mode, seed in [('supervised', s) for s in (1, 2, 3)] + [('shuffled', 1)]:
            directory = ROOT / f'results/vision/{task}-{mode}-{seed}'
            metrics = json.loads((directory / 'metrics.json').read_text())
            with (directory / 'predictions.tsv').open() as source:
                predictions = list(csv.DictReader(source, delimiter='\t'))
            correct = [int(p['true']) == int(p['predicted']) for p in predictions]
            assert len(correct) == 2000
            accuracy = sum(correct) / len(correct)
            assert abs(accuracy - metrics['test_accuracy']) < 1e-8
            exact = sum(all(correct[i:i+4]) for i in range(0, len(correct), 4)) / 500 if task == 'captcha' else None
            records.append(dict(task=task, mode=mode, seed=seed, test_accuracy=accuracy,
                                exact_sequence_accuracy=exact, baseline_test_accuracy=metrics['baseline_test_accuracy'],
                                best_epoch=metrics['best_epoch'], changed_weights=metrics['changed_plastic_weights'],
                                elapsed_seconds=metrics['elapsed_seconds']))
    summary = {}
    for task in ('fashion', 'captcha'):
        runs = [r for r in records if r['task'] == task and r['mode'] == 'supervised']
        values = [r['test_accuracy'] for r in runs]
        summary[task] = dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))
        if task == 'captcha':
            summary[task]['exact_mean'] = statistics.mean(r['exact_sequence_accuracy'] for r in runs)
    (ROOT/'results/vision-summary.json').write_text(json.dumps(dict(summary=summary, runs=records), indent=2)+'\n')
    lines = ['# Vision validation — v0.1.0', '', '[한국어 요약](VISION_REPORT.ko.md)', '',
             'These are held-out test results from the actual Rust connectome-constrained engine, using supervised teacher gradients. The image adapter is label-free and fitted only on training images. No pretrained CNN or OCR model is used.', '',
             '| Task | Training mode | Seed | Test accuracy | Exact 4-digit success | Frozen accuracy | Best epoch |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        exact = '—' if r['exact_sequence_accuracy'] is None else f"{r['exact_sequence_accuracy']:.2%}"
        lines.append(f"| {r['task']} | {r['mode']} | {r['seed']} | {r['test_accuracy']:.2%} | {exact} | {r['baseline_test_accuracy']:.2%} | {r['best_epoch']} |")
    lines += ['', f"Across three seeds, Fashion-MNIST accuracy averages **{summary['fashion']['mean']:.2%}**; CAPTCHA character accuracy averages **{summary['captcha']['mean']:.2%}**, and whole-string accuracy averages **{summary['captcha']['exact_mean']:.2%}**. These seeds share the same data split, so the range measures training variability, not uncertainty across datasets.", '',
              '## Protocol', '',
              '- Fashion-MNIST: balanced 10,000 training, 2,000 validation, and 2,000 official-test images; 10 clothing categories. Training and validation source indices are disjoint.',
              '- Synthetic CAPTCHA: 3,000 training, 500 validation, 500 test four-digit images; full strings are unique across splits. Each 112×28 image is split into four known 28×28 cells, giving 12,000/2,000/2,000 character samples. All four predictions must match for exact-string success.',
              '- Exact train/test image duplicate count: zero for both datasets. Fonts and noise distribution are shared across CAPTCHA splits; this does not test unseen fonts or unknown segmentation.',
              '- Adapter: 637 HOG/pooled-pixel features, training-only standardization and 64-component PCA, fractional whitening 0.5, nonnegative signed-component encoding into 319 actual ALPN IDs.',
              '- Circuit: 27,848 fixed ALPN→KC edges and 62,261 plastic KC→MBON edges, 10% hidden activity, gain 12, learning rate 0.0001. A fixed external opponent decoder assigns +1/−1 gains to output ports; it does not change biological synapse signs.',
              '- Up to 25 epochs; patience 5; validation selects the checkpoint. Seeds 1, 2, 3 were fixed before test evaluation. Packaged models are seed 1, chosen before seeing test results.',
              '- Shuffled control permutes training labels once and uses the same validation selection. The frozen baseline uses initial weights. The Fashion shuffled checkpoint selects epoch 0 because later updates do not improve validation.',
              '- Tuning used validation only: 159 fully whitened components, then lower learning rate, 64 partially whitened components, and opponent decoding. Test was evaluated only after this configuration was fixed.',
              '', '## Evidence and limits', '',
              'Per-run `metrics.json`, `curves.csv`, and `predictions.tsv` are in [vision/](vision/). [Machine-readable summary](vision-summary.json), [dataset provenance](vision-data-manifest.json), and [independent NumPy/Rust checkpoint verification](vision-safetensors-verification.json) accompany this report. Run `python scripts/report_vision.py` to regenerate the summary from predictions.', '',
              'Each supervised run changes 62,260–62,261 plastic weights. Accuracy is well above frozen and shuffled-label controls, demonstrating learning in this circuit. Probabilities remain relatively soft; argmax accuracy is not a claim of calibrated confidence. This is a clothing-image classification example and a locally generated fixed-cell digit-recognition example, not general object detection or a solver for arbitrary live CAPTCHA systems.', '',
              'The bundled examples are predetermined test sample 0: Fashion truth `Shirt`, predicted `T-shirt/top` (incorrect); CAPTCHA truth `5420`, predicted `5420` (correct). They are retained to show both success and failure.', '',
              'Measured on an Apple M2 Ultra, CPU only. Two task pipelines ran concurrently; elapsed times in metrics are descriptive, not isolated throughput benchmarks. See [reproduction instructions](../examples/vision/README.md).', '']
    (ROOT/'results/VISION_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
