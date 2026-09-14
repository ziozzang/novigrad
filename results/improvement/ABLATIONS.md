# Validation mechanism experiments

All rows use training seed 1 and the same training/validation splits. Test data is not evaluated by this report. Accuracy and checkpoint selection use validation only. This is a mechanism comparison, not an independent final accuracy claim.

| Experiment | Task | Accuracy | Cross entropy | True-class probability | Mean output L1 | Zero weights | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| active20 | fashion | 88.00% | 0.358 | 79.58% | 7.21 | 5.2% | 58 |
| active20 | captcha | 99.30% | 0.063 | 94.61% | 8.61 | 6.1% | 75 |
| baseline | fashion | 74.65% | 1.481 | 24.54% | 0.99 | 18.1% | 12 |
| baseline | captcha | 84.20% | 1.620 | 20.67% | 0.99 | 14.9% | 6 |
| batch16 | fashion | 87.85% | 0.365 | 79.06% | 8.06 | 6.9% | 74 |
| batch16 | captcha | 99.00% | 0.096 | 92.11% | 9.13 | 10.3% | 75 |
| batch64 | fashion | 87.95% | 0.352 | 80.07% | 7.65 | 5.7% | 69 |
| batch64 | captcha | 99.15% | 0.101 | 91.55% | 7.26 | 10.1% | 38 |
| gain48 | fashion | 85.75% | 0.588 | 63.33% | 0.97 | 13.0% | 25 |
| gain48 | captcha | 95.55% | 0.549 | 61.50% | 0.96 | 13.0% | 15 |
| longer75 | fashion | 87.85% | 0.365 | 79.06% | 8.06 | 6.6% | 74 |
| longer75 | captcha | 99.00% | 0.096 | 92.11% | 9.13 | 9.5% | 75 |
| no-homeostasis | fashion | 86.70% | 0.438 | 73.72% | 5.56 | 9.1% | 25 |
| no-homeostasis | captcha | 97.95% | 0.202 | 84.11% | 6.62 | 10.0% | 25 |
| pca96 | fashion | 87.60% | 0.377 | 78.27% | 8.10 | 5.3% | 75 |
| pca96 | captcha | 98.50% | 0.127 | 89.97% | 9.22 | 7.9% | 75 |

See each named directory for configuration, complete epoch curves, and metrics; [JSON](ablation-summary.json) also includes maximum output L1 and fraction of weights at their individual limit.
