# Neural interface research report

The [biological robotics supplement](robotics.html) ([한국어](robotics.ko.html)) combines primary-source research, independent read/write adapter design, a replayed 1,280-case typed host-token control diagnostic, and a separate FlyGym CPU physics smoke test. Rebuild with `build_robotics_report.py` using `requirements-robotics.txt`; verify offline layout with `verify_robotics_report.py`. Numerical token tests do not execute an embedding model or LM.

The [causal-feedback supplement](feedback.html) ([한국어](feedback.ko.html)) adds 84 frozen feedback conditions, matched host controls, a separate actuator count-noise assay, and post-hoc one-prototype routing with retained counterexamples. [Methods and portable replay](../../results/causal-feedback/README.md). Rebuild with `.venv/bin/python reports/neural-link/build_feedback_report.py`; browser checks use `verify_feedback_report.py`.

The [bilateral interface supplement](bilateral.html) ([한국어](bilateral.ko.html)) tests typed site/time pooling and actual frozen FunctionGemma soft-prefix conditioning, including 120/480-update comparisons, held-out generated calls, constant/shuffled controls, wrong fixed points, and checkpoint compatibility. [Full methods and replay](../../results/bilateral-bridge/README.md). Rebuild with `.venv/bin/python reports/neural-link/build_bilateral_report.py`.

The new [architecture supplement](architecture.html) ([한국어](architecture.ko.html)) covers context routing, readout sites, fixed landmark pooling, low-rank/full direct bridge updates and adversarial counterexamples. It is a separate frozen study; it does not revise the earlier manuscript's measurements. Rebuild with `.venv/bin/python reports/neural-link/build_architecture_report.py`. [Methods, limitations and replay](../../results/architecture-bridge/README.md).

Open **[report.html](report.html)** for the English manuscript draft or **[report.ko.html](report.ko.html)** for Korean. Each file embeds its figures, styles, JavaScript, and case data and works offline. Source-paper/code links need a network connection; the language switch needs the neighboring HTML file.

The report covers train-only embedding calibration, independent native reward learning, port-placement controls, the separate recording-drift study, schema-validated recording import, and exact experiment reproduction. It distinguishes task accuracy from information preservation and explicitly reports negative results and missing controls. It is a pre-peer-review draft, not evidence of biological neural decoding or a submission-ready clinical study.

## Contents and interaction

- 15 numbered/main and supplementary sections, 4 original vector figures, 9 references.
- All 64 final cases; filter by mapping, language, correctness, or text.
- Download the embedded report data as JSON.
- Print styles remove navigation and controls, and include every case even when a screen filter is active. Journal-specific pagination and author declarations still need editorial review.
- `report-manifest.json` pins the report inputs and HTML outputs. `browser-verification.json` records desktop/mobile/offline/filter/download/print-media checks.

## Rebuild

From the repository root, using the existing local experiment artifacts and NumPy/Matplotlib environment:

```bash
.venv/bin/python reports/neural-link/build_report.py
```

The builder recomputes reported category/native accuracies from stored predictions and checks experiment-lock and drift-source hashes. It does not rerun models. The scientific source snapshot is commit `2d05f48a9c0dbc8c1928d52cbeaf72182dc677b1`.

For browser QA, install Playwright in a separate environment if needed, then use an installed Chrome executable:

```bash
python reports/neural-link/verify_report.py \
  --browser '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
```

Screenshots are written to `/tmp/novigrad-report-qa` by default, outside the repository. Browser QA validates print-media CSS, not journal-specific pagination.

## 한국어 안내

[한국어 보고서](report.ko.html)는 초록·생물학적 근거·방법·결과·대조군·음성 결과·재현 절차·한계와 제출 전 보강 과제를 포함합니다. 인터넷 연결 없이 열리며 그림과 사례 데이터가 파일 내부에 있습니다. 인쇄 시 전체 사례가 포함됩니다. 브라우저의 **인쇄 / PDF 저장** 기능을 사용할 수 있으며, 실제 저널 제출 전에는 저자 검토와 저널 양식에 맞춘 편집이 필요합니다.
