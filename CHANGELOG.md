# Novigrad v0.1.1 — Learning mechanisms, local API and periodic runtime

The executable and Rust crate are now **novi**. Author: **jioh jung <jung@jioh.net>**. Source license: **MIT**.

Eight controlled image experiments identified a stronger configuration without adding anatomical edges, changing synapse signs, or using pretrained CNN/OCR models. It disables output L1 normalization, trains for up to 75 epochs and retains 20% of hidden cells.

| Fresh evaluation metric | v0.1.0 model on the same data | Improved mean, 3 seeds |
|---|---:|---:|
| Clothing, 8,000 images | 71.20% | **85.04%** |
| CAPTCHA characters, 4,000 cells | 83.50% | **99.49%** |
| Exact four-digit strings, 1,000 images | 49.20% | **98.00%** |

These are clothing classification and synthetic fixed-cell CAPTCHA examples, not arbitrary scene detection or live CAPTCHA solving. Configuration selection used validation only; all additional training seeds and shuffled-label controls completed before fresh holdout evaluation. Reports include failed settings, learning curves, provenance, raw predictions and reproduction commands.

## New reusable capabilities

- Mean-gradient minibatches, remainder handling and explicit completed-batch checkpoint boundaries. Batch sizes 16/64 did not improve accuracy over their matched online controls; the feature remains optional.
- `novi_api`: optional local HTTP model metadata, batch inference, opt-in learning/checkpoints, per-model locking and bounded worker jobs. Neuron IDs are decimal strings.
- `novi_rt`: periodic fixed-port execution, reusable buffers, measured deadlines and skipped overdue slots. This is soft real-time under macOS, not a hard RTOS guarantee.
- Delayed reward modulation: bounded eligibility replay and reward-minus-baseline signals. A separate 27-run synthetic odor/virtual-action experiment includes frozen, unrelated-reward and reversal controls. Outputs are virtual foreleg commands, not reconstructed foreleg motor neurons.
- Existing Safetensors models load despite the executable rename; legacy serialized `nobi.*` identifiers are preserved.

## Assets and checks

The Apple Silicon archive contains `novi`, `novi_engine`, `novi_api`, `novi_rt`, `train_classifier`, `classify_features`, `train_connectome` and `odor_motor`. Binaries target Apple M1 instructions and were tested on M2 Ultra; they are unsigned and unnotarized.

The model archive contains original baseline bundles, optimized vision bundles under `examples/vision/models/{fashion,captcha}-optimized/`, logic models and one odor/action model. Extract at the source repository root. Raw datasets and font binaries are excluded. `SHA256SUMS` verifies both archives.

Validation: 34 Rust tests (including real HTTP integration), 8 Python tests, warning-free Clippy, old-model loading, independent NumPy/Rust inference, actual image HTTP/local equivalence and a periodic-runner smoke benchmark. The 1-ms/1,000-tick baseline-model run averaged about 116µs compute with zero measured deadline misses; it is one observation, not an OS scheduling guarantee.

[English documentation](https://github.com/ziozzang/novigrad/blob/v0.1.1/README.md) · [Full experiment report](https://github.com/ziozzang/novigrad/blob/v0.1.1/results/IMPROVEMENT_REPORT.md) · [한국어 문서](https://github.com/ziozzang/novigrad/blob/v0.1.1/README.ko.md)

한국어: 실행 파일 이름을 novi로 변경하고, 학습 개선 실험과 원시 결과를 문서화했습니다. 새 평가에서 의류 85.04%, CAPTCHA 문자 99.49%, 네 자리 전체 98.00%를 확인했습니다. 로컬 API, soft real-time 주기 실행, 지연 보상·가상 행동 예제를 함께 제공합니다.


# Novigrad v0.1.0 — Software-defined Bionic NPU

First public research release: a Mac-focused Rust engine with interchangeable input adapters, connectome-constrained plasticity, external action/reward interfaces, and self-contained Safetensors checkpoints.

- Real FlyWire subcircuit: 319 input neurons, 5,177 hidden neurons, 96 output neurons; 27,848 fixed and 62,261 plastic connections.
- Reward-learning validation: 72 binary/four-way/XOR runs including frozen/random-reward controls and rule reversal. Trained greedy accuracy is 100% on these synthetic tasks; see the report for the narrower test scope.
- Image examples: three-seed held-out means of **72.62% Fashion-MNIST classification**, **83.32% synthetic CAPTCHA character accuracy**, and **48.67% exact four-digit success**. These use supervised learning, a training-only HOG/PCA adapter, and an engineered opponent decoder. CAPTCHA segmentation is fixed; arbitrary live CAPTCHA solving is not established.
- Safetensors schema 4 stores topology, weights, and output gains; schema 3 checkpoints remain readable. Both vision checkpoints pass independent NumPy/Rust inference comparison with maximum probability error below 5e-7.
- Validation: 17 Rust unit tests, 3 CLI integration tests, 4 Python tests, and warning-free Clippy.

## Assets

- `novigrad-v0.1.0-aarch64-apple-darwin.tar.gz`: CPU binaries built for Apple M1 instruction support, tested on M2 Ultra. These binaries are not signed/notarized. Source builds default to native CPU optimization.
- `novigrad-v0.1.0-models.tar.gz`: seed-1 vision adapter/model bundles plus binary/four-way/XOR circuit models. Extract at the source repository root. Raw datasets and font binaries are not included.
- `SHA256SUMS`: archive integrity hashes.

The crate and CLI names remain `nobi` for compatibility. The engine is a rate-model research prototype, not hardware NPU execution or whole-brain biological simulation. Source code is MIT; upstream data rights and attributions are described in `NOTICE.md`.

[English documentation](https://github.com/ziozzang/novigrad/blob/v0.1.0/README.md) · [한국어 문서](https://github.com/ziozzang/novigrad/blob/v0.1.0/README.ko.md)

한국어: Software-defined Bionic NPU의 첫 공개 연구 릴리스입니다. 실제 연결망의 학습과 Safetensors 저장·복원을 검증하고, 의류 분류 및 합성 CAPTCHA 예제와 원시 평가 결과를 함께 공개합니다. 이미지 결과는 지도 학습이며, CAPTCHA는 네 셀의 위치가 고정된 범위입니다.
