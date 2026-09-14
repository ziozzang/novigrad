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
