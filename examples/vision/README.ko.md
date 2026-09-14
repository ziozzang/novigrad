# Bionic NPU 이미지 예제

[English](README.md) · [결과 보고서](../../results/VISION_REPORT.ko.md)

동일한 Rust 회로로 의류 10종 분류와 로컬 생성 네 자리 CAPTCHA의 숫자 인식을 학습한다. 세 seed 평균 시험 정확도는 의류 72.62%, 숫자 83.32%, 네 자리 전체 정답률 48.67%다. 전처리 어댑터가 입력을 319개 포트의 발화율로 변환하며, 기존 연결의 가중치를 지도 학습한다.

저장소 루트에서 실행한다.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-vision.txt
cargo build --release --bins
```

[릴리스](https://github.com/ziozzang/novigrad/releases/tag/v0.1.0)에서 `novigrad-v0.1.0-models.tar.gz`를 내려받고 루트에 풀면 원본 데이터 다운로드 없이 추론할 수 있다.

```sh
tar -xzf novigrad-v0.1.0-models.tar.gz
.venv/bin/python scripts/predict_image.py examples/vision/models/fashion/bundle.json examples/vision/assets/fashion-example.png
.venv/bin/python scripts/predict_image.py examples/vision/models/captcha/bundle.json examples/vision/assets/captcha-example.png
```

의류 예제의 정답은 `Shirt`지만 모델은 `T-shirt/top`으로 오답을 낸다. CAPTCHA는 `5420`을 맞힌다. 둘 다 시험 분할의 첫 이미지이며 성공 사례만 고르지 않았다. 의류는 28×28로 변환하고, CAPTCHA는 112×28에서 위치가 고정된 네 셀을 나눈다. 밝은 바탕이라면 `--invert`를 사용한다.

전체 재학습과 집계:

```sh
.venv/bin/python scripts/run_vision_examples.py
.venv/bin/python scripts/report_vision.py
```

데이터 생성은 기본적으로 Mac의 SFNSMono·Geneva 글꼴을 사용한다. 글꼴 파일은 배포하지 않는다. 다른 글꼴을 사용하면 다른 실험이며, 상세 옵션은 `scripts/prepare_vision.py --help`를 참고한다.

어댑터는 학습 이미지에서만 HOG·픽셀 풀링·PCA를 맞추며 라벨을 보지 않는다. 범용 엔진에는 숫자 입력률만 전달한다. 다른 입력 형식도 같은 포트 계약으로 어댑터를 구현할 수 있다. 이미지 결과는 지도 학습 검증이며, 보상만으로 이미지 과제를 해결했다고 주장하지 않는다. 일반 사진의 오브젝트 탐지나 실제 웹사이트의 임의 CAPTCHA 해결도 검증 범위가 아니다. 텐서 계약과 재현 절차는 [영문 안내](README.md)를 참고한다.
