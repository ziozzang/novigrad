# Novigrad — Software-defined Bionic NPU

Novigrad는 공개 FlyWire 연결망을 회로 구조로 사용해 신호를 계산하고 스칼라 보상으로 기존 시냅스 가중치를 학습하는 CPU 연구 프로토타입이다. Rust crate와 실행 파일 이름은 현재 `nobi`이다. [English README](README.md).

학습 회로는 **ALPN 입력 319개 → KC 5,177개 → MBON 출력 96개**다. ALPN→KC 연결 27,848개는 고정하고 KC→MBON 연결 62,261개의 가중치를 갱신한다. 새 연결은 만들지 않는다. 전체 초파리 뇌를 학습한 결과는 아니다.

## 빌드와 재현

Apple Silicon Mac과 Rust가 필요하다.

```sh
git clone https://github.com/ziozzang/novigrad.git
cd novigrad
cargo build --release --bins
cargo test --release
python3 -m venv .venv
.venv/bin/pip install -r requirements-data.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/prepare_training.py
cargo run --release --bin train_connectome -- \
  --task all --seed-start 101 --seeds 8 \
  --episodes 2000 --reversal-episodes 2000 --eval 1024 \
  --lr 0.02 --gain 12 --out results/v1-confirmation
```

최종 확인은 튜닝에 쓰지 않은 seed 101–108의 세 과제와 정상·0·무작위 보상 조건, 총 72회다. 학습 후 greedy 정확도는 이진·4분류·XOR 모두 100%였고, 정답에 부여한 평균 확률은 각각 99.55%, 98.29%, 89.98%였다. 보상 0 및 무작위 보상 대조군은 대체로 우연 수준이었다. 규칙 전환 후 재학습과 잡음 평가, 개별 결과 및 한계는 [v1 보고서](results/V1_REPORT.md)에 있다. 입력과 보상은 합성 자료다.

## 범용 엔진과 이미지 예제

엔진은 임의의 입력·출력 ID, 두 연결 파일, 행동 수를 받는다. 외부 실행기가 감각 인코딩, 행동 선택, 보상을 제공한다. [영문 README의 CLI 예제](README.md#generic-engine), [구조](docs/architecture.md), [Safetensors 형식](docs/model-format.md)을 참고한다.

이미지 예제는 학습 분할에서만 비지도 HOG·픽셀 풀링·PCA 어댑터를 맞춰 이미지를 319개 입력 포트로 변환한다. Fashion-MNIST는 학습/검증/시험 10,000/2,000/2,000장이고, 별도 로컬 생성 네 자리 CAPTCHA는 전체 이미지 3,000/500/500장과 네 출력 셀을 사용한다. 분류기는 지도 teacher-gradient 방식과 보상 방식을 지원한다. 공개 이미지 실험은 지도 학습과 섞인 라벨 대조군을 평가한다. **세 seed 평균 시험 정확도는 의류 72.62%, CAPTCHA 문자 83.32%, 네 자리 모두 정답 48.67%다.** 고정 가중치와 라벨 섞기 대조군은 약 10% 수준이다. 일반 장면 탐지나 임의의 웹 CAPTCHA 해결을 검증한 것은 아니다. [실행 안내](examples/vision/README.md), [결과 보고서](results/VISION_REPORT.md), [자료 출처](data/README.md)를 참고한다.

## 범위

전체 파생 연결망의 LIF 활성화 실행기는 위 rate 학습 모델과 별개다. 현재 단일 스레드 CPU 구현이며 로컬 Cargo 설정은 `target-cpu=native`를 사용한다. M2 Ultra에서 고정 입력 forward와 보상 0의 측정치는 초당 9,732 episode이며 실제 보상 학습 처리량이나 로봇 제어 지연시간을 뜻하지 않는다.

실제 도파민 신호, 생물학적 시간 상수, 하드웨어 NPU, 전체 뇌 재학습, 보지 않은 XOR 조합에 대한 일반화는 입증하지 않았다. 연결 구조는 생물학 자료이지만 입력 인코딩·행동 매핑·보상·학습 규칙은 공학적으로 설계했다.

## 릴리스

[v0.1.0 다운로드](https://github.com/ziozzang/novigrad/releases/tag/v0.1.0): Apple Silicon 실행 파일, Safetensors 모델 묶음, SHA-256 체크섬. 코드는 MIT이며 데이터 및 외부 자료 권리는 [NOTICE](NOTICE.md)를 참고한다.
