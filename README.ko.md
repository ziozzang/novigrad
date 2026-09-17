# Novigrad — Software-defined Bionic NPU

Novigrad는 공개 FlyWire 연결망을 회로 구조로 사용해 신호를 계산하고 스칼라 보상으로 기존 시냅스 가중치를 학습하는 CPU 연구 프로토타입이다. Rust crate와 실행 파일 이름은 `novi`이다. [English README](README.md).

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

이미지 어댑터는 학습 이미지에서만 HOG·픽셀 풀링·PCA를 맞춰 319개 포트의 입력률을 만든다. 여덟 설정을 비교한 결과, 출력 가중치 총량 정규화를 끄고 최대 75회 반복하며 KC 20%를 활성화하는 설정이 가장 좋았다. 기존 연결 구조·부호·개별 가중치 한도는 유지한다.

| 새 평가 데이터 | 기존 모델 | 개선 3 seed 평균 |
|---|---:|---:|
| 의류 8,000장 | 71.20% | **85.04%** |
| CAPTCHA 문자 4,000개 | 83.50% | **99.49%** |
| 네 자리 전체 1,000장 | 49.20% | **98.00%** |

동일한 새 평가 데이터에서 비교했다. [전체 실험 보고서](results/IMPROVEMENT_REPORT.ko.md), [개선 메커니즘](docs/optimization.ko.md), [이미지 실행 안내](examples/vision/README.ko.md)에 실패한 실험, 대조군, 원시 예측과 한계를 남겼다. 이미지 실험은 지도 학습이고, CAPTCHA는 고정된 셀과 공유된 글꼴 분포 안에서 검증했다.

[Hugging Face 조향 모방 예제](examples/steering/README.ko.md)는 같은 이미지 입력 포트로 시뮬레이터의 조향 명령을 예측한다. 서로 다른 주행 세션으로 오프라인 모방 성능을 확인하는 작은 예제다. 시험 정확도는 60%로 항상 직진하는 기준선58%와 차이가 작다. 30 FPS 재생에서는 M2 Ultra 평균2.77ms, 300회 deadline 초과0회를 기록했다. 실제 카메라·액추에이터 지연은 제외한다.

## API·주기 실행·지연 보상

- [로컬 HTTP API](docs/api.ko.md): 포트 메타데이터·추론·선택적 학습·체크포인트. `cargo build --release --features api --bins`로 빌드한다.
- [주기 실행 런타임](docs/runtime.ko.md): `novi_rt`로 실행 주기와 deadline 초과를 측정한다. Mac의 hard real-time 보장은 아니다.
- [지연 보상](docs/neuromodulation.ko.md): 보상 예측 오차와 과거 행동 재실행을 적용한다. [후각→가상 행동 27회 실험](results/NEUROMODULATION_REPORT.ko.md)에 동결·무관한 보상·규칙 전환 대조군을 포함했다. 앞다리 출력은 가상 행동 의미이며 실제 운동신경 경로를 재구성한 것은 아니다.

모든 인터페이스는 같은 엔진과 포트·Safetensors 모델을 사용한다. 감각 인코딩과 행동 해석·보상 출처는 외부 어댑터와 응용 프로그램이 맡는다.

## 폐루프 주행 게임

[HighwayEnv 예제](examples/driving-game/README.ko.md)는 정답 행동 없이 게임 보상으로 학습합니다. 독립 실험마다 같은 초기 가중치로 시작하고 검증·시험 중에는 학습하지 않습니다. 시험20게임에서 충돌15→1회로 줄었지만, 주로 감속을 배웠으며 항상 감속 규칙은 충돌0회로 더 좋았습니다. [전체 보고서](results/DRIVING_GAME_REPORT.ko.md)와 [게임 GIF](results/driving-game/driving.gif)에 결과와 한계를 기록했습니다. 저장된 모델은 `scripts/play_driving_game.py MODEL.safetensors --human`으로 게임 창에서 볼 수 있습니다.

## 세 정책의 투표

[MAGI형 군집 예제](examples/magi/README.ko.md)는 독립 학습한 세 회로의 다수결과 확률 평균을 비교합니다. 새 시험30게임의 충돌은 최상 단일 모델0회·다수결20회·확률 평균3회였습니다. 약한 두 정책이 같은 판단을 반복해 다수결이 나빠졌습니다. [전체 보고서](results/MAGI_REPORT.ko.md)에 판단 불일치·처리 시간과 한계를 남겼습니다.

## 범위

전체 파생 연결망의 LIF 활성화 실행기는 위 rate 학습 모델과 별개다. 각 회로는 CPU 스레드에서 실행하며 API는 모델별 잠금과 제한된 작업 풀을 사용한다. 또한 로컬 Cargo 설정은 `target-cpu=native`를 사용한다. M2 Ultra에서 고정 입력 forward와 보상 0의 측정치는 초당 9,732 episode이며 실제 보상 학습 처리량이나 로봇 제어 지연시간을 뜻하지 않는다.

실제 도파민 신호, 생물학적 시간 상수, 하드웨어 NPU, 전체 뇌 재학습, 보지 않은 XOR 조합에 대한 일반화는 입증하지 않았다. 연결 구조는 생물학 자료이지만 입력 인코딩·행동 매핑·보상·학습 규칙은 공학적으로 설계했다.

## 릴리스

[v0.1.1 다운로드](https://github.com/ziozzang/novigrad/releases/tag/v0.1.1): Apple Silicon 실행 파일, Safetensors 모델 묶음, SHA-256 체크섬. 코드는 MIT이며 데이터 및 외부 자료 권리는 [NOTICE](NOTICE.md)를 참고한다.

Author: jioh jung <jung@jioh.net> · License: MIT


Python 네이티브 바인딩과 선택적인 MLX/Metal 배치 추론: [예제](examples/python/README.ko.md), [성능 실험 및 미채택 최적화](results/PYTHON_PERFORMANCE_REPORT.ko.md).


Gemma 의미 연결: [실행 예제와 학습 결과](examples/gemma_bridge/README.ko.md), [뇌 영역·연결 위치 연구](examples/gemma_bridge/RESEARCH.ko.md).


FunctionGemma API·LoRA 호출 실험: [예제](examples/function_bridge/README.ko.md), [기억·단서 충돌·목표 변경 사례 연구](examples/function_bridge/CASES_RESEARCH.ko.md).

추가 연구: [목표 공간 연결·센서 신뢰도·보상 학습 메커니즘](examples/function_bridge/DEEP_RESULTS.ko.md). 실패 사례와 개선된 일반 정책 비교도 포함합니다.

생물학적 메커니즘 진단: [EmbeddingGemma 희소성, FunctionGemma 거절, 문맥 기억, 순차 지연 보상](examples/bio_bridge/README.ko.md)과 [일차 문헌 연구](examples/bio_bridge/RESEARCH.ko.md).

신호 타이밍과 세기: [지속 입력·단발 입력, 외부 기억, 보상 크기, 반복·간격 학습 비교](examples/bio_bridge/SIGNAL_RESULTS.ko.md).

초파리 메커니즘 추출·비교: [감각 적응, APL형 억제, 보상 흔적과 대조군 실험](examples/bio_bridge/MECHANISM_RESULTS.ko.md).

임베딩 기반 신경 상태 판독: [KC/MBON 의미 복원과 인과 대조 실험](examples/bio_bridge/THOUGHT_RESULTS.ko.md), [생물학적 판독 연구](examples/bio_bridge/THOUGHT_RESEARCH.ko.md).
