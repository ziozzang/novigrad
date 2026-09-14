# MAGI처럼 세 정책의 투표로 판단하기

독립적으로 학습한 Novigrad 회로 세 개에 같은 관측을 주고, 행동을 투표로 결정합니다. 허구의 세 부분 의사결정 시스템에서 착안한 공학적 앙상블이며 생물학적 초파리 군집 지능을 재현하는 것은 아닙니다.

세 모델은 동일한 초기 가중치에서 시작하지만 행동 탐색 RNG와 학습 게임 seed를 다르게 사용합니다. 목적·보상·구조·학습률은 같습니다. 검증 게임으로 각 체크포인트를 고른 뒤 새로운 시험 게임에서는 모두 동결합니다.

- **다수결:** 각 모델이 고른 행동 중 가장 많은 표를 받은 행동을 선택합니다. 셋 다 다르면 동률 행동들의 평균 확률로 결정하고, 그것도 같으면 작은 행동 번호를 사용합니다.
- **확률 평균:** 세 확률 분포를 평균해 가장 큰 행동을 고릅니다. 확신이 강한 한 모델이 약한 두 선호를 넘어설 수 있어 다수결과 다릅니다.

각 단일 모델, 검증에서 선택한 최상 단일 모델, 두 투표 방식, 항상 감속 기준선을 같은 시험 seed로 비교합니다. 불일치율은 각 투표 게임의 동일 관측에서 측정합니다. 서로 다른 정책이 만든 서로 다른 주행 상태를 같은 관측처럼 비교하지 않습니다.

```sh
.venv/bin/python -m pip install -r requirements-simulation.txt
cargo build --release --features api --bin novi_engine --bin novi_api
.venv/bin/python scripts/train_magi.py --out results/magi-reproduction
```

각 모델128에피소드, 총384에피소드로 단일 모델보다 학습 비용이3배입니다. 학습·검증·시험 seed를 분리했으며 이전 주행 실험의 seed와도 겹치지 않습니다. 모델들은 별도 Safetensors로 보존하고 `bundle.json`이 구성원·행동 의미·인코딩·투표 규칙을 기록합니다.

여러 모델이 같은 실수를 하면 만장일치도 틀릴 수 있습니다. 같은 모델을 세 번 복제해도 새로운 판단 근거가 생기지는 않습니다. 이번 실험은 투표가 실제로 개선되는지와, 학습 seed 차이가 얼마나 다른 판단을 만드는지 확인합니다. [영문 설명](README.md)에 세부 규칙을 기록했습니다.

저장된 군집을 게임 창이나 GIF로 보려면:

```sh
.venv/bin/python scripts/play_magi.py results/magi-reproduction/bundle.json --human
.venv/bin/python scripts/play_magi.py results/magi-reproduction/bundle.json \
  --mode mean --gif /tmp/novi-magi.gif
```

API 프로세스 세 개에 모델을 계속 유지하며 현재는 구성원별 추론을 순서대로 호출합니다. 매 판단마다 체크포인트를 다시 읽지는 않습니다. 이번에는 구성원별 역할이나 보상 목적을 다르게 주지 않았으며, 학습 경험과 탐색의 차이만 사용했습니다.

측정 결과는 최상 단일 모델 충돌0/30, 다수결20/30, 확률 평균3/30입니다. 두 약한 모델이 동일한 판단을 반복해 다수결을 지배했습니다. [전체 보고서](../../results/MAGI_REPORT.ko.md)에 결과와 한계를 기록했습니다.
