# 보상으로 배우는 작은 주행 게임

[HighwayEnv](https://highway-env.farama.org/)를 기존 Novigrad 엔진과 연결했습니다. 차선 변경·가속·감속 명령이 실제 다음 차량 상태를 바꾸는 폐루프입니다. 정답 행동 없이 자기 확률로 행동을 탐색하고, 게임의 누적 보상으로 학습합니다.

첫 버전은 카메라 대신 주변 차량의 위치·속도 등5×5 수치 관측을319개 입력률로 변환합니다. 저수준 조향과 속도 제어는 시뮬레이터가 수행하며 엔진은 다섯 고수준 행동을 선택합니다.

“목적을 스스로 알아내기”는 **외부 점수를 높이는 행동을 탐색해서 발견한다**는 범위입니다. 보상 규칙은 사람이 환경에 정의합니다. 점수도 없이 임의의 게임 목적을 이해하거나 스스로 유용한 목적을 발명한 것은 아닙니다.

시험 게임20개에서 충돌은 초기 모델15회→학습 모델1회, 평균 점수는22.20→27.95로 개선됐습니다. 다만 배운 행동은 주로 감속·유지였고, 추가로 비교한 항상 감속 규칙은 충돌0회·점수28.67로 더 좋았습니다. 복잡한 회피 운전까지 배웠다고 해석하지 않습니다. [전체 보고서](../../results/DRIVING_GAME_REPORT.ko.md)에 대조군과 한계를 기록했습니다.

각 학습률 실험과 무관한 보상 대조군은 같은 초기 가중치에서 독립적으로 시작합니다. 에피소드마다 게임만 초기화하고 학습 가중치는 유지합니다.4에피소드마다 누적 보상 기울기를 한 번 반영합니다. 검증·시험에서는 가중치를 변경하지 않습니다.

```sh
.venv/bin/python -m pip install -r requirements-simulation.txt
cargo build --release --features api --bin novi_engine --bin novi_api
.venv/bin/python scripts/train_driving_game.py --out results/driving-game-reproduction
```

이미 준비한 FlyWire 연결 데이터가 필요합니다. 기존 결과를 덮어쓰지 않도록 새 출력 디렉터리를 사용합니다. 모델은 Safetensors로 저장되며 학습 기록과 테스트별 결과·GIF를 남깁니다.

저장된 정책으로 게임 창을 열려면:

```sh
.venv/bin/python scripts/play_driving_game.py \
  results/driving-game-reproduction/model.safetensors --human
```

창 없이 GIF로 보려면 `--human` 대신 `--gif /tmp/novi-driving.gif`를 사용합니다. 외부 카메라나 실제 차량에 연결하지 않습니다. [예시 GIF](../../results/driving-game/driving.gif)와 [영문 설명](README.md)도 제공합니다.
