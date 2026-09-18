# Novi를 위한 범위가 제한된 로보틱스 브리지

이 문서는 실험을 제안할 뿐, 새로운 실행 결과나 학습된 언어 모델, FlyGym 결과를 보고하지 않는다. 현재 Novi는 실제 연결에서 가져온 단순화된 PN→KC→MBON rate 회로를 구현한다. 중앙복합체(CX), 하행뉴런(DN) 경로, 배쪽신경삭, 근육, 생물학적 감각운동 폐루프는 구현하지 않았다. 따라서 소프트웨어 변수에 파리 기능의 이름을 붙였다는 이유로 빠진 계통을 구현했다고 암시하면 안 된다.

## 생물학에서 측정한 것과 그것이 보장하지 않는 것

### 현재 방향과 목표 방향

[Seelig와 Jayaraman (2015)](https://www.nature.com/articles/nature14446)은 구속된 파리의 중앙복합체 나침반 뉴런에서 동물의 현재 방향을 따라 회전하고 시각·자기운동 단서로 갱신되는 activity bump를 측정했다. 이는 시험 조건에서 동적으로 유지되는 방향 변수를 지지한다. 임의의 임베딩 좌표가 나침반 뉴런이라는 뜻은 아니다.

[Mussells Pires 등 (2024)](https://www.nature.com/articles/s41586-023-07006-3)은 목표 방향 표현을 현재 방향과 구분하고 FC2/PFL3 회로를 방향 행동과 연결했다. 따라서 **원하는 방향**과 **관측한 방향**을 서로 다른 typed variable로 유지해야 한다. 이 목표는 항법 방향이었으며 제한 없는 자연어 의도가 아니었다.

### 학습 가치, 내부 상태, 행동 문맥

[Aso 등 (2014)](https://elifesciences.org/articles/04580)은 MBON 집단 activity와 인과 조작을 학습된 양·음의 valence 및 행동 선택과 연결했다. 이는 가치에 민감한 버섯체 출력 단계를 지지한다. MBON이 일반 언어 의미를 담거나 관절 torque를 직접 지정한다는 결과가 아니다.

[Krashes 등 (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2780032/)은 배고픔 상태가 확인된 neuromodulatory mechanism을 통해 appetitive memory의 인출·표현을 gate함을 보였다. 로보틱스 대응물은 내부 상태를 명시적 입력으로 노출하고 기억 의존 정책 변화가 있는지 시험할 수 있다. `hunger`라는 software scalar는 여전히 공학적 추상화이며 측정된 호르몬 회로가 아니다.

[Kim 등 (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6327952/)은 초파리 visuomotor processing에서 efference copy의 세포 수준 근거를 제시했다. 확인된 시각 뉴런은 saccade 중 예상되는 시각 운동을 상쇄하기에 적합한 timing과 sign의 motor-related input을 받았다. 따라서 command의 예상 감각 결과, 외부 시각 신호의 residual, 실제 관측된 운동을 서로 분리해야 한다. 명령은 관측이 아니다. 미끄러짐, 충돌, actuator 고장으로 둘은 달라질 수 있다.

### 고유감각과 하행 제어

[Mamiya 등 (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/)은 tibia 위치, 운동 방향과 속도, 또는 진동에 선택적인 femoral chordotonal organ(FeCO) 고유감각 뉴런 subclass를 규명했다. 공학적으로는 서로 다른 고유감각 양을 하나의 일반적인 body-state scalar로 합치지 않고, motor command와 분리하여 timestamp 및 uncertainty와 함께 공급해야 한다.

[Namiki 등 (2018)](https://elifesciences.org/articles/34272)은 뇌 영역을 배쪽신경삭에 잇는 하행뉴런을 지도화했고, [Cande 등 (2018)](https://elifesciences.org/articles/34275)은 하행뉴런 활성화로 개별·조합 경로와 행동의 관계를 조사했다. 이는 단어 하나를 motor 하나에 대응시키기보다 다대다 하행 제어 interface를 지지한다. 고수준 목표에서 locomotor command로 가는 완전한 역함수를 제공하지는 않는다.

이 결과들은 typed state, 기억·가치 조절, 방향 비교, 고유감각 feedback, 하행 action interface에 영감을 준다. 현재 Novi PN→KC→MBON 회로를 CX 또는 DN 모델로 검증하지 않는다.

## 실제 양방향 인터페이스의 선례

[Flesher 등 (Science, 2021)](https://pubmed.ncbi.nlm.nih.gov/34016775/?dopt=Abstract)은 운동피질 신호로 로봇 팔을 제어하는 읽기 경로와, 체성감각피질 자극으로 촉각 피드백을 전달하는 쓰기 경로를 결합했다. 연구한 참가자에서 촉각 피드백은 과제 완료 시간을 개선했다. 여기서 가져올 공학적 원리는 역할이 다른 읽기·쓰기 경로를 각각 보정하고 실제 과제 결과로 평가한다는 것이다. 이것이 파리–언어모델 연결을 입증하거나 소프트웨어 prefix를 신경 자극과 동등하게 만들지는 않는다. 제안하는 소프트웨어도 행동의 실제 효과와 환경에서 되돌아오는 정보의 기여를 모두 검증해야 한다.

## 세 가지 token interface

Interface 선택에 따라 무엇을 학습해야 하는지가 달라진다.

1. **Typed text.** 목표, 관측 상태, uncertainty, 허용 tool로 이루어진 작은 schema를 일반 text token으로 직렬화한다. 언어 모델 tokenizer를 바꾸지 않고 감사할 수 있지만 길고 metric precision을 잃을 수 있다. Text는 측정을 기술해야 하며 정답 action이나 미래 outcome을 몰래 포함해서는 안 된다.
2. **연속 soft prefix, `m × d_model`.** 학습 bridge가 고정 길이 state window를 모델 embedding stream에 넣을 `m`개의 floating-point vector로 바꾼다. 이 vector는 token ID, 단어, 뉴런이 아니다. Perceiver 계열 latent query, [Flamingo](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf), [BLIP-2](https://openreview.net/pdf?id=KU9UojoX7U)는 비텍스트 입력을 동결 또는 부분 동결 언어 모델에 압축하는 방법을 보여 주지만 vector를 유용하게 만드는 paired objective는 여전히 필요하다.
3. **학습된 discrete token.** 상태나 action chunk를 양자화하고 token ID를 확장하거나 재사용한다. [RT-2](https://arxiv.org/abs/2307.15818)와 [OpenVLA](https://openvla.github.io/)는 robot trajectory로 vision-language-action 모델을 학습하며, [FAST](https://huggingface.co/physical-intelligence/fast/blob/main/README.md)는 연속 action sequence용 tokenizer를 학습한다. ID만 배정해서 grounded meaning이 생기지 않는다. 모델이 code를 읽거나 출력하도록 학습·적응해야 하며 tokenizer collision도 배제해야 한다.

같은 차원이나 embedding table을 공유하는 것은 전송 규약일 뿐 alignment가 아니다. Embedding encoder는 비생성 semantic feature extractor다. Text similarity를 요약할 수 있지만, 임의의 물리 측정을 같은 폭으로 projection한다고 semantic해지지는 않는다.

## 제안하는 두 clock 구조

안정화에는 **빠른 motor loop**, 목표·mode 변경과 recovery 결정에는 **느린 semantic loop**를 사용한다.

대략 20–100 Hz의 specialist controller는 pose, contact, local sensory feature, 현재의 제한된 goal을 받아 joint target이나 작은 motor primitive를 출력한다. 대략 0.5–2 Hz 또는 event 발생 시 language path가 집계된 state window를 읽고 `set_goal_heading`, `approach_odor`, `stop`, `request_recovery` 같은 typed command를 고른다. Safety layer가 범위, 만료, 허용 tool을 검사한 뒤 command를 확정한다. 정확한 속도는 실험 parameter이며 생물학 주장이 아니다.

Read bridge와 write bridge는 독립된 map이다.

```text
관측 -> state encoder R -> text / soft prefix / token code -> LM
LM tool 결정 -> parser와 guard -> action-grounding map W -> specialist controller
                                      ^                           |
                                      |------ 관측 outcome -------|
```

`W`를 `Rᵀ`로 정의하거나 `R`의 역함수라고 가정해서는 안 된다. Reading은 어떤 latent state가 semantic decision을 예측하는지 물으므로 calibration 또는 paired alignment 근거가 필요하다. Writing은 어떤 grounded command가 몸을 안전하게 바꾸는지 묻는다. 그 grounding은 측정된 analytic controller나 fitted model에서 올 수 있다. 두 경로 모두 독립적으로 검증된 mapping이 필요하지만 반드시 learned dynamics가 필요한 것은 아니다. Task success, valid-command loss, paired observation의 representation alignment, 미래 관측 상태 예측을 objective와 점검 항목으로 함께 쓸 수 있다. Cycle consistency는 보조 loss가 될 수 있으나, 표현력이 큰 두 network가 물리 의미 없이 정보를 숨겨 서로 복원할 수 있다. Goal, observation, action availability, feedback 중 하나를 조작하고 matched alternative를 고정하는 인과 대조가 필요하다.

Typed message는 실제 관측과 의도를 구분해야 한다.

```json
{
  "seq": 1842,
  "frame": "arena/world",
  "time_s": 27.440,
  "valid_until_s": 27.940,
  "goal": {"type": "heading", "rad": 1.20},
  "observed": {"heading_rad": 0.83, "sigma_rad": 0.12, "odor": 0.41},
  "commanded": {"turn_rate_rad_s": 0.30, "issued_at_s": 27.420},
  "ack": {"command_seq": 1841, "status": "accepted"}
}
```

Sequence number는 오래된 acknowledgement를 막고, frame과 unit은 좌표 오류를 막으며, expiry는 느린 모델 latency를 제한한다. 가능하면 bridge를 지나는 동안 distribution, interval, top-k hypothesis를 보존한다. 이른 argmax는 ambiguity를 없애고 근거보다 downstream decision이 더 잘 calibrated된 것처럼 보이게 할 수 있다.

## 쓰기 브리지의 두 목적지

로봇을 제어하는 tool call과 신경집단 상태를 바꾸는 입력은 서로 다른 쓰기 대상이다. 둘 다 연결할 수 있지만 각각의 보정이 필요하다.

**의미에서 회로 입력으로.** 명령 임베딩을 확인된 Novi 입력 포트의 비음수 패턴으로 바꾸고 강도·지속 시간·시작 시각을 붙일 수 있다. 기존 PortBridge는 이미 공학적인 임베딩→PN 표현을 제공하지만 텍스트 임베딩을 매핑했다는 사실이 자연적인 파리 감각 코드임을 입증하지는 않는다. 목표 방향을 원형 활동 분포로 쓰는 후속 모듈도 생각할 수 있다. 다만 현재 PN→KC→MBON에는 그 분포를 넣을 FC2/PFL3 동역학이 없으므로 별도 구현과 검증이 필요하다. 임의 projection에 FC2라는 이름만 붙여서는 안 된다.

**회로 출력 또는 모델 의도에서 운동으로.** 모의 신경집단의 읽기 결과나 검증된 언어모델 의도를, 효과를 보정한 운동 패턴에 연결할 수 있다. 신경 포트에 쓰는 제안이라면 `u = B a`로 정의한다. `B`의 각 열은 허용하는 공간 패턴이고 `a`는 적은 수의 강도 값이며, 시간 포락선으로 지속 시간을 붙인다. 강도·부호·시간 제약은 실제 시뮬레이터/API에 맞춰야 한다. 동물 자극 처방이 아닌 소프트웨어 개입이다. 먼저 `(상태, 입력 u) -> 다음 관측/행동`의 순방향 효과를 보정하고, 그다음 원하는 과제 변화를 만드는 쓰기 정책을 맞춘다. 상태를 잘 읽었다고 복원한 표현을 주입해 그 상태를 만들 수 있는 것은 아니다.

같은 초기 상태와 난수에서 패턴 계수를 하나씩 바꾸고, 실제 과제 변수의 변화로 국소 효과 행렬을 추정한다. 행렬의 rank와 조건수를 검사하면 도달할 수 없는 방향이나 구별되지 않는 입력을 찾을 수 있다. 임베딩이 커져도 액추에이터가 만들 수 없는 행동은 복구하지 못한다. 이전 Novi 실행부의 도달 가능 행동 수 제한이 이런 문제의 예다. 의도하지 않은 효과, 비선형 강도 반응, 입력을 제거한 뒤 회복도 검사해야 하며 국소 행렬만으로 전역 제어 가능성을 입증할 수는 없다.

읽기·쓰기 보정은 독립적으로 측정한 결과로 평가해야 한다. 자기 latent를 복원하는 순환은 임의 코드를 서로 주고받으면서도 몸을 잘못 움직일 수 있다. 자극 대응 순열, 같은 크기의 무효 입력, 액추에이터 차단, 같은 관측을 받는 전문 제어기를 대조군으로 둔다. 환경에서 나온 보상과 언어모델의 보상 예측도 분리한다. 모델이 성공했다고 말한 문장이 학습 보상이나 실제 성공 판정이 되어서는 안 된다.

## 범위가 제한된 실험과 예측

모든 scenario에서 먼저 LM 없이 작동하는 rule-based 또는 learned specialist controller를 사용해야 한다. 언어 interface는 선언한 goal이나 mode만 바꾼다.

* **냄새–바람 anemotaxis:** odor 유무, wind direction, turbulence, plume loss를 교차한다. Plume loss 뒤에는 temporal state가 latest-frame input보다 나을 것으로 예측한다. Oracle plume vector, hand-coded cast/surge, linear/MLP state policy, structured text, soft prefix를 비교한다.
* **어둠 속 목표 방향:** 방향을 설정한 뒤 visual landmark를 제거하고 통제된 self-motion bias를 넣는다. 정확한 방향을 꾸며내는 대신 uncertainty가 증가하고 성능이 완만하게 저하되어야 한다. Oracle heading, path integration, 마지막 heading 고정, shuffled goal을 비교한다.
* **내부 상태 memory gating:** 기억한 odor value와 독립적으로 공급한 hunger-like state를 교차한다. Memory와 state가 함께 요구할 때만 상태 의존 선택이 나타날 것으로 예측한다. Memory-shuffled, state-shuffled, state-only, label-oracle 대조를 둔다.
* **고유감각 교란:** 발행 command는 유지하면서 관측 leg state를 offset 또는 delay하고, 별도로 actuator를 교란한다. Sensing fault와 execution fault가 다른 signature를 보여야 한다. Command-only, observation-only, matched-delay, no-perturbation을 비교한다.
* **Reversal과 reward omission:** 습득 뒤 하나의 goal–outcome 관계를 뒤집고 때때로 기대 reward를 생략한다. Trial 단위 adaptation, perseveration, recovery를 측정한다. Host Bayesian update, supervised retraining, fixed policy는 서로 다른 baseline이다. 측정된 mechanism 없이 어느 것도 dopamine learning이라 부르지 않는다.
* **Tool–body 조합 일반화:** semantic tool, body side, terrain, language paraphrase의 조합을 holdout한다. Action grounding을 독립적으로 학습한 경우에만 typed composition이 도움이 될 것으로 예측한다. Flat class ID, compositional schema, text-only state, oracle action을 비교한다.

고정 episode seed를 쓰고 겹치는 window가 아니라 전체 scene/trajectory 단위로 split한다. Success, 시간, fall/collision, energy 또는 distance, invalid/stale command, intervention 수, deadline miss, stop latency, 교란 후 recovery, calibration을 보고한다. 느린 loop의 semantic accuracy와 빠른 loop의 control quality를 따로 보고한다. 필수 인과 대조는 zero/random/shuffled state, permuted state-to-case pairing, swapped goal, unavailable-action mask, delayed feedback, 동일 관측 replay, 같은 capacity의 non-connectome encoder다. Valid tool call은 motor competence의 증거가 아니다. Closed-loop 향상만으로 LM, host algorithm, specialist controller 중 무엇이 원인인지 식별할 수 없다.

## 현재 robot foundation model과의 관계

RT-2와 OpenVLA는 robot action을 언어 모델식 출력으로 이산화하고 robot data로 학습한다. 이는 grounded interface의 공동 학습을 지지하며, 일반 LM에서 motor meaning이 zero-shot으로 생긴다는 증거가 아니다. FAST는 이런 discrete autoregressive action generation을 위한 sequence compression을 개선한다. 반대로 [π0](https://www.physicalintelligence.company/download/pi0.pdf)는 vision-language backbone으로 조건화된 별도의 continuous action expert를 사용한다. Semantic reasoning과 빠른 연속 제어가 context를 공유하되 output clock은 공유하지 않을 수 있다는 점에서 Novi에 특히 적합한 선례다.

[NDT2](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html)와 [POYO-1](https://poyo-brain.github.io/) 같은 neural-data model은 timestamp event, heterogeneous session, contextual adaptation의 유용한 선례다. Neural activity가 LLM에 alignment되었다거나 여러 recording session으로 학습한 decoder가 causal motor command를 제공한다는 결과는 아니다.

## FlyGym 실현 가능성과 release 규율

[FlyGym 2.x](https://neuromechfly.org/installation/)는 NeuroMechFly 기반의 Apache-2.0 simulation environment이며 CPU 실행을 지원한다. Optional NVIDIA Warp 경로는 Mac용 경로가 아니다. 현재 repository 환경은 Python 3.11을 쓰지만 현재 FlyGym 2.x 설치 안내는 Python 3.12를 대상으로 하므로 repository interpreter를 바꾸지 말고 격리된 pinned environment를 사용해야 한다. FlyGym 2.x는 논문 시기의 1.x와 API도 다르므로 정확한 package/version, MuJoCo version, timestep, arena, controller parameter, asset, reset seed를 고정한다.

첫 readiness check는 native rule controller를 사용한 CPU smoke test여야 한다. Deterministic reset, 수백 step의 안정적 simulation, 유효한 observation/action, 고정 환경에서의 동일 replay를 확인한다. 그다음 heading-token 또는 soft-prefix toy test를 할 수 있지만 여전히 interface test다. 별도의 고정 실험으로 측정하기 전에는 LM training, embodied success, 생물학적 motor circuit을 주장하지 않는다.
