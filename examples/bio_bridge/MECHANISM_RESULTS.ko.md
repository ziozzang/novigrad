# 메커니즘 비교 측정 결과

이 문서는 [MECHANISM_RESEARCH.ko.md](MECHANISM_RESEARCH.ko.md)의 세 가설을 제한된 소프트웨어 진단으로 옮긴 결과를 기록한다. 대상은 인과적 입력 적응, 은닉층 경쟁, 지연 보상 귀속이다. 실험은 파리의 생물학적 메커니즘을 입증하지 않는다. EmbeddingGemma 벡터는 언어 특징이고 은닉 유닛은 설계된 Novi 유닛이며, 정적 커넥톰 엣지로 여기의 동역학이나 상수를 알아낼 수 없다.

코어 API와 기본값은 바뀌지 않았다. 정확도는 모두 greedy top-1 분류 정확도다. 원본 JSON에는 정답 클래스 확률, 엔트로피, 시드별 값, 해시, 개별 실행이 보존되어 있다.

![측정한 메커니즘 비교](../../results/mechanism-bridge/mechanism-comparison.png)

실험 간 machine-readable 요약: [summary.json](../../results/mechanism-bridge/summary.json).

## 결과 요약

| 질문 | 핵심 측정값 | 후보 판정 |
|---|---:|---|
| 지속적인 언어 특징 배경에서 인과적 채널 적응이 전경을 복원하는가? | 정적 강도 1, 8틱 전경: direct 37.50%, divisive EMA 46.16%, subtractive EMA 52.08%, oracle 52.08% | subtractive EMA를 과제 한정 후보로 유지. 강한 배경과 배경 전환은 부정적 결과로 유지 |
| APL에서 영감을 받은 감산 변환이 기존 희소 표현을 개선하는가? | 재사용 32건의 무배경 조건에서 native 2% top-k 52.08%, global/local 감산 46.88%/48.96% | 현재 체크포인트에는 native top-k 유지. APL 메커니즘으로 해석하지 않음 |
| 일반적인 누적 eligibility trace가 지연 보상을 해결하는가? | lag 16에서 exact cached event는 영어 test 79.17%, 모든 plain/norm-matched trace는 25.00% | 이 스트림의 지연 귀속 후보에서는 폐기. exact queue는 oracle 진단으로만 유지 |

## 1. 인과적 입력 적응

### 측정한 규칙

음이 아닌 입력 `x_t`에 대해 상태는 0에서 시작하며 현재 출력을 낸 **뒤에** 갱신된다.

```text
a = exp(-1 / 8)
y_t는 m_t로 계산
m_(t+1) = a m_t + (1-a) x_t

direct:                 y_t = x_t
divisive EMA:           y_t = x_t / (0.05 + m_t)
subtractive EMA:        y_t = max(x_t - m_t, 0)
fixed divisive:         y_t = x_t / (0.05 + mean_training_input)
shuffled subtractive:   y_t = max(x_t - permute(m_t), 0)
oracle background:      y_t = max(x_t - known_background_t, 0)
```

shuffled 대조군은 값이 들어 있는 256개 임베딩 포트만 섞고 패딩은 그대로 둔다. oracle은 정확한 배경 벡터를 받으므로 인과적 배포 시스템에서는 사용할 수 없다. 모두 호스트 측 필터이며, 피팅된 ORN/PN 동역학이나 생물학적 gain control의 증거가 아니다.

전체 요인 설계는 216회 실행이다. 동결된 체크포인트 3개 × 배경 강도 3개 × 전경 스케줄 2개 × 배경 조건 2개 × 규칙 6개다. 48틱 에피소드에서 같은 전경이 12틱과 32틱에 1틱 또는 8틱 나타난다. drift 조건은 24틱에서 방해 클래스가 바뀐다.

### 체크포인트 3개의 평균 전경 정확도

| 배경 | 스케줄 | Drift | Direct | Divisive EMA | Subtractive EMA | Fixed divisive | Shuffled subtractive | Oracle |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|
| 0.25 | sustained 8 | 아니오 | 52.08% | 47.59% | 52.60% | 46.88% | 44.53% | 52.08% |
| 1.0 | sustained 8 | 아니오 | 37.50% | 46.16% | 52.08% | 35.42% | 36.91% | 52.08% |
| 1.0 | sustained 8 | 예 | 36.98% | 43.29% | 48.96% | 34.90% | 36.26% | 52.08% |
| 4.0 | sustained 8 | 아니오 | 27.08% | 42.51% | 44.79% | 21.88% | 31.12% | 52.08% |
| 4.0 | sustained 8 | 예 | 22.92% | 29.36% | 33.33% | 20.31% | 29.62% | 52.08% |
| 1.0 | pulse 1 | 아니오 | 37.50% | 53.65% | 52.60% | 35.42% | 42.19% | 52.08% |
| 4.0 | pulse 1 | 예 | 22.92% | 36.46% | 35.42% | 20.31% | 28.65% | 52.08% |

사전 지정한 주 비교가 지지하는 범위는 좁다. 인과적 감산이 정적 강도-1 조건의 점수를 oracle 수준까지 복원했다. 하지만 강도 4의 drift+sustained 조건에서는 33.33%로 떨어졌다. 강도 1의 단일 pulse에서는 divisive EMA가 더 높았고, 주 비교의 sustained 조건에서는 subtractive EMA가 더 높았다. 고정 학습 평균과 채널 셔플은 주 결과를 재현하지 못했으므로, 이 구성 과제에서는 시간적이고 채널 정렬된 상태가 영향을 줬다고 볼 수 있다.

검출 gate는 없다. 작은 잔여 벡터도 downstream engine에서 정규화되어 행동을 만들 수 있다. 전경 자료는 이전에 사용한 confirmation 문장 32개이고, 이미 선택된 128차원·2% 활성 체크포인트 3개로 평가했다. 새로운 일반화 split이 아니며 세 개의 독립 데이터셋도 아니다.

원본: [adaptation.json](../../results/mechanism-bridge/adaptation.json).

## 2. 은닉 경쟁과 희소성

### 측정한 규칙

재구성한 native 억제 전 활성을 `h = relu(Wx)`라 하자. 모든 변형은 억제 뒤 행별 최대값으로 정규화된다.

```text
native top-k:       가장 큰 ceil(0.02 * 5177) = 104개 유지
20% top-k:          가장 큰 1036개 유지
global subtraction: relu(h_i - g_global * mean_j(h_j))
local subtraction:  relu(h_i - g_local * mean_{j in group(i)}(h_j))
```

32개 local group은 은닉 인덱스를 32로 나눈 나머지로 정했다. 해부학적 의미가 없다. gain은 old training 행에서 평균 활성 약 2%가 되도록 보정한 뒤 고정했다(`g_global=3.42445`, `g_local=3.41291`). 이는 APL에서 영감을 받은 경쟁 진단이며 APL 억제의 구현이나 식별이 아니다.

기존 readout을 그대로 사용한 무배경 재사용 confirmation 결과는 다음과 같다.

| 은닉 규칙 | 정확도 | 활성 유닛 | 다음 클래스 은닉 cosine | 분리 delta: 입력 cosine - 은닉 cosine |
|---|---:|---:|---:|---:|
| Native 2% top-k | 52.08% | 104.0 | 0.5250 | +0.2368 |
| 20% top-k | 38.54% | 1036.0 | 0.6783 | +0.0835 |
| Global subtraction | 46.88% | 91.3 | 0.6193 | +0.1425 |
| 임의 local subtraction | 48.96% | 91.7 | 0.6368 | +0.1250 |

배경 비율 1.0에서는 모두 저하됐다. native 37.50%, 20% top-k 40.63%, global 38.54%, local 37.50%였다. 기존 readout은 native 2% top-k로 학습되었으므로 이 policy 점수에는 표현 품질과 readout 호환성이 섞여 있다.

탐색적인 matched-readout 대조군은 각 표현에 동일한 행 정규화 supervised dual-kernel ridge head(`lambda=0.1`)를 old training label 32개로 피팅했다. 무배경 재사용 confirmation에서 native 65.63%, 20% top-k 65.63%, global 68.75%, local 53.13%였다. 배경 비율 1에서는 각각 31.25%, 43.75%, 40.63%, 34.38%였다. 모든 head의 training resubstitution 정확도는 100%였다.

ridge 대조군은 동결 readout 결과를 본 뒤 추가했다. 같은 작은 학습 label을 사용하고, 별도 held-out tuning이 없으며, 5,177개 좌표가 같아도 변형별 유효 활성 수는 다르다. readout mismatch가 순위를 바꿀 수 있음을 보여 주지만 global subtraction을 인과적 승자로 정하지는 못한다. 세 Novi 체크포인트의 input-to-hidden tensor는 동일하므로 세 개의 독립 encoder replicate가 아니다.

원래의 108개 행은 체크포인트 3개 × 데이터셋 3개 × 배경 비율 3개 × 은닉 규칙 4개다. 데이터셋은 old training, old validation/calibration, 재사용 confirmation 32건이다. gain 보정과 ridge 피팅에는 old training만 사용했다.

희소 NumPy/SciPy shadow forward는 parity 행에서 native engine 확률과 최대 절대 오차 `2.98e-8` 이내로 일치했다. 이는 측정용 shadow 계산의 수치적 검증이며 생물학적 검증이 아니다. 보고서는 저장된 ridge probe 4개의 체크포인트 해시와 재로딩 후 점수의 정확한 일치도 기록한다.

원본: [inhibition.json](../../results/mechanism-bridge/inhibition.json).

## 3. 지연 귀속과 eligibility

### 측정한 규칙

이 reference policy는 별도의 319×4 float64 선형 softmax 모델이다. 동결된 unit-L2 특징 `x_t`에서 `p_t`로 행동 `a_t`를 샘플링할 때 event와 trace는 다음과 같다.

```text
g_t = outer(x_t, onehot(a_t) - p_t)
lambda = exp(-1 / tau)
e_t = lambda e_(t-1) + g_t
W <- W + 0.03 * reward * credit
```

`current_only`는 지연 보상을 보상 시점의 event에 적용한다. `exact_cached_gradient`는 행동 시점의 `g_t`를 queue에 저장한다. trace 조건은 현재 누적 `e_t`에 보상을 적용한다. 사후 norm-matched trace는 `e_t`를 해당 cached event의 norm으로 재조정한다. 이 과정은 과거 norm을 oracle로 읽으므로 배포 가능한 메커니즘이 아니다.

조건별로 training action 2,048개를 샘플링했고, 행동마다 scalar reward 하나(`정답 +1`, `오답 -1`)를 전달한 뒤 lag만큼 warmdown했다. 평가는 기존 영어 test 24건과 한국어 test 12건을 사용했다. 세 sampling seed는 같은 데이터 split을 공유하므로 독립 데이터셋이 아니다.

### sampling seed 3개의 평균 greedy 정확도

| Lag | 귀속 규칙 | 영어 test | 한국어 test |
|---:|---|---:|---:|
| 0 | current / exact cached event | 81.94% | 83.33% |
| 0 | trace, tau 8 | 54.17% | 58.33% |
| 0 | norm-matched trace, tau 8 | 43.06% | 47.22% |
| 4 | current event | 25.00% | 27.78% |
| 4 | exact cached event | 79.17% | 86.11% |
| 4 | trace, tau 8 | 38.89% | 41.67% |
| 4 | norm-matched trace, tau 8 | 33.33% | 33.33% |
| 16 | current event | 25.00% | 19.44% |
| 16 | exact cached event | 79.17% | 83.33% |
| 16 | trace, tau 2 / 8 / 32 | 25.00% / 25.00% / 25.00% | 27.78% / 25.00% / 25.00% |
| 16 | norm-matched tau 8 / 32 | 25.00% / 25.00% | 25.00% / 25.00% |
| 전체 | zero reward | 25.00% | 25.00% |

원 설계는 42회 실행이었고, 사후 norm-matched 대조 12회를 더해 총 54회가 되었다. norm matching은 지연 성능을 복원하지 못했으며 lag 0과 4의 tau-8 결과도 더 낮췄다. 따라서 update 크기만으로 실패를 설명할 수 없다.

trace 수식 자체는 고립 대조를 통과했다. cue 전에 reward를 주면 update가 0이었고, 즉시 plain update와 norm-matched update는 정확히 같았으며, cue 하나 뒤에 blank event만 있을 때 trace는 `exp(-delay/tau)`와 정확히 일치했다. 연속 스트림에서는 중간 cue가 방향과 크기를 모두 바꿨다. tau 8, lag 16에서 원 event 방향 projection은 5.915, cross-talk L2는 2.808이었지만, 고립 event라면 잔여 비율은 0.1353이어야 했다. 따라서 부정적 결과는 연속 스트림의 event 간 귀속 문제이며 지수 감쇠 구현 오류가 아니다.

exact cached-event queue는 행동 시점 gradient를 보관하는 과거 oracle이다. 올바른 event에 reward가 연결되면 학습할 수 있음을 보이지만 생물학적으로 타당한 synaptic trace는 아니다. 측정한 누적 trace 역시 일반적인 설계 규칙일 뿐, 분자 수준의 per-synapse 모델도 아니고 저장소의 다른 곳에서 사용한 bounded per-action queue도 아니다.

원본: [eligibility.json](../../results/mechanism-bridge/eligibility.json).

## 해석 경계

이 진단으로 정당화되는 구현 판단은 측정 과제 범위에 한정된다.

- causal subtractive adaptation은 별도 held-out 검증 후보로 유지하고 divisive adaptation은 경쟁 모델로 둔다.
- 현재 체크포인트와 호환되는 은닉 규칙으로 native 2% top-k를 유지한다.
- 밀집 cue 스트림의 지연 귀속에는 측정한 일반 누적 trace를 폐기한다.

결과는 ORN, PN, APL, KC, dopamine receptor 또는 mushroom-body compartment가 여기의 계산을 수행한다는 뜻이 아니다. 구조적 경로나 synapse 수는 상호작용 위치의 후보를 제한할 수 있지만 EMA 상수, 감산 gain, 희소 threshold, eligibility decay를 식별할 수 없다. 그런 주장은 생리 동역학, 인과적 perturbation, held-out prediction이 필요하다.

## 재현

저장소의 기존 로컬 임베딩과 체크포인트를 사용한다.

```bash
.venv/bin/python examples/bio_bridge/adaptation_mechanism.py
.venv/bin/python examples/bio_bridge/inhibition_mechanism.py
.venv/bin/python examples/bio_bridge/eligibility_mechanism.py
.venv/bin/python examples/bio_bridge/summarize_mechanisms.py
.venv/bin/python -m unittest discover -s examples/bio_bridge -p 'test_*mechanism.py'
```

스크립트는 CPU 진단이며 JSON에 소스·데이터·체크포인트 해시를 기록한다.
