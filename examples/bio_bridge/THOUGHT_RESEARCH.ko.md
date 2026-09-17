# 파리 영감 임베딩 브리지의 잠재 상태 검증

## 여기서 “생각”이 의미할 수 있는 것

이 프로젝트는 파리의 사적 경험, 내적 언어, 주관적 생각을 추출할 수 없다. 검증 가능한 대상은 **조작적으로 정의한 잠재 변수**다. 즉 현재 자극과 같지 않으면서 행동을 예측하고, 독립 trial의 신경 활동에서 decode되며, 관련 회로를 교란했을 때 행동을 바꾸는 양이다.

다음 네 가지를 구분해야 한다.

1. **감각 증거**: 현재 관찰에 들어 있는 정보
2. **내부·동기 상태**: 같은 cue의 사용법을 바꾸는 hunger, satiety, arousal
3. **결정 변수·기대 가치**: 현재 자극 정체성을 넘어 다음 선택을 예측하는 학습된 양
4. **heading·운동 계획**: allocentric 방향 추정 또는 행동 관련 신호

Decoder가 이 값을 읽을 수 있어도 회로가 그 값을 사용한다는 뜻은 아니다. 수동 encoding 경로가 실험자가 제공한 입력을 복원할 수도 있다. 따라서 cross-validated decoding은 측정 단계이고, 인과적 사용은 별도 perturbation과 행동 특이성으로 검증해야 한다.

## 1차 문헌과 한계

| 1차 연구 | 측정·조작 | 지지하는 내용 | 지지하지 않는 내용 |
|---|---|---|---|
| [Aimon et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6395010/) | 행동 중 성체 파리의 빠른 near-whole-brain calcium/voltage imaging. 보행 때 전역 활동이 변했고 국소 source도 감각·행동 변수와 연관됨 | 신경 기록을 자극·행동과 정렬해야 하며 전역 운동 상태가 큰 공변량임 | 저차원 component가 곧 생각이나 결정 변수라는 뜻은 아님. Calcium imaging만으로 인과성도 정해지지 않음 |
| [DasGupta et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4206523/) | 난이도를 바꾼 시각 선택에서 반응시간·정확도 측정, FoxP 조작 | 행동에서 evidence accumulation 가설을 세우고 psychometric/chronometric 예측을 반증 가능하게 만들 수 있음 | 행동 signature가 하나의 신경 accumulator를 유일하게 식별하지 않음. 언어 classifier logit도 파리의 언어적 숙고가 아님 |
| [Seelig & Jayaraman (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4704792/) | 가상 navigation 중 ellipsoid-body population bump가 시각 heading을 추적하고 암흑에서는 self-motion으로 갱신됨 | Heading은 원형 기하를 가진 구체적 잠재 변수이며 행동·감각 조건과 대조할 수 있음 | 이 bump는 범용 belief나 semantic intent가 아님 |
| [Kim et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28473639/) | Optogenetic 조작으로 heading bump를 이동시킨 뒤 회로가 이를 유지하고 자연스러운 동역학으로 전개 | Decode된 상태가 단순 입력 상관이 아니라 회로 동역학에 참여하는지 perturb-and-recovery로 검증 가능 | Heading 회로의 결과를 일반적인 “thought attractor”로 확장할 수 없음 |
| [Krashes et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2780032/) | Hunger/satiety와 dNPF/dopamine 표적 조작이 appetitive odor memory 표현을 변화 | 내부 상태가 학습 cue의 retrieval과 action을 gate할 수 있음 | Hunger가 추상 언어 벡터로 decode된 것이 아니며 fly PFC나 범용 value channel을 뜻하지 않음 |
| [Sterne et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10523640/) | 연속 operant choice, reward schedule, MB 조작, plasticity model로 matching behavior의 reward expectation 검증 | 기대 보상은 선택 history에서 추론하고 contingency 변화와 회로 조작으로 반증해야 함 | Model fit만으로 정확한 생물학적 학습 규칙을 확정할 수 없음 |
| [Shiu et al. / FlyWire Consortium (2024)](https://www.nature.com/articles/s41586-024-07558-y) | 성체 암컷 파리 뇌의 synapse-resolution reconstruction | Cell type, 후보 경로, 방향성, chemical synapse 수로 intervention 후보를 제한할 수 있음 | 정적 표본에는 활동, 막 동역학, 개체의 동기 상태, 학습 weight, 주관적 내용이 없음 |

## 측정 단계

### 1. 활동을 보기 전에 변수를 정의한다

Hunger–benefit–cost 과제에서 다음을 사전 지정한다.

- 관찰 `o_t`: 현재 food cue와 cost/threat cue
- 상태 `s_t`: 실험적으로 지정한 hunger 또는 host가 보존하는 상태값
- offer: 독립적으로 바꾼 benefit `b_t`, cost `c_t`
- 선택 `a_t`: approach, avoid 등의 고정 행동 집합
- 결과 `r_t`: 가능하면 물리 환경에서 측정한 reward/cost
- nuisance: text family, 문장 길이, movement, 시간, trial history, action 빈도

Class label을 “thought”라 부르지 않는다. 검증할 질문은 현재 cue를 통제한 뒤에도 hidden activity가 hunger를 나타내는지, 또는 intervention이 cue decoding을 유지하면서 cost sensitivity만 바꾸는지와 같은 조건부 예측이다.

### 2. 행동 계산을 먼저 확인한다

Hidden activity를 해석하기 전에 held-out choice model을 적합한다.

```text
P(approach) = sigmoid(theta_0 + theta_b b_t - theta_c c_t
                      + theta_s s_t + theta_bs b_t*s_t
                      + history terms)
```

Interaction은 내부 상태가 benefit 사용을 바꾸는지 검증한다. Benefit, cost, hunger를 독립적으로 조작하지 않으면 food word나 특정 action과 항상 함께 나타나는 hunger를 읽을 뿐이다. Contingency reversal과 state switch로 현재 증거, 학습 가치, perseveration을 구분한다.

### 3. 엄격한 cross-validation으로 decode한다

Hidden activity `h_t`에서 단순 probe를 training trial에만 적합한다.

```text
hunger probe:       s_hat = f_s(h_t)
value probe:        v_hat = f_v(h_t)
choice probe:       a_hat = f_a(h_t)
heading probe:      (sin(phi_hat), cos(phi_hat)) = f_phi(h_t)
```

Regularized linear probe부터 사용하고 held-out balanced accuracy 또는 circular error를 보고한다. 생물 데이터는 fly 전체를, 임베딩 실험은 semantic paraphrase family 전체를 hold out한다. 유사 paraphrase를 무작위 행 split으로 나누면 lexical memorization이 가능하므로 intent representation의 증거가 아니다. Scaler, prototype, ridge coefficient, threshold는 각 training fold 안에서만 적합한다.

필수 대조군:

- frozen embedding을 직접 쓰는 **input-only probe**
- 동일 split과 fitting 절차의 **shuffled-label probe**
- length, lexical family, action, trial time nuisance probe
- behavior-history 및 current-stimulus baseline
- held-out state switch와 benefit–cost 조합
- 여러 seed는 반복 최적화·샘플링 실행으로 보고하며 독립 동물처럼 취급하지 않음

Hidden probe가 input-only보다 낫지 않으면 graph가 잠재 계산 증거를 추가하지 못한 것이다. Hunger 단어가 명시된 입력에서 둘 다 decode하면 제공된 입력의 encoding만 보인 것이다.

### 4. 개입과 특이성을 검증한다

후보 상태 표현은 개입이 사전 지정한 행동 변화만 선택적으로 만들 때 메커니즘적 의미를 갖는다.

- 감각 embedding을 고정하고 후보 hunger component를 clamp 또는 swap
- training data로만 고른 후보 port/hidden unit를 silence
- norm-matched random-unit와 input-only intervention 비교
- raw cue identity와 무관한 선택은 유지하면서 benefit sensitivity만 바뀌는지 검사
- intervention 제거 또는 반전 뒤 회복 검사

같은 trial로 unit 선택, probe tuning, intervention 효과 추정을 모두 하면 순환 논증이다. 넓은 output 손실은 state 제거가 아니라 classifier 손상일 수 있다.

## Frozen EmbeddingGemma 실험

제안된 저장소 실험은 **semantic embedding bridge simulation**이며 파리 생각 측정이 아니다.

### 데이터 조건

다음을 독립적으로 교차한 언어 관찰을 만든다.

- need: hungry / satiated
- benefit: 낮음 / 높음
- cost: 낮음 / 높음
- semantic family: train/validation/test에 겹치지 않는 paraphrase template
- state presentation: 문장에 상태를 명시하거나, 초기 상태를 host에 저장한 뒤 후속 관찰에서는 생략

Frozen EmbeddingGemma 특징, 기존 signed port encoding, frozen Novi input-to-hidden connectivity를 사용한다. 임베딩 모델은 fine-tune하지 않는다. 기존 confirmation 사례는 호환성 설명에는 쓸 수 있지만 재사용 사례이므로 새로운 일반화 결과가 아니다.

### 서로 다른 주장을 갖는 세 readout

1. **Embedding prototype matching**: training text로 만든 prototype과 frozen input vector를 비교한다. 언어 모델이 이미 label을 분리하는지 보는 encoding-only baseline이다.
2. **Hidden ridge probe**: frozen native hidden activity에서 state, net value, intended action을 예측한다. Graph 변환 뒤 선형 접근 가능성을 검사한다.
3. **Native policy readout**: 기존 engine이 고르는 action을 측정한다. 현재 policy가 정보를 사용하는지 보지만 이전 학습과 readout compatibility에 제약된다.

Prototype 성공은 semantic similarity이지 cognition이 아니다. Hidden probe 성공은 decodability이지 causal use가 아니다. Native action 변화도 직접적인 wording 때문에 생길 수 있다.

### 반증 가능한 테스트

| 테스트 | 의도한 계산이 있다면 예측 | 실패 해석 |
|---|---|---|
| Semantic-family holdout | 보지 못한 paraphrase family로 state/value decoding 전이 | Random-row split은 lexical template 재사용을 측정했을 수 있음 |
| Explicit 대 host-maintained state | State word가 없어도 지속 host state가 있을 때만 이후 cue 처리가 state-dependent | Stateless feedforward engine은 생략 변수를 잃어야 하며 이는 생물학적 forgetting이 아님 |
| Benefit × cost grid | 두 요인을 독립 조작할 때 choice와 value probe가 각각 단조롭게 변함 | Need-word classifier는 value integration이 아님 |
| State switch | 같은 cue와 cost에서 hunger 변화 직후 choice가 바뀜 | 이전 선택 지속은 history/perseveration 또는 host-state 오류 가능 |
| Decoder-selected intervention | 표적 조작이 norm-matched random 조작보다 state-dependent benefit weighting을 더 변화 | 동일 손상은 일반적 파괴 또는 decoder selection bias |
| Shuffled labels | Probe가 permutation null로 복귀 | Null보다 높은 값은 leakage 또는 잘못된 split |

### 정체성 decoding과 인과적 사용

Cue class와 semantic goal의 read-only identity decoder를 intervention 실험과 함께 실행한다. 높은 identity 정확도는 행동 사용 없이도 가능하다. 반대로 identity probe가 안정적인데 native action만 변할 수도 있다.

```text
encoding:      input/hidden activity로 제공 label을 복원할 수 있는가?
behavior:      frozen policy의 선택이 달라지는가?
intervention:  후보 component 조작이 행동을 선택적으로 바꾸는가?
```

현재 저장소만으로 가능한 가장 강한 결론은 “frozen language embedding과 graph transform에서 사전 지정한 state/value label이 decode되었고, held-out intervention이 지정한 policy dependence를 바꿨다” 정도다. 여전히 소프트웨어 결과다. 언어 벡터 축을 실제 파리의 신경 상태와 정렬하려면 동시 신경 기록, 행동, 자극·내부 상태 label, 개체 간 registration, 인과적 perturbation이 필요하다.

## 잠재 상태처럼 보이게 만드는 혼입

- **입력 복원**: decoder가 변환·유지된 상태가 아니라 제공된 state word를 읽음
- **운동 leakage**: choice와 연관된 walking/turning 전역 활동을 읽음
- **outcome leakage**: 선택·보상 뒤 활동으로 선택 전 value를 decode
- **semantic-family leakage**: 가까운 paraphrase가 train/test에 동시에 존재
- **action imbalance**: hungry일 때 항상 approach하여 action decoder가 hunger decoder처럼 보임
- **trial-history leakage**: 같은 episode의 인접 sample이 fold를 넘음
- **probe flexibility**: native readout이 사용할 수 없는 우연한 정보를 고용량 decoder가 추출
- **selection circularity**: 같은 trial로 unit 선택, tuning, 효과 추정을 수행
- **connectome 과장**: 경로나 synapse 수를 동적 belief, value, thought로 서술

## 향후 연구의 수용 기준

신뢰할 수 있는 소프트웨어 진단은 사전 지정 label/contrast, semantic-family held-out 평가, input-only 및 shuffled-label baseline, 시간 인과적인 특징, 추가 정보를 보이는 hidden probe 또는 선택적 policy intervention, state-switch 및 benefit–cost 테스트, 전체 seed 결과를 요구해야 한다. 생물학적 주장은 여기에 행동과 짝지은 신경 데이터 및 파리의 인과적 perturbation이 추가로 필요하다. 그런 데이터가 없다면 “파리 영감 임베딩 브리지의 상태 decoding”이라고 쓰고 “파리의 생각 추출”이라고 쓰지 않는다.
