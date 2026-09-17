# 임베딩–회로 브리지의 아키텍처 후보

## 범위와 주장 경계

현재 브리지는 과제 범주를 잘 보존하지만 정확한 문장 정체성은 많이 잃고 행동 확률의 분리도 약하다. 이는 새 아키텍처 비교의 동기이며, connectome이 언어 의미를 학습했다는 증거는 아니다. 아래 세 후보는 일부 회로 결과에서 제약을 얻은 **소프트웨어 가설**이다.

새 측정 결과는 이 문서에 없다. 입력 임베딩에서 출력으로 가는 직접 경로는 **의미 운반(semantic transport)**이지 neural decoding이 아니다. 순환 소프트웨어 상태는 파리의 기억 흔적이 아니다. 실제 파리의 자극·신경활동·행동·상태를 쌍으로 측정한 데이터에 맞춘 경우에만 contrastive alignment를 생물학적 정렬이라 부를 수 있다.

## 1차 문헌의 근거와 한계

| 1차 연구 | 설계와 관련된 결과 | 소프트웨어 유추의 한계 |
|---|---|---|
| [Aso et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4273436/) | MBON은 병렬·구획화된 출력을 이루며, 특정 MBON 조작이 학습된 접근/회피를 편향시켰다. | 과제/가치 채널 분리를 뒷받침하지만 언어 residual이나 보편적 의미 bottleneck은 뒷받침하지 않는다. |
| [Dolan et al. (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6226615/) | 특정 lateral-horn 뉴런이 PN의 hardwired 입력과 MBON의 학습 입력을 함께 받고, 일부 선천적 접근과 기억 인출에 필요했다. | 수렴하는 두 경로의 비교에는 근거가 되지만 직접 embedding bypass가 이 파리 경로와 같다는 뜻은 아니다. |
| [Kim et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28473639/) | central-complex heading bump를 이동시키면 회로가 바뀐 heading 추정치를 유지하고 갱신했다. | 특정 항법 회로의 순환 상태 근거이며 일반 의미 belief state의 근거가 아니다. |
| [Green et al. (2019)](https://www.nature.com/articles/s41593-019-0444-x) | heading 활동과 내부 목표가 지속적인 방향 보행을 예측했고, 교란과 행동 분석으로 항법 역할을 검사했다. | 목표 상대 항법이 임의 언어 context routing을 입증하지 않는다. |
| [Westeinde et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10881393/) | FC2/PFL 경로가 목표와 heading 신호를 측정 가능한 비선형 상호작용으로 steering 관련 활동으로 바꿨다. | goal-dependent 신호 사용을 시험할 근거일 뿐, 모든 과제에 FC2/PFL 명칭이나 일반 gate를 적용할 근거는 아니다. |
| [Aimon et al. (2019)](https://doi.org/10.1371/journal.pbio.2006732) | 행동 중 near-whole-brain 기록에서 광범위한 운동 관련 활동과 국소 감각·행동 상관을 함께 관찰했다. | 실제 정렬은 동시 행동과 공변량을 필요로 하며 정적 connectome 모사는 기록을 대신하지 못한다. |
| [Schneider, Lee, and Mathis (2023)](https://www.nature.com/articles/s41586-023-06031-6) | CEBRA는 쌍을 이룬 시간/행동 변수와 신경 기록으로 contrastive latent를 학습하고 기록 간 일관성을 검사했다. | 정렬 방법의 근거이지 frozen language vector가 이미 파리 신경활동과 맞는다는 근거가 아니다. |

정적 connectome edge는 가능한 경로와 synapse 수를 제약한다. 뉴런 동역학, 학습된 weight, context, 자극–신경 대응은 제공하지 않으므로 edge만으로 세 후보를 선택할 수 없다.

## 별도의 적응 축: bridge rank와 model fine-tuning

두 가지 “low rank”를 혼동하면 안 된다.

**Bridge adaptation**은 EmbeddingGemma를 고정하고 embedding-to-port 경계의 행렬만 학습한다. base map `W0`에 대해 다음 조건을 비교할 수 있다.

```text
W = W0 + A B,    A in R^(319 x r), B in R^(r x 768)
r in {4, 16, 64}, 그리고 명시적으로 이름 붙인 unrestricted-W 조건
```

unrestricted 조건은 “full-rank bridge fitting”이지 full-model fine-tuning이 아니다. 학습 문장 32개를 centering한 empirical design matrix의 rank는 최대 31이다. 따라서 rank 64와 unrestricted bridge parameter는 이 표본에서 64개의 독립 방향으로 뒷받침되지 않으며 주로 parameterization과 interpolation 자유도를 늘린다. 실제 update rank, singular value, held-out semantic family 일반화를 보고한다. 높은 rank 조건은 개선을 전제하지 않는 stress test다.

**Embedding-model adaptation**은 308M-parameter EmbeddingGemma encoder 내부 parameter를 바꾼다. [Hu et al. (2021)](https://arxiv.org/abs/2106.09685)의 LoRA는 pretrained weight를 고정하고 low-rank additive update를 학습한다. [Liu et al. (2024)](https://openreview.net/forum?id=3d5CIRG1n2)의 DoRA는 weight magnitude와 direction을 분리하고 direction에 low-rank update를 적용한다. Full fine-tuning은 선택한 model weight 전체를 갱신한다. 이 논문들의 결과가 이 소규모 multilingual bridge에서 LoRA, DoRA 또는 full tuning의 개선을 보장하지 않는다.

[Google 공식 EmbeddingGemma 안내](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers)는 Sentence Transformers fine-tuning과 CPU/GPU 실행을 문서화한다. [공식 개요](https://ai.google.dev/gemma/docs/embeddinggemma)는 308M parameter, 768-to-128 MRL output, 200MB 미만의 **quantized inference** memory를 설명한다. 이 inference 수치는 training-memory 추정치가 아니다. 공식 예제는 CUDA 실행을 보이지만 Apple Silicon/macOS peak memory나 throughput을 보장하지 않는다. 따라서 정확한 dependency/model revision으로 작은 MPS/CPU compatibility 및 peak-memory probe를 먼저 해야 Mac 실행 가능성을 판단할 수 있다.

현재 32개 문장은 308M encoder 적응의 근거로 부족하다. 반복 paraphrase와 번역은 표면 표본 수를 늘려도 독립 semantic family 수를 늘리지 않는다. model adaptation을 시도한다면 훨씬 많은 train-only pair, untouched family/language, frozen-base 비교를 갖춘 별도 실험으로 둔다. LoRA rank 4부터 시작하고 development 결과가 있을 때만 rank를 늘린다. DoRA와 full tuning은 resource/overfitting stress test이며 필수 단계가 아니다. 이번 architecture round에서는 어떤 model fine-tuning도 수행하지 않았다.

## 후보 1: 과제 bottleneck과 residual 의미 운반

두 경로를 명시적으로 분리한다.

```text
e                 = frozen sentence embedding
p                 = fitted 319-port input
z_task             = circuit_task_summary(native_circuit(p))
z_residual         = R(e)                    # 직접 운반
semantic_output    = D([z_task, z_residual])
action_output      = policy(z_task)          # 기본적으로 residual 제외
```

과제 경로는 고정 topology가 행동 관련 compact representation을 지지하는지 묻는다. residual 경로는 과제 bottleneck이 버릴 수밖에 없는 문장 세부정보를 보존한다. 이는 학습/선천 경로가 상호작용하는 파리 결과와 넓은 수준에서만 유사하다. `R(e)`가 native graph를 우회하면 그 경로가 복원한 정보는 neural decoding이라 부르지 않는다.

필수 ablation은 task-only, residual-only, fusion, shuffled residual, frozen random projection, `e`에서 직접 예측하는 parameter-matched MLP/ridge다. 정확 문장 retrieval과 과제 정확도를 따로 보고하고, 평가 때 각 경로를 0으로 만들어 fusion 의존도를 측정한다. 별도 선언한 실험이 아니면 residual은 행동 policy에 전달하지 않는다.

이 후보는 정체성을 보존하는 가장 빠른 방법이다. 그러나 높은 정체성 점수가 단순 복사로 생길 수 있으므로 주된 신경과학 결과가 아니라 공학적 상한선과 강한 대조군으로 취급한다.

## 후보 2: 목표 조건부 routing과 bounded recurrent state

현재 문장만으로 풀 수 없는 계산을 시험한다. 최소 상태 모델의 예는 다음과 같다.

```text
g_t = sigmoid(W_g [context_t, h_(t-1)] + b_g)
x_t = g_t * circuit_features(port_transform(e_t))
u_t = sigmoid(W_u [x_t, h_(t-1)] + b_u)
h_t = clip((1-u_t) * h_(t-1) + u_t * tanh(W_x x_t + W_h h_(t-1)), -1, 1)
action_t = softmax(W_a [x_t, h_t, goal_t])
```

이 식은 engineered gated state이며 FC2, PFL 또는 ring-attractor 동역학을 추출한 것이 아니다. 상태는 episode 사이에서 reset하고, 차원과 수치 범위를 고정하며, 해당 시점에 관측 가능한 입력만 받는다. host가 제공한 목표는 provided context로 표시한다. 목표 변수를 숨기고 관측 이력에서 추정해야 할 때만 inferred belief를 평가한다.

결정적인 데이터는 현재 관측은 같지만 목표나 이력에 따라 정답 행동이 달라지는 episode를 포함해야 한다. cue dropout과 odometry, 지연된 목표 전환, 과거 오류로 신뢰도를 관측할 수 있는 충돌 cue, distractor 구간을 권한다. feedback은 환경 outcome에서 와야 하며 state update에 숨은 정답을 넣으면 안 된다.

필수 대조군은 no-context, shuffled context, additive conditioning, multiplicative conditioning, 매 step state reset, frozen/random recurrence, 용량을 맞춘 feed-forward history window, 가능한 경우 analytic task-state baseline이다. future observation, correct action, simulator truth가 예측 전에 tensor로 유입되지 않았는지 검사한다. 전체 정확도뿐 아니라 switch 직후, dropout 중, distractor 이후를 따로 비교한다.

현재 실험 가능한 후보 중 실질적인 아키텍처 변화로는 이 후보를 우선 추천한다. 명확한 반증 조건이 있다. bounded state와 goal routing은 이력이나 목표가 행동을 바꾸는 조건에서만 이득을 내야 하며, 맞춘 stateless trial에서는 이득이 사라져야 한다. 시간 대조군에서 실패하면 semantic reconstruction이 좋아도 이 mechanism은 기각한다.

## 후보 3: predictive/contrastive cross-modal alignment

실제 paired data가 있을 때 언어/자극 encoder와 neural encoder를 따로 학습한다.

```text
q_t = f_language(description_t, measured_stimulus_t)
k_t = f_neural(recording_t)
L_pair = -log exp(sim(q_t, k_t)/temperature)
               / sum_j exp(sim(q_t, k_j)/temperature)
L_pred = distance(P(k_(<=t)), k_(t+1))
L = L_pair + lambda * L_pred
```

positive pair는 같은 기록 trial과 정렬된 시간창을 공유해야 한다. negative는 자극, 행동, session, 개체, 움직임을 균형화해 모델이 카메라·개체·batch를 식별하는 것을 막는다. 개체, session, 자극 family, 시간 block을 hold out한다. ridge, CCA/Procrustes, time-only embedding, trial-ID baseline, time-shift pair, shuffled pair와 비교하고 모든 전처리는 train split 안에서만 fit한다.

현재 프로젝트에는 paired fly recording이 없다. 따라서 합성 circuit rate는 이 주장을 검증할 수 없고 파일 형식과 leakage guard만 시험할 수 있다. 실제 데이터 전까지 후보 3은 점수를 낸 생물학 결과가 아니라 data contract와 사전등록 protocol로 둔다.

## 권장 실험 순서

1. **주 실험: 후보 2.** 관측, 목표, 이력을 독립적으로 조작한 순차 episode를 만든다. 언어 encoder와 native topology를 고정하고 state 크기, gate 형태, regularization은 development family에서만 선택한다.
2. **공학적 상한선: 후보 1.** 같은 split에서 실행해 보존 가능한 정체성의 상한을 잰다. residual-only와 fusion은 모두 semantic transport로 표시한다. 제품상 유용할 수 있으나 circuit decoding 증거는 아니다.
3. **연기된 생물학 정렬: 후보 3.** versioned paired-data schema는 지금 정의하되 실제 stimulus–neural–behavior recording과 개체/session ID가 확보된 후에만 학습한다.

후보 2는 seen/held-out semantic family, same/switched goal, current cue/dropout, clean/distractor의 full factorial로 평가한다. 주 시간 metric을 사전등록하고 family 단위 결과를 보고한다. sampling seed는 반복 소프트웨어 실행이지 독립 생물학 표본이 아니다.

## 강한 baseline과 채택 조건

모든 후보는 동일 split과 parameter accounting 아래 다음과 비교한다.

- raw frozen-embedding nearest prototype과 regularized linear model;
- locked PCA bridge와 native policy;
- connectome layer가 없는 parameter-matched direct recurrent model;
- fixed history concatenation과 가능한 경우 analytic state estimator;
- shuffled label, shuffled episode order, shuffled context, time-shifted feedback;
- residual-only와 circuit-only;
- greedy accuracy뿐 아니라 calibration metric과 action margin;
- exact identity retrieval, semantic-family accuracy, task reward의 분리 보고.

후보는 사전등록된 temporal/interaction 이득이 여러 seed와 held-out family에서 반복되고, capacity-matched direct baseline을 이기거나 재현 가능한 tradeoff를 보이며, serialization/replay와 no-truth-leak test를 통과할 때만 유지한다. 직접 residual 복사만으로 오른 전체 점수는 공학 결과이지 mechanism 결과가 아니다. 후보 3은 추가로 held-out 개체/session과 paired biological data가 필요하다.

### 적대적 반증 세트

하나의 underlying scenario에 속하는 모든 paraphrase, 번역, template variant, counterfactual을 같은 split group으로 묶는다. PCA, port map, gate, adapter, threshold, prototype, calibrator는 train에서만 fit한다. final example이 prompt 작성, vocabulary 선택, hard-negative mining, early stopping, architecture 선택에 쓰이지 않았음을 log와 hash로 확인한다.

final set에는 표면 유사성이 정답 행동과 충돌하는 minimal pair를 포함한다.

- 부정: “물이 필요하다”와 “물이 필요하지 않다”;
- scope: “따뜻함이 아니라 물이 필요하다”와 “따뜻하지만 목마르지는 않다”;
- 인용/가정: “그녀가 ‘배고프다’고 말했다”와 agent 자신이 배고픈 경우;
- stale state: 이전 목표가 참이었지만 이후 명시적으로 전환된 경우;
- lexical decoy: 음식 단어는 많지만 실제 제약은 위협 또는 보온인 경우;
- 의미가 같은 한/영 pair와 관계 하나만 바뀐 pair;
- 현재 관측은 같고 이력만 다른 경우, 표현은 다르고 이력은 같은 경우.

일반 paraphrase는 통과하지만 negation/counterfactual pair에 실패하거나, lexical family holdout에서 붕괴하거나, raw-input baseline과 사전등록 tolerance 안에서 같으면 semantic architecture를 기각한다. shuffled history나 future-leaking history가 같은 성능을 내거나, episode reset 뒤 state가 남거나, stateless trial에서만 이득이 있고 history-dependent trial에서 없으면 recurrent mechanism을 기각한다. additive conditioning 또는 parameter-matched direct model이 held-out family 전반에서 같으면 multiplicative routing을 기각한다. time-shift 또는 animal/session-ID control에서도 이득이 유지되면 contrastive alignment를 기각한다.

생물학적 inspiration도 반증 가능해야 한다. 인용 연구를 정의하는 변수나 인과 구조가 없어도 계산이 작동한다면 해당 유추의 지지는 사라진다. 예를 들어 goal shuffle 뒤에도 작동하는 “goal routing”, 매 step reset 뒤에도 작동하는 “persistent state”, paired modality 없이 학습한 “cross-modal alignment”가 그렇다. 이 실패는 소프트웨어 도구 자체를 무효화하지 않지만 해당 생물학적 해석을 무효화한다.

### 적응 비교

추후 충분한 추가 학습 데이터를 모으면 frozen base, bridge rank 4/16/64/unrestricted, 사전등록 rank의 encoder LoRA, DoRA, full encoder tuning을 같은 family-level split에서 비교한다. optimizer step을 맞추고 trainable parameter, peak memory, elapsed time, effective update rank, frozen general-domain regression test를 보고한다. 큰 방법의 held-out gain이 seed variation을 넘지 못하거나 calibration/multilingual robustness가 크게 나빠지거나, temporal-task 개선 없이 training-family 암기로 exact retrieval만 오르면 기각한다. Full fine-tuning은 실행 전에 데이터 규모와 compute 근거가 필요하며 32개 예제의 기본 baseline이 아니다.

## 다음 구현의 계약

새 작업은 `examples/bio_bridge/architecture*.py`, 대응 test, `results/architecture-bridge/`에 격리한다. split manifest, 입력 hash, model revision, topology hash, fitted parameter, state reset 규칙, 정확한 prediction array를 저장한다. development에서 아키텍처를 선택하고 final authored-family set은 한 번만 연다. 여기서 replay는 저장 계산의 결정적 재현이지 생물학적 experience replay가 아니다.
