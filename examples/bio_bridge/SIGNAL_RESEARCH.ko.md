# 지속 recall, one-shot 신호, 신호 강도

## 먼저 분리해야 할 네 현상

1. **유지된 감각 입력:** cue가 계속 존재한다. 입력이 계속되므로 반응이 이어질 수 있으며 memory가 필요하지 않다.
2. **Cue 제거 뒤 working memory:** cue가 사라진 뒤에도 내부 상태가 유지되거나 갱신된다. recurrent state, integration, 또는 외부 state store가 필요하다.
3. **Associative recall:** 현재 cue가 과거에 학습한 value를 읽는다. 감각 반응은 짧아도 association은 오래갈 수 있다.
4. **One-shot training:** 한 번의 cue–outcome episode가 이후 행동을 바꾼다. 이는 학습 횟수이며 cue 지속 시간이나 recall 메커니즘이 아니다.

현재 frozen embedding → feedforward `novi` 평가는 call 사이에 자율 temporal state가 없다. 같은 vector를 반복하면 출력을 다시 계산할 뿐이다. 연속성은 host가 입력을 반복하거나, update가 weight를 바꾸거나, host state를 명시적으로 저장할 때만 생긴다. Working memory라고 부를 수 없다.

## 일차 연구와 소프트웨어 예측

| 일차 연구 | 직접 결과 | 한계와 반증 가능한 예측 |
|---|---|---|
| [Seelig & Jayaraman, 2015](https://www.nature.com/articles/nature14446) | Ellipsoid body activity bump가 visual landmark를 추적했고 어둠에서는 self-motion으로 heading을 추적했다. | Heading-state dynamics이며 MB associative recall이나 language memory가 아니다. Cue를 제거하고 odometry를 통제한 뒤 recurrent/stateful estimator와 reset feedforward를 비교해야 한다. |
| [Neuser et al., 2008](https://www.nature.com/articles/nature07003) | 초파리가 잠깐 본 visual orientation을 유지했고 ellipsoid-body ring neuron 기능과 plasticity가 인과적으로 관여했다. | 특정 spatial assay의 결과다. Cue-off delay를 바꾸고 reset, distractor, motor-only control을 두어 delay curve를 보고해야 한다. |
| [Honegger et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869/) | KC population은 여러 농도에서 sparse했지만 첫 presentation이 더 넓었고 강한 자극의 이전 이력이 뒤 반응에 영향을 주었다. | Intensity sweep 순서가 confound다. 농도, 무작위 순서, 반복, 간격을 교차한다. Stateless embedding은 동일 vector의 history effect를 만들 수 없어야 한다. |
| [Lüdke et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5960692/) | Calcium imaging에서 odor-specific post-offset activity가 관찰되었고, 측정한 위치 중 KC soma calcium pattern은 15–16초에 이전 odor를 decode할 정보를 유지했다. 저자들은 이를 짧은 sensory memory의 가능한 substrate로 제안했다. | Decodability는 상관적이며 KC-soma calcium이 행동 recall을 일으킨다는 인과 증거가 아니다. Cue 제거 뒤 input, hidden, output state를 각각 기록하고 reset한다. Host가 vector를 복사해 둔 것은 engineered trace이지 KC analogue 증거가 아니다. |
| [Masek & Heisenberg, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2556361/) | 행동적으로 odor quality와 intensity의 기억이 분리되었고 intensity memory는 더 짧았으며 농도 범위에 따라 generalization이 달랐다. | 정규화된 text-vector norm을 odor intensity로 볼 수 없다. Semantic identity와 별도의 host reliability/amplitude를 독립 조작한다. |
| [Hattori et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5806120/) | 반복 odor가 특정 α′3 MBON 반응을 dopamine-dependent하게 억제했지만 KC 감소는 더 작고 다른 구획은 같은 억제를 보이지 않았다. | 반복 억제는 전역 fatigue나 memory strength가 아니다. Early adaptation, stimulus-specific familiarity, 둘 다, 둘 다 없음을 비교하고 gap recovery와 specificity를 측정한다. |
| [Hige et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/) | 정의된 dopamine input과 odor pairing이 좁은 순서 규칙으로 odor-specific depression과 행동 학습을 만들 수 있었다. | 보편적인 “한 prompt면 충분” 규칙이 아니다. 정확히 한 action–outcome update 뒤 새 paraphrase를 검사하고 unpaired, reversed-order, shuffled-label control과 비교한다. |
| [Tully et al., 1994](https://pubmed.ncbi.nlm.nih.gov/7923375/) | Extended aversive training에서 구분되는 consolidated memory 성분이 나타났고 spaced와 massed protocol은 같은 memory form을 만들지 않았다. | Update 수가 같아도 schedule은 같지 않다. Reward 총량을 맞춘 one-shot, massed, spaced 조건 뒤 no-update retention을 검사한다. Static weight만 있는 feedforward policy는 명시적 dynamics 없이 consolidation을 보일 수 없다. |
| [Felsenberg et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5392358/) | 학습한 appetitive cue를 기대 reward 없이 다시 제시하면 조건에 따라 extinction/reconsolidation 관련 회로가 동원되었다. | Evaluation 중 omission/update가 있으면 반복 probe가 중립적이지 않다. No probe, read-only probe, expected outcome, omitted outcome을 분리하고 pure measurement에서는 weight를 freeze한다. |
| [Yang et al., 2026](https://www.nature.com/articles/s41593-026-02381-2) | Single-trial aversive training의 행동 표현이 사라진 뒤, 같은 context의 반복 reminder가 avoidance와 active MBON trace를 회복했다. 변형된 reminder는 원래 unpaired odor 회피를 만들 수 있었다. 실험한 조건에서 texture/light 변경은 회복을 막았지만 temperature 변경은 막지 않았다. | 낮은 행동 점수가 항상 숨은 기억을 뜻하지는 않는다. Acquisition checkpoint를 보존하고 paired, unpaired, novel, changed-context reminder를 비교한다. Weight를 바꾸는 reminder는 passive recall이 아니라 retraining이다. False-memory 표현에는 실제 원 episode와 misattribution control이 필요하다. |

## 제안 실험

1. **Cue-on 대 cue-off:** cue를 한 step 보여준 뒤 1, 2, 4, 8, 16 step 제거한다. Stateless, host-held observation, bounded recurrent estimator, oracle을 비교한다. Gap 동안 no motion, valid odometry, misleading odometry를 교차한다. Stateless 모델은 진짜 retention을 보이면 안 된다.

2. **한 update 대 연속 강화:** 한 outcome, massed 반복, 같은 수를 unrelated trial 사이에 spaced한 조건을 비교한다. Acquisition과 recall paraphrase family를 분리하고 reward 합과 optimizer exposure를 맞춘다. One-shot 주장은 정확히 한 causal update가 control보다 개선될 때만 가능하다.

3. **재학습 없는 retrieval:** acquisition checkpoint를 복제해 no re-exposure, read-only cue, cue+expected outcome, cue+omitted outcome으로 나눈다. 모든 probe 전후 weight를 기록한다. Read-only inference가 weight나 행동을 바꾸면 구현 오염이다.

4. **Strength/reliability/identity factorial:** normalized semantic vector는 고정하고 host amplitude/reliability를 별도 조절한다. Identity, amplitude, cue duration, presentation order를 완전 교차한다. Vector direction, norm, reliability, reward magnitude를 섞지 않는다.

5. **Adaptation 대 persistent recall:** exact repeat, 같은 의미 paraphrase, unrelated distractor를 같은 간격으로 제시한다. Early adaptation은 input-channel history, familiarity는 stimulus specificity, associative recall은 과거 outcome, working memory는 cue removal 뒤 성능으로 각각 구분한다.

6. **Silent-trace reminder challenge:** 한 update 뒤 immutable checkpoint를 보존하고, 사전 정의한 readout 약화 또는 decay로 행동이 control 범위에 들어가게 한다. Trained cue/context, 원래 unpaired cue, novel cue, changed context reminder를 비교한다. Inference-only와 update 허용 reminder를 분리한다. 현재 stateless feedforward engine은 inference-only reminder로 숨은 상태를 복구할 수 없다. Update 뒤 회복은 engineered system의 재학습 또는 reconstruction으로 보고한다.

## Confound와 수용 기준

- Intensity 순서를 무작위화한다. 고정 순서는 concentration effect를 바꿀 수 있다.
- Inference probe는 read-only로 둔다. Environment가 omission을 명시적으로 전달하고 learner가 update할 때만 omission event다.
- 반복 text를 생물학적 자극 시간으로 보지 않는다. Logical time과 duration을 정의한다.
- Vector direction/norm, host reliability, reward amplitude, update 횟수를 분리한다. 정규화는 norm 정보를 없앤다.
- Paraphrase family와 번역은 같은 split에 둔다. 한 checkpoint의 여러 trial은 독립 replicate가 아니다.
- Primary contrast를 사전 정의하고 seed-level effect를 보고한다.
- Working memory는 cue 제거 뒤 control 이상 성능과 truth-leak 부재가 필요하다. One-shot learning은 한 update가 만든 변화가 필요하다. Associative recall은 저장된 update와 이후 cue가 필요하다. Cue가 계속 보일 때의 성능만으로는 어느 것도 증명하지 못한다.
