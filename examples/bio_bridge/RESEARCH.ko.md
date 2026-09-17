# 바이오 브리지: 초파리 연구에서 가져온 검증 가능한 제약

## 범위

이 문서는 실험 설계이며 새 실행 결과를 보고하지 않는다. 아래 연구는 일차 문헌이지만, 소프트웨어 구성은 생물학적 메커니즘의 재현이 아니라 영감을 받은 가설이다.

EmbeddingGemma의 고정 언어 벡터는 냄새 부호, 투사뉴런 패턴, Kenyon cell(KC) 집단 부호가 아니다. FunctionGemma의 typed tool 출력도 mushroom body 출력이 아니다. 두 인터페이스를 초파리의 전전두엽 상동물로 부를 근거는 없다. 행동과 나중 피드백을 연결하는 host-side queue는 실제 시냅스 eligibility trace가 아니라 감사 가능한 기록 장치다.

## 일차 연구와 제한된 소프트웨어 대응

| 연구 | 직접 관찰과 인과 한계 | 반증 가능한 소프트웨어 시험 |
|---|---|---|
| [Lin et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/) | APL 출력을 차단하면 KC 냄새 반응이 넓어지고 서로 더 상관되며, 비슷한 냄새의 학습된 구별이 선택적으로 손상되었다. 조작 자체가 sparsity만을 유일한 매개로 증명하지는 않는다. | 고정 embedding 뒤 top-k 경쟁을 넣고 입력 유사도와 k를 교차한다. 효과가 경계 근처가 아니라 모든 항목에 동일하거나 단순 scale 차이로 설명되면 기각한다. |
| [Baltruschat et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8741211/) | APL 억제는 KC 수상돌기/PN bouton 부근 반응을 정규화하며 상호작용은 국소적이다. 하나의 정확한 전역 정규화 공식으로 볼 수 없다. | 전역, feature-group별, 단순 rescaling을 비교한다. 강한 방해 신호 옆의 약한 진단 신호를 국소 방식이 더 잘 보존하는지 본다. |
| [Hige et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/) | 특정 MB 구획에서 냄새와 dopamine neuron 활성의 순서가 좁은 시간 창의 냄새 특이적 depression을 만들었다. 보편적 시간 상수는 아니다. | 실제 행동을 낸 head에만 feedback을 route하고 delay/order를 사전 정의하여 sweep한다. global 및 shuffled routing과 비교한다. |
| [Aso & Rubin, 2016](https://elifesciences.org/articles/16135) | dopamine cell type마다 학습 횟수, 지속, 용량, timing 규칙이 달랐다. 광유전학적 충분성은 모든 자연 보상이 같은 규칙을 쓴다는 증거가 아니다. | 서로 다른 고정 학습률/감쇠율을 가진 채널과 단일 global learner를 acquisition, reversal, retention에서 함께 비교한다. 채널 배정을 섞은 대조군을 둔다. |
| [Handler et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012144/) | 냄새와 강화의 순서가 행동 및 KC–MBON 가소성 방향을 바꾸었고 DopR1/2 경로의 기여가 달랐다. 수용체의 MB 밖 발현 때문에 완전한 국소 해석에는 한계가 있다. | cue-before-outcome, cue-after-outcome, unpaired 조건을 교차하고 부호 있는 순서 민감 업데이트를 시험한다. 단순 recency로 설명되면 기각한다. |
| [Felsenberg et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5392358/) | 기대한 보상 없이 cue를 다시 제시했을 때 조건에 따라 서로 다른 갱신 체계가 작동했다. 행동 약화는 곧 erasure가 아니다. | reminder 길이와 context를 바꾸고 renewal/reinstatement를 검사한다. overwrite와 dual-trace 모델의 held-out 예측을 비교한다. |
| [Felsenberg et al., 2018](https://doi.org/10.1016/j.cell.2018.08.021) | aversive extinction은 기존 기억 삭제보다 병렬적인 반대 기억의 형성과 통합으로 설명되었다. 정의된 후각 과제 밖으로 일반화할 수 없다. | A에서 학습하고 B에서 extinction한 뒤 A/B, 지연 회복, 무신호 outcome 후를 검사한다. B에서의 억제만으로 forgetting이라 부르지 않는다. |
| [Das et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3169145/) | 반복 냄새에 대한 habituation은 antennal lobe 억제성 local neuron 가소성과 연결되었다. 이는 semantic interpretation 이전의 감각 회로 현상이다. | 정확한 문자열 반복, embedding 유사도, 의미를 독립 조작한다. 초기 adaptation은 시간 경과 뒤 회복해야 하며 vector가 가깝다는 이유만으로 다른 의미를 familiar로 처리해서는 안 된다. |
| [Hattori et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5806120/) | 반복 냄새는 특정 α′3 MBON 반응과 alerting을 dopamine 의존적으로 낮췄지만 다른 구획의 변화는 달랐다. 전역 novelty scalar의 증거가 아니다. | 반복 이력 예측용 familiarity head를 정책과 분리한다. exact repeat, 같은 의미 paraphrase, 어휘는 비슷하나 의미가 다른 요청을 교차한다. |
| [Krashes et al., 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2780032/) | hunger가 dNPF와 소수 dopamine neuron을 통해 appetitive memory의 표현을 조절했다. 일반 utility 계산기의 증거는 아니다. | 동일 cue에서 host가 측정한 hungry/sated 상태를 바꾸고 state-gated, no-state, shuffled-state 정책을 비교한다. FunctionGemma가 숨은 상태를 만들어내게 하지 않는다. |

Suvrathan et al. (2016)은 초파리 MB 연구가 아니라 포유류 소뇌 Purkinje cell의 timing 연구이므로 위 근거 집합에서 제외한다. 비교적 경고로만 쓸 수 있다. 또한 fly heading 회로의 dopamine 가소성은 movement/head-direction dynamics와 관련될 수 있어 reward와 동일시하면 안 된다([Fisher et al., 2022](https://www.nature.com/articles/s41586-022-05485-4)).

## 사전등록 가능한 다섯 실험

1. **희소 분리:** label을 보기 전에 test pair를 frozen-embedding cosine 구간으로 고정한다. raw linear policy와 top-k, 전역 정규화, group 정규화를 같은 parameter/step/seed로 비교한다. 주 결과는 class 간 유사도에 따른 정확도 곡선이며, 언어 벡터가 냄새처럼 부호화된다는 주장은 하지 않는다.

2. **지연 credit:** host가 cue, 검증된 행동, timestamp, output head를 bounded queue에 기록하고 환경 전이 뒤에만 feedback을 낸다. delay와 순서를 class와 독립적으로 무작위화한다. no eligibility, exponential, rectangular, shuffled timestamp, outcome-before-action을 비교하고 전체 delay curve를 보고한다.

3. **retrieval/extinction/erasure:** context A에서 학습하고 B에서 기대 outcome을 생략하거나 반전한다. 즉시 및 지연된 A/B test와 unsignaled outcome 뒤 retest를 한다. 행동과 weight를 함께 검사한다. B에서 non-expression만 관찰해 삭제라고 해석하지 않는다.

4. **adaptation 대 semantic novelty:** exact repeat, 같은 의미 paraphrase, 어휘가 가까운 다른 요청, 무관 요청을 짧고 긴 간격과 교차한다. preprocessing gain, frozen vector, familiarity, action, latency를 각각 기록한다. 반복/시간 의존 감쇠와 task consequence로 정의되는 의미적 새로움을 분리한다.

5. **내부 상태와 grounded reward:** 같은 언어에서 host의 물·음식·온기·휴식 deficit을 바꾼다. 환경이 행동 뒤 상태 변화, 자원 비용, 안전 제약으로 reward를 계산한다. FunctionGemma가 reward나 hidden state를 제공하지 않는다. state-gated, concatenated, absent, shuffled 조건을 비교하며 stated request와 측정 deficit이 충돌하는 경우도 포함한다.

## 추론 안전장치

- paraphrase family와 scenario template 단위로 먼저 split하며 번역과 near-duplicate는 같은 split에 둔다.
- FunctionGemma 출력은 제안일 뿐이다. schema, 순서, state precondition을 mutation 전에 host가 검증한다.
- hidden state와 reward는 환경 전용이다. “배고프다”라는 문장은 신뢰도를 평가할 관찰이지 자동 ground truth가 아니다.
- seed별 불확실성과 trial별 변동을 분리한다. 한 정책의 많은 trial을 독립 biological replicate처럼 세지 않는다.
- 낮은 선택 확률을 erasure로, vector 거리를 novelty로, queue를 synaptic trace로, software head를 MB compartment로, 기능적 유사성을 neural homology로 해석하지 않는다.
