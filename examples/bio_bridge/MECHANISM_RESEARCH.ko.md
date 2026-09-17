# 초파리 후각 회로에서 검증 가능한 메커니즘 추출

## 추출 가능한 것과 불가능한 것

Connectome edge table은 어떤 root ID가 연결되는지와 synapse count를 줄 수 있다. Receptor kinetics, membrane state, adaptation time constant, inhibitory gain, dopamine release, eligibility variable은 담지 않는다. ORN/PN–APL–KC–DAN–MBON 경로를 정확히 찾더라도 구조적 근거일 뿐이다.

아래 식은 **mechanism-inspired hypothesis**다. State와 상수는 선언하고 반증해야 하며 edge만으로 추출할 수 없다. Frozen EmbeddingGemma vector는 odor가 아니고, 명시적 host/synaptic state를 추가하지 않은 `novi` graph는 feedforward rate policy다.

## 1. ORN/PN adaptation과 gain control

[Gorur-Shandilya et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5524537/)은 naturalistic/Gaussian odor stream에서 ORN gain이 mean과 variance에 적응함을 측정했다. Mean gain은 주로 transduction, variance gain은 transduction과 spiking에서 나타났으며 complementary kinetics가 intensity 변화 속에서도 encounter timing을 보존했다.

[Olsen et al. (2010)](https://doi.org/10.1016/j.neuron.2010.04.009)은 adult antennal lobe의 feedforward와 lateral input을 독립 조작했다. PN normalization은 total ORN activity에 따라 커지고 saturation에 필요한 drive를 늘리며 response를 더 transient하게 했다. 이는 population divisive normalization을 지지하지만 모든 attenuation이 divisive라는 뜻은 아니다.

[Cafaro (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4831330/)은 연결된 ORN과 PN을 background/pulse에서 비교했다. PN spike adaptation이 더 컸고 background와 pulse adaptation에 서로 다른 site가 기여했다. 하나의 “sensory decay” state로 합치면 이 구분을 잃는다.

최소 state와 경쟁식은 다음과 같다.

```text
m_i(t+1) = (1-α)m_i(t) + αx_i(t)
subtractive: y_i = relu(x_i - βm_i)
divisive:    y_i = f(x_i)/(σ + βm_i)
population:  p_i = f_i(x_i)/(σ + γΣ_j f_j(x_j))
```

`α, β, γ, σ`는 connectome 값이 아닌 engineered constant다.

반증 조건:

- Subtractive는 threshold/silence를, divisive는 양수 영역 ratio scaling을 예측한다. Adaptation sequence로 fit하고 보지 않은 mean, variance, pulse amplitude에서 비교한다.
- Target channel history를 고정하고 다른 channel만 키운다. Population denominator는 cross-channel suppression을 예측하지만 local adaptation은 필수가 아니다.
- High→low/low→high 순서를 무작위화하고 onset, steady background, pulse, offset, recovery gap을 모두 넣는다.
- 평균 output을 맞춘 static gain, shuffled history, 매 tick reset state를 control로 둔다. 평균 attenuation만으로 feedback을 주장하지 않는다.

현재 engine의 max normalization은 positive global scale에 거의 invariant하지만 instantaneous/stateless다. Weber–Fechner adaptation, high-pass, recovery, cross-time feedback이 아니다.

## 2. APL inhibition과 KC pattern separation

[Lin et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/)은 KC→APL activation과 APL→KC inhibition을 조작했다. APL output 차단은 KC odor representation을 넓히고 correlation을 높였으며 비슷한 odor pair의 learned discrimination을 선택적으로 손상시켰다. 저자들도 sparsity만이 유일한 mediator임을 형식적으로 증명한 것은 아니라고 제한했다.

[Amin et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7541083/)은 local stimulation과 volumetric calcium imaging으로 APL 내부 propagation을 시험했고 spatially localized activity/inhibition을 지지했다. 하나의 uniform global scalar는 불완전한 생물학 표현이다.

```text
u = relu(W_PN→KC x)
global: a = mean(u) 또는 Σu; z_k = relu(u_k - βa)
local:  a_j = Σ_k L_jk u_k; z_j = relu(u_j - βa_j)
top-k:  z = keep_k_largest(u)
```

Top-k는 active count를 강제하지만 APL cell, feedback latency, GABA dynamics, local reciprocity가 없다. Policy에 되먹이지 않는 shadow-forward APL 계산은 audit probe이지 functional inhibition이 아니다.

반증 조건:

- Label/result를 보기 전에 similar/dissimilar pair를 고정하고 active fraction, representation overlap, margin, held-out discrimination을 측정한다.
- No inhibition, global, local, top-k를 비교하며 norm 또는 active count를 맞춘다.
- 핵심 예측은 interaction이다. Similar pair에서 overlap 감소와 discrimination 향상이 dissimilar pair보다 커야 한다. Uniform gain만으로는 불충분하다.
- `L`의 row sum을 보존한 shuffle, group 교환, parameter-matched global operator를 control로 둔다.
- Weak diagnostic feature+strong distractor mixture를 시험한다. 낮은 activity 자체가 좋은 separation은 아니다.
- Geometry 측정 때 readout을 고정하고 이후 matched readout을 학습해 encoding과 learning 변화를 분리한다.

실제 PN→KC/MB edge는 `W`에 존재 가능한 연결을 제한할 수 있지만 `β`, state dynamics, functional locality, APL-like causality를 정하지 못한다.

## 3. Dopamine timing과 eligibility

[Hige et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/)은 odor와 정의된 dopamine input을 pairing해 특정 MB compartment의 odor-specific depression을 측정했다. Plasticity는 상대 timing에 강하게 의존했고 공간적으로 compartmentalized했다. 보편 trace kernel의 증거는 아니다.

[Handler et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012144/)은 odor/reinforcement 순서에 따른 trial-by-trial behavioral reversal과 bidirectional KC–MBON plasticity를 보였고 DopR1/2 경로의 서로 다른 기여를 밝혔다. 정의된 odor assay와 MB 밖 receptor expression 때문에 일반적 local rule로 단정할 수 없다.

최소 eligibility 가설:

```text
e_t = λe_(t-1) + φ(x_t,a_t)
Δw_t = ηr_t e_t

또는 Δw = ηr K(Δt)φ,  Δt=t_reward-t_cue
```

`K`는 reward-before-cue와 reward-after-cue에서 크기나 부호가 다를 수 있다. `λ, η, K`는 가설이다. 과거 row/action 전체를 ID로 저장하는 queue는 exact delayed replay다. Identity credit은 풀 수 있지만 discrete record를 decay/superposition 없이 보존하므로 synaptic eligibility trace가 아니다.

반증 조건:

- Outcome-before-cue, simultaneous, cue-before-outcome을 포함한 signed lag 전체 곡선을 사전등록한다.
- Exact ID replay, exponential, rectangular, zero trace, wrong-current, shuffled timing/action을 비교한다.
- Reward amplitude를 맞춘다. 고정 lag에서 `exp(-lag/τ)`는 단지 작은 constant reward이므로 immediate amplitude control이 필요하다.
- Online delay 주장에는 실제 sequential sampling과 batch-1 update를 쓴다. Frozen rollout 안의 delay는 sampling phase에 가려진다.
- Full observation 무기한 저장을 피하고 local state와 memory를 bound한다. 그렇지 않으면 replay가 eligibility처럼 보인다.
- Pending cue가 여러 개인 superposition/crossed outcome을 시험한다. Scalar trace는 예측 가능한 interference가 있지만 exact ID map은 완전히 분리할 수 있다.
- Declared head/compartment routing과 shuffled routing을 비교한다. Software 이름은 MB compartment 대응 증거가 아니다.
- Read-only inference가 weight를 바꾸지 않고 reward가 한 번만 소비되는지 검증한다. Near-uniform argmax 때문에 accuracy와 weight change를 함께 보고한다.

## 통합 시험표

| Mechanism | 필요한 state | 핵심 조작 | Negative control | 실패 기준 |
|---|---|---|---|---|
| ORN local adaptation | channel별 running statistic | 현재 pulse 고정, history 변경 | reset/shuffled history | held-out recovery/history effect 없음 |
| PN population normalization | population drive, 선택적 slow depression | 다른 channel activity 변경 | matched static gain | cross-channel suppression 없음 |
| APL global | activity scalar | feedback 제거/scale | norm/active-count matched | similar-pair 선택 효과 없음 |
| APL local | localized vector | neighborhood 유지/셔플 | row-sum/parameter matched global | locality의 held-out 이득 없음 |
| Eligibility | decaying local feature/action state | signed lag | exact replay/amplitude controls | replay 또는 reward scale로 설명됨 |

각 mechanism은 먼저 따로 negative control을 통과해야 한다. 여러 flexible state를 한 번에 합치면 같은 행동을 맞추면서 어떤 원인이 작동했는지 식별하지 못할 수 있다.
