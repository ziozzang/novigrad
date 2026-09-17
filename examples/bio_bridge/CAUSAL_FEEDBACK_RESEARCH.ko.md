# Novi–LLM bridge의 인과적 행동–환경 feedback

## 범위

다음 실험은 생성된 행동이 외부 환경 상태를 바꾸고, 그 결과 관측이 다음 결정을 개선하는지 물어야 한다. 모델 자신의 이전 출력으로 고른 prototype을 다시 입력하는 것보다 강한 시험이다. 그러나 여전히 engineered software experiment다. Novi rate는 모사 신호이고 환경은 사람이 설계하므로, 결과가 파리의 생각·dopamine 생리·생물학적 brain–LLM interface를 입증하지 않는다.

세 신호를 분리한다.

```text
action copy:       c_t = copy(a_t), 결과 전에 사용 가능
sensory residual:  epsilon_(t+1) = o_(t+1) - F(s_t, a_t)
value error:       delta_(t+1) = clip(r_(t+1) + gamma V(s_(t+1)) - V(s_t))
```

action copy는 자기 행동의 감각 결과를 예측한다. sensory residual은 예측과 실제 다음 관측의 차이다. value error는 보상 기대의 오차다. 모두 “feedback”이라는 하나의 신호로 합치면 안 된다.

### 첫 assay의 구현 범위

첫 구현은 아래에서 제안하는 dynamic experiment보다 의도적으로 약하다. 고정 demand, 최대 네 번의 결정, 실행 행동이 demand와 같을 때의 종료로 구성된 four-choice hidden-goal bandit이다. feedback은 실행 행동과 성공/실패 reward를 제공하지만 locomotion, resource dynamics, learned forward model에서 나오지 않는다. 따라서 reward는 이 assay 안의 target-class matching이며 물리적으로 근거를 둔 goal discovery 신호가 아니다.

FunctionGemma와 Novi는 고정돼 있다. host Bayes rule이 네 supervised class prototype에 대한 belief를 갱신해 continuous prefix를 제공한다. model learning, dopamine prediction error, sensory residual, recurrent world-model learning은 없다. 이번 run은 outcome을 반영한 host stimulation이 self-writeback 등 control과 비교해 이후 frozen-LM 선택을 바꾸는지를 시험할 수 있다. autonomous goal discovery, 일반적인 closed-loop control, 생물학적 feedback mechanism은 입증할 수 없다. 아래의 dynamic environment, expectation-matched prediction-error, efference-copy perturbation은 후속 제안이다.

## 1차 연구와 적용 한계

| 1차 연구 | 실제 측정·개입 | 이 software 연구의 제약 |
|---|---|---|
| [Felsenberg et al. (2017)](https://www.nature.com/articles/nature21716) | 기대 보상의 제공/누락에 따라 재활성화된 appetitive memory가 extinction 또는 reconsolidation으로 달라졌고, MBON–dopamine-neuron recurrence의 compartment-specific 역할을 시험했다. | 기대와 결과를 분리해 시험해야 한다. software reward를 측정된 DAN signal이라 부르거나 하나의 전역 dopamine scalar로 일반화할 수 없다. |
| [König et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7893153/) | imaging, optogenetics, 행동으로 reversal 중 예상 shock의 누락을 reward로 encode하는 특정 mushroom-body relay를 밝혔다. | outcome의 단순 존재가 아니라 학습된 기대가 중요하다. 특정 compartment와 과제의 결과다. |
| [Kim, Fitzgerald, and Maimon (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6327952/) | tethered flying fly의 whole-cell recording에서 saccade의 예상 visual consequence를 상쇄하기에 적절한 시간·부호의 cell-type-specific motor input을 관찰했다. | 결과 전 action copy와 그 시간·부호를 시험하되, 모든 motor modulation을 하나의 vector로 보거나 외부의 예상 밖 motion까지 제거하면 안 된다. |
| [Shanechi et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/27035820/) | non-human-primate closed-loop BMI에서 intention estimation과 adaptive neural decoding을 나누고 여러 session의 control을 평가했다. | offline accuracy만으로 부족하다. 상호작용 중 trajectory, 오류 회복, 지연, 안정성을 측정해야 한다. |
| [Zhang et al. (2021)](https://www.nature.com/articles/s41551-021-00736-7) | 자유 행동 동물에서 online neural-state detection이 intervention을 유발하고 그 행동 결과를 측정했다. | detect–act–measure 규율의 근거이지 Novi port와 생물 stimulation의 동등성 근거가 아니다. |

## 최소 실행 실험

네 need를 가진 bounded two-step resource environment를 사용한다. 각 episode는 기록된 hidden environment state에서 시작하고 일부 관측만 노출한다. FunctionGemma가 schema-valid action을 제안하면 환경이 transition을 실행하고 다음 관측과 보상을 반환한다. model output이나 target label은 transition에 들어가지 않는다.

```text
(hidden x_t, observation o_t)
  -> frozen embedding/Novi state -> frozen FunctionGemma tool call
  -> parse/order validation -> committed a_t
  -> environment.step(a_t, rng) -> x_(t+1), o_(t+1), r_(t+1)
  -> optional forward prediction F(s_t, a_t)
  -> residual/value-error update -> next decision
```

resource에는 실제 동역학을 둔다. 물을 마시면 thirst가 줄지만 수원이 고갈되고, 회전하면 보이는 장소가 달라지며, wind나 obstacle 때문에 같은 action의 결과가 달라질 수 있다. 보상은 need 감소와 collision cost 같은 사전 등록된 상태 변화로 계산한다.

주 지표는 generated-rollout episode return과 goal completion이다. transition prediction error, 예상 밖 perturbation 뒤 회복 step, invalid-call rate, action/observation latency도 보고한다. per-step classification은 진단 지표다.

## 반증 가능한 control

1. **No-effect action:** call은 검증하지만 환경은 바꾸지 않는다. closed-loop 이득이 사라져야 한다.
2. **Yoked outcome:** marginal은 맞추되 다른 episode의 observation/reward를 재생해 현재 action–outcome 관계를 끊는다.
3. **Action shuffle:** 제안 행동을 기록한 뒤 균형 잡힌 다른 행동을 실행한다.
4. **Open-loop replay:** action이 이후 입력을 바꾸지 못하는 동일 prerecorded sequence를 별도 평가한다.
5. **Action-copy ablation:** parameter budget과 관측은 유지하고 copy를 제거·지연·부호반전하거나 이전 action으로 바꾼다.
6. **Unexpected perturbation:** 행동 뒤 외부 motion을 추가한다. 올바른 cancellation은 예측된 self-generated 부분만 제거하고 unexpected residual은 남겨야 한다.
7. **Expectation control:** 실제 reward는 같게 두고 학습된 기대만 바꾼다. 예상 reward에는 작은 update, surprise reward에는 큰 update가 나와야 한다.
8. **Omission/delay:** 예상 outcome을 누락하거나 eligibility window 밖으로 지연한다. exact-order, wrong-order, reward-shuffled, magnitude-matched 조건을 비교한다.
9. **강한 baseline:** text-only state serialization, observation-only, action-copy-only, 동일 관측의 direct analytic controller, deployment에 무효라고 표시한 label-oracle ceiling.

동일한 pre-action state와 environment RNG seed에서 paired intervention을 실행한다. `같은 state + 다른 action`과 `같은 action + 다른 disturbance`를 모두 시험해야 self-confirming prototype loop에 없던 transition dependence를 확인할 수 있다.

## 순환 근거 방지

- `environment.step` 뒤에만 다음 입력을 만든다. future observation/reward는 현재 prefix, normalizer, target에 들어갈 수 없다.
- action copy는 outcome을 예측할 수 있지만 outcome이 실제 발생했다는 증거가 아니다.
- proposed, validated, committed, executed action을 분리한다. invalid call은 선언된 no-op/error transition을 만든다.
- forward model, value function, normalizer, threshold는 train/development episode에만 fit한다. 한 episode의 모든 time step과 언어 변형은 같은 split에 둔다.
- 주 결과는 generated history를 쓴다. oracle-history와 teacher forcing은 ceiling이다.
- final 평가 전에 environment transition/reward code와 final episode manifest를 freeze한다.
- episode마다 recurrent state, queue, RNG stream, pending action을 모두 reset한다.

raw generation, parsed call, pre/post environment hash, proposed/executed action, RNG seed, observation, reward, predicted observation/reward, residual, update ID, causal timestamp를 기록한다. saved replay는 이 log에서 transition과 결정을 재현해야 한다.

## 기각 기준

no-effect action, yoked outcome, action shuffle에서도 이득이 유지되거나, reward가 환경 결과가 아니라 expected class로 계산되거나, action copy만으로 성공을 관측했다고 간주하거나, oracle history에서만 효과가 있거나, perturbation trial에서 wrong-sign/wrong-time copy가 aligned copy와 같다면 causal-feedback 해석을 기각한다.

통과해도 입증되는 것은 engineered action–environment–observation loop가 이 frozen-model system을 개선한다는 사실뿐이다. 생물학적 정렬에는 동물의 paired neural recording, 측정 행동, identified compartment, causal perturbation이 필요하다.
