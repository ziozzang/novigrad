# Novi state ↔ frozen LLM 양방향 인터페이스

## 범위

이 프로젝트는 engineered Novi state가 frozen FunctionGemma의 유효한 tool call 생성에 도움을 주는지, 검증된 call이 다음 Novi 입력을 바꾸는지 시험할 수 있다. 파리의 생각을 읽거나 neural language를 확립하거나 생물학적 brain–LLM interface를 입증할 수는 없다. 로컬 Novi rate는 언어 embedding과 connectome-inspired graph에서 나온 모사 상태이며 동물에서 기록한 신호가 아니다.

두 방향을 분리한다.

```text
read:  현재 Novi tensor -> resampler -> FunctionGemma conditioning -> tool call
write: 검증된 tool call -> 선언된 signed port transform -> 다음 Novi input
loop:  environment outcome -> trusted feedback -> 다음 state; reward는 model이 작성하지 않음
```

read 성공만으로 LLM이 state를 인과적으로 사용했다고 할 수 없다. write 성공도 지정 PN root ID에 그 의미가 생물학적으로 존재함을 뜻하지 않는다.

## 1차 연구가 보여 준 것

| 1차 연구 | 관련 결과 | 이 프로젝트의 한계 |
|---|---|---|
| [Li and Liang (2021)](https://aclanthology.org/2021.acl-long.353/) | 학습된 continuous prefix로 frozen language model을 조건화했다. | prefix는 trial마다 변하는 neural measurement가 아니라 task parameter였다. |
| [Lester et al. (2021)](https://aclanthology.org/2021.emnlp-main.243/) | soft prompt vector로 frozen LM을 적응시켰고 효과는 model scale에 크게 좌우됐다. | 더 작고 function-call 특화된 FunctionGemma에서의 효과는 직접 시험해야 한다. |
| [Alayrac et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) | Flamingo는 trainable Perceiver Resampler와 gated cross-attention으로 가변 visual feature를 frozen LM에 제공했다. | Novi neuron row는 visual token이 아니며 transformer block 변경은 최소 frozen-model 실험을 벗어난다. |
| [Li et al. (2023)](https://proceedings.mlr.press/v202/li23q.html) | BLIP-2는 frozen image model과 LM 사이에 learned Q-Former를 사용했다. | 대규모 paired image–text data의 결과를 작은 합성 goal label에 그대로 적용할 수 없다. |
| [Tang et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304553/) | 피험자별 fMRI encoding model과 LM으로 지각·상상한 연속 언어의 일부 의미를 복원했고 많은 paired data와 협조가 필요했다. | 인간 fMRI 연구이며 direct soft-prefix injection도, simulated fly state의 언어성 근거도 아니다. |
| [Ye et al. (2025), BrainLLM](https://www.nature.com/articles/s42003-025-07731-7) | 피험자별 adapter가 fMRI feature를 LLM text-embedding width의 vector로 바꾸고 preceding-text embedding과 연결했다. LLM은 고정하고 generative loss로 adapter를 학습했으며 aligned brain input을 permuted input 및 standard LLM과 비교했다. | 복원 대상 continuation에 반응한 기록과 피험자별 수백~수천 paired sample을 사용했다. 직접 continuous-input의 선례이지 simulated Novi feature가 언어와 대응한다는 근거는 아니다. |
| [Willett et al. (2023)](https://www.nature.com/articles/s41586-023-06377-x) | intracortical temporal feature를 phoneme 확률로 decode하고 LM과 결합했으며 held-out sentence와 raw decoder 지표를 보고했다. | LM이 decoder 오류를 교정할 수 있으므로 최종 text 품질만으로 neural channel 정보를 판단할 수 없다. |
| [Kim et al. (2017)](https://pubmed.ncbi.nlm.nih.gov/28473639/) | 파리 heading representation을 optogenetic하게 옮기자 회로가 바뀐 상태를 유지했다. | 특정 heading 회로의 read–perturb–read 논리이며 PN에 언어 concept를 쓰는 근거가 아니다. |
| [Zhang et al. (2021)](https://www.nature.com/articles/s41551-021-00736-7) | 동물 closed-loop BMI가 online neural-state decoding을 stimulation과 연결하고 행동 결과를 측정했다. | loop의 시간·인과 규율을 보여 줄 뿐 현재 software port와 생물 stimulation의 대응 근거는 아니다. |

최근 preprint 두 편은 확립된 1차 근거가 아니라 제안과 주의점으로만 사용한다. [Tang et al. (2026), NOBEL](https://arxiv.org/abs/2602.21522)은 EEG, MEG와 dual-path fMRI를 shared token space 및 LLM backbone에 정렬하고 direct stimulus input도 사용하는 구성을 제안한다. arXiv v1이므로 여기서는 multi-source alignment 질문만 참고하며 보고된 일반성이나 인과 해석을 전제하지 않는다. [Dhiman (2026)](https://arxiv.org/abs/2604.04033)은 connectome-constrained model의 겉보기 이점이 shared random initialization과 degree-preserving rewired control에서 줄어들었다고 보고한다. 이 single-author arXiv v1은 더 엄격한 topology control의 동기이지 Novi의 독립 검증이 아니다.

## 기존 로컬 인터페이스

현재 `function_bridge`에는 `set_goal`, `observe_heading`, `choose_action`, `get_status` 네 tool의 엄격한 grammar가 있다. `Session`은 완전한 call을 mutation 전에 검증하고 pending action을 하나만 허용하며 trusted environment-only hook으로 reward를 받는다. `NoviBackend`는 네 고정 goal 문장을 rate로 바꿔 native inference를 실행한다. 현재 FunctionGemma generation에는 text/schema만 들어가며 continuous Novi tensor는 들어가지 않는다.

새 실험은 parser를 약화하거나 reward를 model-callable로 만들지 않고 conditioning channel만 추가한다.

## 가장 작은 runnable 실험

로컬 BF16 FunctionGemma 270M weight를 완전히 고정한다. 선언된 Novi state snapshot을 model input-embedding width의 `K` vector로 바꾸는 작은 resampler만 학습한다.

```text
S_t in R^(N x F)                   # label이 아닌 선택 rate
Q in R^(K x d_r)                   # learned query parameter
H_t = cross_attention(Q, project(S_t))
P_t = layer_norm(H_t W_out)        # K soft input token
LLM input = [P_t ; embedded chat-template tokens]
```

첫 bounded run의 `S_t`는 PN port, pooled KC rate, MBON rate, 현재 heading encoding, pending-action flag처럼 이미 계산 가능한 소수 tensor로 고정한다. source를 각각 기록한다. target action, class label, expected tool name, future reward, post-action state는 포함하지 않는다.

FunctionGemma의 실제 `inputs_embeds`, attention mask, position 처리를 사용한 뒤 greedy generation과 기존 strict parser를 그대로 실행한다. token-level prefix가 주 조건이다. pooled control은 전체 snapshot을 하나의 vector 또는 반복된 하나의 vector로 만들되 trainable-parameter budget을 맞춘다. 비교 목적은 여러 prefix token이 source/temporal 차이를 보존하는지 보는 것이며 token을 neuron이라 부르지 않는다.

### 데이터 과제

동일 user text라도 관측 가능한 Novi state나 이력에 따라 다른 call/argument가 필요한 순차 episode를 만든다.

- 서로 다른 valid goal 뒤의 모호한 “계속해”;
- goal이 있고 pending action이 없을 때만 가능한 `choose_action`;
- environment feedback 뒤 goal switch;
- heading 보고 뒤 status 요청;
- target label이 아니라 bounded state가 이전 관측을 유지하는 cue dropout.

paraphrase나 한국어 번역을 만들기 전에 scenario family로 split한다. 기존 FunctionGemma train/holdout은 inherited diagnostic으로 보고할 수 있지만 양방향 주장에는 새 authored-family holdout이 필요하다.

## 필수 read-channel 대조군

모든 조건을 실제 frozen FunctionGemma generation과 strict parsing까지 통과시킨다.

1. **Text-only:** prefix 없는 현재 schema/chat 경로.
2. **Zero prefix:** shape와 mask는 같고 vector만 0.
3. **Row-shuffled state:** 필요하면 label strata 안에서 균형화해 다른 episode state를 제공.
4. **Time-shifted state:** 정당한 history 조건을 제외하고 이전/다음 episode state와 현재 text를 결합.
5. **Label-oracle prefix:** expected call/argument를 직접 encoding. 누출 상한이며 deployment에는 무효.
6. **Pooled prefix:** trainable parameter 수를 맞춘 하나의 pooled state vector.
7. **Token prefix:** `K`개 resampled vector.
8. **Structured-tool text:** 인과적으로 이용 가능한 state만 일반 text로 직렬화한 강한 해석 가능 baseline.

text-only 또는 structured state text가 soft prefix와 같다면 continuous interface의 추가 가치는 없다. shuffled state가 aligned state와 같다면 model이 Novi channel을 무시한다. label oracle만 성공하면 interface 배관은 작동하지만 측정 state가 부족한 것이다.

### Continuous injection과 reranking 비교

language prior가 지배할 수 있는 두 방식을 구분한다.

```text
continuous injection: state adapter -> soft token -> frozen LLM generation
reranking:             text-only LLM candidate -> state score가 candidate 선택
```

BrainLLM은 첫 방식을 permuted-brain 및 standard-LLM control과 비교했다는 점에서 직접 관련된다. Tang et al. (2023)은 candidate 탐색/점수화에 neural evidence를 사용했다. 이 프로젝트에서는 같은 candidate budget과 state feature로 두 방식을 실행한다. Reranking은 candidate set에 제한되며 continuous injection도 prefix를 무시한 채 fluent한 prior 출력을 낼 수 있다. task accuracy만으로 어느 쪽이 우월하다고 결론 내리지 않는다.

**Prior-only candidate coverage**를 추가한다. 올바른 serialized call이 text-only top-`k` generation에 이미 있는지를 측정하고, target이 이미 포함된 경우와 없는 경우의 개선을 나눠 보고한다. aligned state를 row-permuted state, scenario 안 time-shifted state, trainable constant prefix와 비교한다. text-only target surprisal별 결과도 보고한다. high-surprisal target에서 aligned-minus-permuted 차이가 커지면 aligned input이 generation을 바꾼다는 근거지만 여전히 software dependence 결과다.

target을 각 split 안에서 균형화하고 constant user prompt도 시험해 learned constant/class prior를 차단한다. 상충하는 text도 평가한다. text는 한 goal을 요구하고 state는 다른 goal에 대응하도록 만들되 어느 입력이 우선인지 사전에 정한다. 항상 text를 따르면 prefix가 사용되지 않은 것이고 항상 state를 따르면 context-sensitive arbitration 대신 shortcut일 수 있다. label-oracle 조건은 adapter 선택에 절대 사용하지 않는다.

connectome 기여를 주장하려면 locked biological graph 밖의 topology null을 추가한다. shared initialization의 degree-preserving rewire ensemble, input-degree strata를 보존한 port permutation, feature width와 norm을 맞춘 random projection을 사용한다. 총 edge 수만 맞춘 sparse random graph는 부족하다. 이는 계산적 반증 control이며 생물 회로가 언어를 구현하는지를 결정하지 않는다.

exact valid-call accuracy, schema rejection, tool-name accuracy, argument accuracy, negative-case refusal, generated-history rollout의 sequence success를 측정한다. 기술적으로 신뢰 가능한 경우 expected first call token의 log-probability margin도 보고하되 calibration 평가 없이 softmax를 calibrated confidence라 부르지 않는다.

## Write channel과 causal loop

완전히 parse되고 schema validation을 통과한 call만 write command를 만들 수 있다. proposed write와 committed write를 분리한다.

```text
generated bytes
  -> strict parse + schema/order check
  -> proposed command (mutation 없음)
  -> host state-machine check
  -> 선언된 gain/TTL의 signed port vector
  -> engine tick
  -> environment observation/reward
  -> 다음 read snapshot
```

`set_goal`은 bounded TTL의 engineered goal vector를 선언된 input port에 쓸 수 있다. `observe_heading`은 environment가 제공한 circular observation만 쓸 수 있고 FunctionGemma가 값을 만들어 내면 안 된다. `choose_action`은 native policy inference를 요청하지만 reward를 제공하지 않는다. `get_status`는 read-only다. write log에는 source call ID, port ID, signed value, gain, TTL, pre/post state hash, environment outcome을 남긴다.

최소 causal ablation은 committed write, no write, sign-flipped write, port-permuted write, delayed write, gain-matched random write다. write 뒤 native action이 바뀌면 engineered input에 대한 software sensitivity를 보인다. 생물학적 PN semantic role을 식별한 것은 아니다.

## Teacher forcing과 leakage

학습 때 teacher forcing으로 올바른 이전 tool call/action을 넣을 수 있지만 deployment에서는 model 자신의 parsed output을 받는다. 이를 섞으면 첫 오류가 뒤 상태를 바꾸는 실제 문제를 oracle history가 숨긴다.

두 mode를 분리 보고한다.

- **oracle-history diagnostic:** 올바른 이전 call/reward 제공; conditional prediction만 측정;
- **generated rollout:** validated generated call과 실제 environment outcome만 다음 step에 전달; invalid call은 선언된 no-op/error transition.

주 sequence success는 generated rollout을 사용한다. 같은 row의 target label로 만든 state에서 read resampler를 학습·평가하면 안 된다. normalizer, pooling, resampler parameter, early stopping, threshold는 train/development episode에서만 fit한다. 한 episode의 모든 time step과 paraphrase/translation은 같은 split에 둔다.

## 채택과 기각

aligned token prefix가 held-out scenario family에서 text-only, zero, shuffled, time-shifted, parameter-matched pooled control보다 좋고 generated rollout에서도 이득이 유지될 때만 soft-prefix mechanism을 유지한다. saved-weight exact replay와 unchanged FunctionGemma hash를 요구한다.

다음이면 bilateral 해석을 기각한다.

- final state에 target/tool/label field 또는 post-outcome feature 포함;
- shuffled/time-shifted state에도 이득 유지;
- oracle-history 성공이 generated rollout에서 소실;
- direct label-oracle prefix만 성공;
- PN write가 보고 state는 바꾸지만 사전등록 downstream action/reward는 바꾸지 못함;
- direct structured-text baseline이 prefix와 같음;
- final example이 resampler size, token count, prompt, checkpoint, stopping 선택에 영향.

채택되더라도 이는 frozen FunctionGemma와 Novi 사이의 engineered cybernetic loop다. 생물학 검증에는 실제 paired neural recording, controlled stimulation, behavior, held-out animal/session, causal perturbation이 필요하다.

## 구현 경계

새 코드는 기존 public protocol을 바꾸지 않는 전용 bilateral experiment module과 result 경로에 둔다. versioned episode manifest, split group, model/tokenizer hash, exact chat template, state-source schema, causal timestamp, resampler weight, optimizer history, generated byte, parse result, write log, environment outcome을 저장한다. development 선택을 동결한 뒤 final을 encode한다. replay는 saved weight로 prediction을 다시 만드는 것이며 biological replay가 아니다.
