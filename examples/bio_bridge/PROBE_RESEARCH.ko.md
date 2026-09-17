# 여러 계층의 probe와 학습 토큰 인터페이스: 실험 제안

2026-09-17 연구 메모. 새 실험 결과나 성능 우위를 보고하는 문서가 아니다. 먼저 현재 시뮬레이션의 여러 위치에서 같은 조건의 읽기용 probe를 비교하고, 이후 명시적인 상태 개입을 시험하는 순서를 제안한다. 큰 어댑터나 학습 토큰이 생물학적 정렬을 입증하지는 않는다.

## 현재 코드에서 가능한 일

| 위치 | 구현 | 의미와 한계 |
|---|---|---|
| 텍스트 표현 | `encode_precise_holdout.py`의 고정 FP32 인코더와 Classification 프롬프트 | 저장된 것은 정규화한 768차원 pooled 벡터다. 토큰열이나 transformer 중간 블록 표현은 별도 추출해야 한다. |
| PN 입력 | `precise_bridge.PortBridge`, `context_memory.encode_context`, native `Engine.infer[_batch]` | 순서가 정해진 입력 ID에 비음수 인공 rate를 넣는다. 임베딩 좌표와 문맥 코드는 측정된 PN 반응 특성이 아니다. |
| 희소화 전 KC | `inhibition_mechanism.ShadowEngine.raw_hidden` | 고정 투사 **후 ReLU까지 적용한** 값이다. ReLU 이전 값은 별도 hook이 필요하다. |
| 희소화 후 KC | `ShadowEngine.inhibit`, Rust `Engine::hidden_activity` | 억제·정규화된 값을 읽을 수 있다. 확인한 Python binding은 hidden ID를 노출하지만 hidden activity getter나 임의 내부 상태 쓰기 API는 제공하지 않는다. |
| MBON과 행동 | `thought_embedding.mbon_for`, `ShadowEngine.probabilities`, native 확률 출력 | shadow MBON 벡터와 외부에서 그룹·gain을 적용한 행동 출력은 서로 다른 대상이다. |
| 시뮬레이션 개입 | `thought_interventions.intervene`와 shadow 후단 계산 | KC 값을 바꾸고 후단을 다시 계산하는 개입이다. native 실행 중 상태 주입이나 동물 자극을 뜻하지 않는다. |

`context_memory.encode_context`는 문맥 포트를 추가하거나 정규화하고, 또는 다른 입력 블록으로 특징을 보낸다. modular 정책은 별도 engine을 선택할 수 있다. 이는 시뮬레이션 입력·경로 변경이다. 반면 KC 벡터와 문맥을 외부 decoder에 이어 붙이는 것만으로 KC·MBON 내부 상태에 문맥이 쓰이지는 않는다. 전극에 비유하려면 기록인지 자극인지, 위치·세기·시간·후단 경로를 명시해야 한다. 해부학적 ID는 주소이며 학습 토큰의 의미가 아니다. 현재 인터페이스에는 실제 전극 기록·자극 장치가 없다.

## 원 연구가 제안하는 것

- **Prompt와 prefix tuning:** prompt tuning은 고정 모델의 입력 임베딩을 학습한다. prefix tuning은 내부 계층 활성도 포함하는 연속 조건 벡터를 가상 토큰처럼 참조하게 한다. 신경 신호로 토큰을 만들었다면 실제 후단 transformer가 소비해야 한다. 벡터를 펼쳐 분류기에 넣는 것과는 구별한다. [Prompt tuning](https://arxiv.org/abs/2104.08691), [prefix tuning](https://arxiv.org/abs/2101.00190).
- **Perceiver Resampler:** Flamingo는 시각 특징을 고정 개수의 latent 토큰으로 줄여 후단 cross-attention에 연결한다. 이를 위치·시간별 신호에 적용하는 것은 구조적 응용 제안이지 Flamingo 학습의 재현이 아니다. [Flamingo](https://arxiv.org/abs/2204.14198).
- **Q-Former:** BLIP-2는 고정 인코더와 언어 모델 사이에 query 벡터와 transformer를 학습한다. 원래 Q-Former 자체가 188M 파라미터이므로 작은 query attention 모듈을 같은 모델이라고 부르면 안 된다. 작은 구현은 차이를 밝힌 query-resampler 기준선으로 명명한다. [BLIP-2](https://arxiv.org/html/2301.12597v3).
- **LoRA:** 실제 transformer LoRA는 지정한 transformer 가중치 행렬에 저랭크 갱신을 추가한다. 임베딩에서 PN으로 가는 저랭크 행렬만 학습했다면 **bridge adapter**다. 같은 행렬 분해를 쓴다고 transformer LoRA가 되지 않는다. 수정한 모듈과 파라미터 해시를 기록한다. [LoRA](https://arxiv.org/abs/2106.09685).
- **mHC는 Manifold-Constrained Hyper-Connections:** 잔차 스트림을 확장하고 Sinkhorn-Knopp로 혼합 행렬을 doubly stochastic하게 제약한다. 생물학적 연결을 보장하거나 전체 신경망의 모든 변환을 제한하는 방법은 아니다. PN/KC/MBON 분기를 제약 혼합한다면 mHC에서 착안한 bridge이지 transformer mHC 구현은 아니다. [원 논문](https://arxiv.org/html/2512.24880v2). 후속 PEFT 연구도 단독 mHC가 LoRA보다 일관되게 낫다고 보고하지 않는다. 비교할 근거이지 우위의 보증은 아니다. [PEFT 연구](https://arxiv.org/abs/2607.18130).

## 위치 × 용량 × 토큰 비교

인코더·연결 구조·과제·학습 예시를 먼저 고정한다. 작은 개발 실험은 PN/희소 KC/MBON 세 위치, linear/rank-8/rank-32 세 bridge 용량, pooled 벡터/학습 query 4개 두 인터페이스의 18조건과 동일한 세 seed로 구성할 수 있다. Query projection도 학습 파라미터 예산에 포함한다. 위치마다 입력 차원이 다르므로 같은 rank만으로 공정한 용량 비교가 되지 않는다. 예산 오차 허용치를 정하고 초과 조건은 별도 표기한다.

개발 데이터에서만 후보를 좁힌 후, 미리 정한 후속 비교로 억제 전 KC와 transformer 초기·중간·후기 블록, 토큰 수 1/4/16, full-rank bridge와 2층 MLP를 추가한다. **Bridge 전체 학습**과 **인코더 전체 미세조정**은 다르다. Transformer 계층 읽기와 회로 위치 읽기도 같은 개입 축으로 합치지 않는다. 위치별 원래 차원과 공통 병목 차원을 함께 보고하고, 차원 축소·정규화는 학습 데이터만 사용한다.

Transformer 적응은 별도 비교군으로 둔다. 고정 인코더, 지정 attention projection의 실제 LoRA rank 8/32/64, 선택적으로 인코더 전체 미세조정이다. 갱신 파라미터 수, optimizer 메모리, 모델·tokenizer 버전, 조기 종료 규칙, 시간·최대 메모리를 기록한다. 기존 학습 문장 32개로 300M 인코더 전체 학습의 일반적 우위를 주장할 수 없으므로 더 큰 독립 개발 자료가 필요하다. Hard top-k를 거치는 미분 surrogate는 별도 검증 대상이다. 그렇지 않으면 bridge의 지도·복원 학습과 native sampled-reward 학습을 분리하고 end-to-end 역전파라 부르지 않는다.

토큰은 위치·시간 벡터와 mask·metadata를 명시해 구성한다. Pooled projection, 고정 무작위 query, 학습 query, 채널·ID metadata 교란을 비교한다. 뉴런별 scalar에는 attention 전에 표현 투사가 필요하다. ID와 활성값을 분리하고 토큰 이름에 정답을 넣지 않는다. Transformer prompt/prefix 주입 경로와 native PN rate 주입 경로도 별도로 취급한다.

mHC 후보는 동일 폭의 identity routing, 무제약 혼합, row-stochastic 혼합, doubly stochastic 혼합을 같은 파라미터·계산 예산으로 비교한다. 유한 Sinkhorn 반복 후 행·열 합 오차, gradient·활성 norm, 처리량과 loss를 기록한다. 부호 있는 시냅스 가중치를 비음수 확률 행렬로 바꾼 뒤 원래 생물학적 회로라고 부르면 안 된다.

## 강한 반증 기준

1. **직접 경로가 동등하거나 우수함:** 같은 용량의 embedding→action 모델과 비교한다. 회로가 새로운 상황군의 일반화나 견고성에 이득을 주지 않으면 회로의 필요성 주장을 지지하지 못한다.
2. **무작위 구조가 해부학적 구조와 동등함:** degree·부호·예산을 맞춘 재배선과 무작위 확장을 비교한다. 효과가 같으면 해부학적 배치에 특화된 계산이라는 해석이 약해진다.
3. **주장한 경로를 지워도 decoder가 잘 작동함:** 문맥과 decoder를 고정하고 해당 위치를 영점화·교란·교체한다. 행동이 그대로라면 그 경로는 필요하지 않다. 활성 수와 개입 에너지를 맞춘다.
4. **문맥만으로 행동을 맞힘:** 모든 의미 단서를 모든 문맥 규칙과 균형 있게 교차한다. 같은 단서/다른 문맥, 같은 문맥/다른 단서를 모두 시험한다. 의미 라벨과 규칙에 따른 행동 라벨을 분리한다. 단서를 섞거나 문맥만 주어도 같다면 구조의 의미 처리 주장은 실패한다.
5. **토큰이 단지 추가 파라미터임:** 학습 query는 작은 linear 모델뿐 아니라 고정 query와 같은 예산 MLP도 넘어야 한다. 토큰 순서·개수 실험으로 ID와 위치 사용을 확인한다.
6. **mHC 제약이 역할을 하지 않음:** 같은 계산량의 identity·무제약 혼합이 성능과 안정성에서 같다면 이 과제에서 제약 구조의 필요성은 입증되지 않는다.

읽기용 probe의 성공은 그 decoder가 정보를 읽을 수 있다는 뜻이다. 회로가 실제로 그 정보를 사용한다는 뜻은 아니다. 쓰기 실험은 덧셈/교체, 비선형 전후 위치, 세기, 시점, 후단 정책 재계산을 명시한다. 교란 후 원래 상태를 복원하는 rescue 검사가 saliency 그림보다 강한 근거가 된다.

## 데이터와 보고 경계

개발 자료는 상황군·출처·세션 단위로 나누며 한영 번역 쌍을 같은 그룹에 둔다. 불확실성은 쌍 또는 상황군 단위로 계산한다. 새 architecture 최종·적대적 세트는 각각 64문장이지만 바탕 상황은 32쌍이고, 외부 검증 자료가 아닌 직접 작성한 자료다. 적대적 세트의 부정·인용·해결된 필요·정정·가정·제삼자·시간 변화·단어 미끼 역시 템플릿 공유에 따른 의존성이 있다.

위치·용량·rank·토큰 수·종료 규칙·후보 정의를 개발 데이터에서만 고른다. 최종 두 세트를 인코딩하거나 채점하기 전에 해시를 고정한다. 이미 확인한 최종 자료는 이름을 바꿔도 새 검증 자료가 아니다. 사전 등록한 모든 조건과 정확도, 최악 상황군·언어 성능, 보정도, 쌍 단위 불확실성, transformer 수정 후 무관한 언어 과제의 망각, 전송 포함 지연시간을 보고한다. 이 문서를 위해 예측이나 새 실험을 실행하지 않았다.
