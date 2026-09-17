# 정밀한 embedding-to-fly 브리지 fitting

## 질문과 경계

공학적 질문은 frozen language embedding을 기존 319 input port에 어떻게 배치해야 고정 PN→KC topology가 유용한 의미 구조를 보존하는가이다. 이는 representation-fitting 문제다. 언어 관찰과 실제 파리 PN 활동의 paired data가 없으므로 파리의 감각 코드를 복원하는 실험이 아니다.

실제 connectome edge를 유지하면서 그 앞단의 인공 좌표계를 학습할 수 있다. 그러나 “topology 유지”와 “생물학적 sensory alignment 학습”은 다른 주장이다.

```text
해부학 주장: 선택한 PN→KC와 KC→MBON edge가 connectome에서 왔다
소프트웨어 주장: train-fitted transform이 언어 좌표를 input port에 할당했다
생물학적 alignment 주장: fly stimulus와 neural recording의 paired data가 필요하며 현재 없음
```

따라서 허용되는 결론은 하나의 frozen train-only transform이 다른 transform보다 이 engineered bridge에서 held-out semantic information을 잘 보존했다는 비교다.

## 1차 연구가 주는 생물학적 제약

### PN→KC 수렴은 확장적이고 희소하지만 완전히 random하거나 semantic하지 않다

[Turner, Bazhenov, Laurent (2008)](https://doi.org/10.1152/jn.01283.2007)은 in vivo KC odor response가 PN input보다 훨씬 sparse하고 selective함을 보였다. 따라서 embedding-to-port transform 뒤 active fraction과 pattern overlap을 측정해야 한다. 이 연구는 sentence embedding의 차원이 어떤 PN identity에 대응하는지 정하지 않는다.

[Honegger, Campbell, Turner (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869/)은 100개 이상의 KC를 population imaging하여 다양한 odor와 concentration에서 distinct sparse pattern을 관찰했다. 최근 odor history도 sparseness에 영향을 줬다. 소프트웨어 비교는 classification accuracy뿐 아니라 representation geometry와 sparsity를 보고해야 한다. 언어 paraphrase는 odor concentration series와 같지 않다.

[Caron et al. (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4148081/)은 성체 KC 200개의 glomerular input을 추적하여 대부분이 서로 다른, apparently random한 glomerulus 조합을 받는다고 보고했다. Mixing/expansion 비유를 지지하지만 language dimension이 named PN에 대응하거나 PCA axis가 glomerulus라는 뜻은 아니다.

[Eichler et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5806122/)의 larval MB reconstruction에서는 대부분의 KC가 apparently random input을 조합했지만 일부는 stereotyped single-PN input을 받았고 non-olfactory stream에는 구조가 있었다. Random projection 설명에는 발달 단계와 modality 한계가 있다.

[Zheng et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9413950/)은 성체 PN→KC connectome에서 주로 food-responsive PN type 사이의 above-chance convergence를 발견했다. 관찰 topology는 일반 discrimination에서는 randomized network보다 불리할 수 있었지만 특정 food-related input에는 이점이 있었다. 해부학 연결을 범용 최적 embedding projection으로 보면 안 된다.

[Li et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7909955/)은 adult hemibrain MB connectome에서 mixed olfactory input, subtype bias, 분리된 visual 및 thermo-hygro stream을 보였다. Topology는 edge를 제약하지만 modality assignment와 activity distribution도 생물학적으로 중요하다. Learned PCA coordinate를 PN root ID에 재할당하는 것은 engineered port convention이지 learned anatomy가 아니다.

### Representation alignment에는 paired observation과 독립 평가가 필요하다

[Haxby et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3201764/)은 paired fMRI response trajectory에서 orthogonal transform을 학습해 common representational space를 만들고 독립 experiment에 적용했다. 핵심 절차는 matched observation으로 alignment를 학습하고 fitting 밖의 data에서 평가하는 것이다. 여기서 pairing은 language embedding과 simulated hidden rate뿐이며 language와 fly neural activity가 아니다.

[Raghu et al. (2017)](https://proceedings.neurips.cc/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf)은 dimensional reduction과 CCA로 learned representation을 비교하는 SVCCA를 제안했다. Effective dimensionality와 affine coordinate 변화에 대한 불변성을 강조한다. 큰 output coordinate 수를 더 많은 정보라고 간주하지 말고 rank, singular value, held-out representation similarity를 보고해야 한다.

이 alignment 연구는 계산 방법을 제공하지만 파리와 언어 모델이 semantic space를 공유한다는 증거는 아니다.

## 제안된 transform 비교 감사

후보는 다음 세 가지다.

1. 고정 128차원 MRL mapping
2. centered 128차원 MRL mapping
3. train-only PCA를 128개 좌표에 저장하고 signed split하여 256 port에 넣은 뒤 319로 padding한 mapping

계획은 old train 32건, transform 선택용 validation 12건, untouched authored semantic-family set의 최종 평가다. Native graph topology는 그대로 두고 semantic embedding reconstruction과 별도로 reward-trained native readout을 측정한다. 이번 round에는 supervised class probe가 없다.

### 구현된 PCA128의 rank는 31이다

Train embedding matrix를 `X ∈ R^(32×D)`, training mean을 `mu`라 하자.

```text
X_c = X - 1 mu^T
X_c = U S V^T
z(x) = (x - mu) V_k
```

32개 observation을 centering하면:

```text
rank(X_c) <= min(D, 32 - 1) = 31.
```

128개 좌표를 요청해도 data가 지지하는 principal direction은 최대 31개다. 구현은 `768×128` projection의 앞 31개 column에 fitted PC를 저장하고 나머지 97개 column은 정확히 0으로 둔다. Signed split은 informative dimension 128개를 만들지 않는다. 31개의 informative scalar coordinate에서 최대 62개의 nonzero signed port가 생길 뿐이다.

따라서 **stored width**와 **fitted rank**를 모두 밝혀 `PCA128-rank31`로 설명해야 한다. 지원되지 않는 97개 PCA column과 일반 padding port 63개는 정확히 0이어야 한다. PCA coordinate 128개, signed port 256개, engine input 319개가 모두 독립적으로 fit된 정보를 담는다고 암시하면 안 된다.

PCA128-rank31은 compact interface 실험이지 “319 port를 모두 사용”하는 방법이 아니다. 31개보다 많은 informative direction을 추정하려면 독립 fitting observation이 더 필요하거나 평가 전에 선언한 외부 unsupervised corpus가 필요하다. 평가 text로 corpus를 fit하면 distribution leakage가 생긴다.

### Centering과 normalization 순서를 고정한다

MRL truncation, centering, projection, normalization, signed split은 순서를 바꿀 수 없다. 조건별로 정확한 순서를 사전 지정한다.

```text
fixed MRL128:
    q = L2_normalize(full_embedding[:128])

centered MRL128:
    q = L2_normalize(full_embedding[:128] - train_mean_128)

PCA rank r:
    q = L2_normalize((full_embedding - train_mean_full) @ V_r)

signed ports:
    p = concat(max(q, 0), max(-q, 0), zero_padding)
```

Mean과 PCA basis는 train 32행에서만 fit한다. Basis, mean, singular value, normalization 순서, model revision, hash를 저장하고 validation/test에 변경 없이 적용한다. 각 PC의 가장 큰 절댓값 loading을 양수로 강제하는 식의 deterministic sign convention도 선언한다. Sign은 PCA geometry를 바꾸지 않지만 port assignment와 artifact 재현성을 보장한다.

Centering 뒤 training mean과 가까운 sample은 zero 또는 near-zero가 될 수 있다. Zero-norm 동작을 실행 전에 정의하고, normalization이 collapse를 숨기지 않도록 전 norm을 보고한다.

### Port identity는 engineered assignment다

Rotation이나 PCA projection은 각 PN root ID에 들어가는 scalar를 바꾼다. 실제 edge table은 고정되지만 root에 할당한 input 의미는 바뀐다. 즉 fixed anatomy의 **interface를 fit**하는 것이며 anatomy를 학습하는 것이 아니다.

PN degree, synapse count, sign, KC target이 다르고 adult 연구에서 sampling bias도 발견되었으므로 PN→KC graph는 세부적으로 exchangeable하지 않다. PCA axis를 high-degree PN에 놓으면 같은 axis를 low-degree PN에 놓을 때보다 더 많은 KC에 영향을 줄 수 있다. Dataset variance와 graph degree의 우연한 정렬이 gain처럼 보일 수 있다.

필수 port-assignment 대조군:

- 같은 transformed coordinate를 eligible PN port에 배치하는 고정 random permutation 최소 5개
- coordinate magnitude rank에 할당되는 PN out-degree 분포를 유지하는 degree-stratified permutation
- PCA와 rank/norm이 같은 fixed random orthogonal rotation
- 지원되지 않는 PCA column 97개와 padding port 63개가 계속 0인지 검사
- 사전 지정 reference인 기존 fixed MRL mapping

Random seed 중 최고만 비교군으로 선택하지 말고 전체 분포를 보고한다.

## 평가 설계

### Split과 선택

개별 문장이 아니라 semantic family 단위로 split한다. 같은 scenario의 번역과 가까운 paraphrase는 모두 한 group에 둔다.

1. **Train 32:** mean, PCA, semantic ridge reconstruction, 별도로 정의한 reward learner fit
2. **Validation 12:** transform과 선언한 rank/normalization 선택
3. **Final authored-family test:** 선택 뒤 한 번만 실행하고 모든 case 보고

Validation 결과를 보기 전에 final authored family를 작성하고 hash로 고정한다. 작성자는 model output이나 validation failure mode를 보면 안 된다. Family 하나는 좁은 평가이므로 broad generalization을 지지하지 못한다. 가능하면 여러 사람이 독립적으로 작성한 scenario family를 쓰고 영어/한국어 번역은 같은 group으로 묶어 family-level 결과를 보고한다.

Old 32/12 사례가 이전 architecture나 active-fraction 선택에 영향을 줬다면 inherited development data라고 표시한다. 이 round에서 confirmatory set은 final family뿐이다.

### 두 downstream task는 주장이 다르다

**Semantic reconstruction**은 simulated activity `H_train`에서 원 embedding `E_train`으로 ridge map을 학습한다.

```text
B = argmin_B ||H_train B - E_train||_F^2 + lambda ||B||_F^2
E_hat = H_test B
```

Nearest-description category와 embedding cosine은 공급된 input geometry가 선형적으로 얼마나 남았는지 측정한다. Intent decoding이 아니다. `lambda`는 training만으로 고정하거나 validation selection에 포함하고 final test로 선택하지 않는다.

**Native reward learning**은 mapping별로 동일한 seed, example, update budget, optimizer 설정을 사용해 sampled action과 scalar reward로 기존 KC→MBON action readout을 학습한다. 해당 representation이 저장소의 action-learning 절차를 지원하는지 측정한다. Semantic reconstruction이 아니며 supervised class probe도 아니다.

두 결과를 합치지 않는다. Generic embedding geometry는 보존하면서 four-class boundary를 해칠 수도 있고 반대도 가능하다.

### Frozen policy와 retrained readout

Port transform을 바꾸면 native policy가 받는 distribution도 바뀐다.

- **Frozen native readout:** 기존 checkpoint와의 compatibility
- **Transform별 동일 budget으로 새 readout 학습:** 해당 representation의 learnability

Frozen 결과는 원 mapping으로 checkpoint를 학습했기 때문에 원 mapping에 유리할 수 있다. Retrained 결과는 새 mapping에 유리할 수 있지만 optimization variance가 추가된다. 가능하면 둘 다 보고하되 final test에서 checkpoint를 선택하지 않는다. Frozen-readout 손실을 정보 감소로 해석하면 안 된다.

## 핵심 metric과 대조군

합칠 수 없는 target별 primary metric을 사전 지정한다. Reconstruction에는 final-family semantic category accuracy, 별도 reward-trained native readout에는 final-family greedy action accuracy를 사용한다. 둘을 하나의 score로 평균내지 않는다.

Transform별 보고 항목:

- mathematical rank와 training singular-value spectrum 전체
- training explained variance와 validation/test reconstruction residual
- normalization 전후 input norm
- positive/negative signed port 수와 zero-port fraction
- KC active count/fraction, dead-case rate, normalization 전 row maximum
- input 및 KC stage의 within-class/between-class cosine 분포
- ridge reconstruction cosine과 nearest-description category
- reward-trained native action accuracy/probability와 confusion matrix
- random port permutation 및 optimization seed 전체

유용한 negative control:

1. native learning의 zero-reward 또는 shuffled-reward control
2. reconstruction의 shuffled `(hidden, embedding)` pair
3. constant-mean reconstruction
4. graph 전 identity/raw-embedding prototype baseline
5. random orthogonal rank-31 projection
6. vector norm을 보존한 permuted port assignment
7. 정보가 학습 앞단에 있는지 확인하는 untrained downstream readout

### Small-sample 안정성

Training row가 32개이므로 subspace rank가 정확히 제한될 뿐 PCA direction이 안정적이라는 뜻은 아니다. Semantic family 단위 bootstrap으로 PCA basis를 다시 fit하고 projection matrix 또는 principal angle을 비교한다. Singular value가 비슷하면 개별 PC correlation은 신뢰하기 어렵다. Bootstrap downstream variation은 보고하되 resample을 독립 dataset으로 취급하지 않는다.

PCA는 class label에 대해서는 unsupervised지만 training input에는 fitting된다. Author, language, 문장 길이, template variance를 잡을 수 있다. Nuisance를 회귀 또는 층화하고 language-held-out 결과를 비교한다. Family leakage 뒤의 높은 semantic score는 bridge를 검증하지 못한다.

## 계층형 architecture와 replay

과학적으로 해석 가능한 구현은 layer별로 freeze하고 audit해야 한다.

```text
Layer 0  frozen EmbeddingGemma vector
Layer 1  train-only interface transform: fixed, centered, PCA, control
Layer 2  fixed PN root ID에 할당한 signed port
Layer 3  fixed connectome PN→KC expansion과 선언한 sparsification
Layer 4a semantic reconstruction probe (측정 전용)
Layer 4b 별도로 reward-trained KC→MBON/native action readout (behavioral policy)
Layer 5  validation-only selection과 hash-locked final evaluation
```

각 artifact는 모든 upstream artifact hash를 선언해야 한다. Downstream decoder를 fit한 뒤 final 결과를 PCA나 port assignment에 다시 반영하면 안 된다. Layer 4a의 gain은 입력 정보 보존이고 Layer 4b의 gain은 reward-trained task performance다. 어느 쪽도 Layer 1을 anatomical하게 만들지 않는다.

이 실험에서 **replay는 reproducibility replay**를 뜻한다. Semantic decoder를 다시 fit하고 frozen input에서 저장 inference를 다시 실행한 뒤 tensor artifact와 prediction의 exact equality를 요구한다. Native replay는 seeded CPU reward-training 절차를 독립적으로 반복하여 development report와 checkpoint를 비교한다. Experience buffer가 아니며 development 뒤 training example을 추가하지 않는다.

Command contract는 holdout import/encoding, train-only port fitting, circuit/decoder development, protocol freeze와 verify, 한 번의 final evaluation, replay를 명시적으로 분리해야 한다. Hash lock은 source, data, transform, checkpoint, final output이 바뀌면 거부해야 한다. 이는 provenance 검증이며 fly memory replay, sleep consolidation, 생물학적 KC trace가 아니다.

## 판정 표

| 결과 | 해석 | 조치 |
|---|---|---|
| PCA128-rank31이 validation과 frozen final family에서 fixed/centered MRL보다 높고 random rotation/permutation은 그렇지 않음 | Dataset-fitted low-rank coordinate가 software interface를 개선 | Engineered candidate로 유지하되 axis를 PN semantics라고 부르지 않음 |
| PCA가 validation에서만 이김 | Selection overfit 또는 authored-family shift | Generalization 주장 폐기, exploratory로 보존 |
| Centered MRL이 PCA와 같음 | Principal-axis ordering보다 mean removal이 gain을 설명 | 더 단순한 centered transform 선택 |
| Random rotation/port permutation이 PCA와 같음 | PCA alignment의 특정 orientation을 지지하지 않음 | PCA 정렬에 효과를 귀속하지 않음 |
| 지원되지 않는 PCA column이나 padding이 nonzero가 됨 | Serialization, transform, indexing artifact | 해석 전에 debug |
| Reconstruction만 개선되고 reward-trained policy는 개선되지 않음 | Task benefit 없이 input geometry 접근성만 증가 | Reconstruction을 descriptive로만 유지 |
| Frozen policy는 하락하고 동일 budget 새 readout은 개선 | Old checkpoint와 distribution mismatch | 정보 손실이 아니라 interface migration으로 해석 |

## 수용 기준

정밀한 positive result에는 train-only transform artifact, centered training 32건에서 31 이하인 명시적 rank, validation-only model choice, 미리 작성하고 손대지 않은 family test, 완전한 baseline, family/bootstrap 분석에서 안정적인 방향이 필요하다. 정확한 case 수와 uncertainty를 기술적으로 보고하며 seed 3개를 독립 dataset 대신 사용하지 않는다.

깨끗한 positive result도 frozen language feature와 fixed fly-derived topology의 fitting 개선을 보여 줄 뿐이다. 생물학적 sensory alignment를 주장하려면 동시 stimulus와 PN/KC recording, 알려진 cell registration, train-fitted map을 held-out fly에 적용한 결과, aligned activity가 fly behavior를 예측하거나 바꾸는 causal test가 필요하다.
