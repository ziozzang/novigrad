# 정밀 embedding-to-fly 브리지 결과

## 범위

이 실험은 EmbeddingGemma feature를 고정된 fly-derived PN→KC topology에 넣는 세 frozen mapping을 비교한다. Semantic reconstruction과 별도로 reward-trained native action policy를 측정했다. 언어를 실제 파리 감각 활동에 align하거나 PN semantics를 식별하거나 fly cognition을 입증하지 않는다.

Protocol은 [PRECISE_RESEARCH.ko.md](PRECISE_RESEARCH.ko.md)를 따른다. Schema-1 protocol lock이 final 평가 전에 data, source, mapping, decoder, checkpoint, permutation, environment hash를 고정했다. Final set은 fitting이나 selection에 사용하지 않았다.

![정밀 브리지 비교](../../results/precise-bridge/comparison.png)

Machine-readable 개요: [summary.json](../../results/precise-bridge/summary.json). Vector figure: [comparison.svg](../../results/precise-bridge/comparison.svg).

## 계층형 실험

```text
authored text
  -> frozen EmbeddingGemma-300m, Classification prompt, normalized 768D
  -> train-only port transform
       fixed:    앞 128 MRL coordinate
       centered: 앞 128 coordinate - train mean
       PCA:      저장 좌표 128개, fitted rank 31
  -> positive/negative split: nonnegative port 256개 + zero padding port 63개
  -> fixed fly-derived PN→KC topology + native 2% top-k (104/5,177 KCs)
  -> 둘 중 하나:
       semantic ridge -> 128D input semantics 복원 -> candidate description
       native KC→MBON policy -> sampled-action reward learning -> 네 action
```

Training row 32개만 mean, PCA, semantic ridge decoder, native policy fitting에 사용했다. Inherited validation 12개에서 semantic ridge `{0.01, 0.1, 1}`을 category accuracy, cosine, 작은 ridge 순으로 선택했다. Native reward-training 설정은 validation 전 고정했으며 semantic reconstruction으로 선택하지 않았다.

Final set은 authored text 64개다. Scenario 32개마다 영어와 한국어 text 하나씩 있고 water, food, warmth, rest에 균형을 맞췄다. 번역 pair는 종속 observation이다. 이번 round의 새 authored-family test이지만 외부 benchmark나 생물학 dataset은 아니다.

## Development 선택

| Mapping | Mathematical rank | 선택된 semantic ridge | Validation semantic accuracy | Validation cosine |
|---|---:|---:|---:|---:|
| Fixed MRL128 | 128 | 0.01 | 83.33% | 0.8526 |
| Centered MRL128 | 128 | 0.1 | 83.33% | 0.8472 |
| PCA128 | 31 | 1.0 | 91.67% | 0.8515 |

PCA128은 `768×128` projection을 저장하지만 training row 32개를 centering한 뒤 fit 가능한 direction은 31개뿐이다. 나머지 projected coordinate 97개는 0이다. Signed split과 padding은 정보 rank를 늘리지 않는다.

Development split은 이전 연구에서 재사용되었다. 이 score는 selection evidence이며 독립 confirmation이 아니다.

**Frozen provenance 정정:** 실행 코드의 selection key는 `(accuracy, cosine, -ridge)`이므로 accuracy와 cosine이 모두 정확히 같으면 작은 ridge를 선택한다. Frozen `development.json` 안의 narrative string은 잘못해서 “larger regularization”이라고 기록했다. 실제 선택 trial은 앞의 두 metric이 모두 같은 tie가 아니었으므로 선택과 결과에는 영향이 없다. Hash와 exact replay를 유지하기 위해 frozen metadata는 수정하지 않았다.

## Final semantic reconstruction

Semantic decoder는 native KC activity에서 원래 128차원 MRL target으로 가는 ridge map이다. Nearest-description category는 closed-catalog semantic retrieval이다. Text generation, calibrated confidence, intent, value가 아니다.

| Mapping | 전체 64 | 영어 32 | 한국어 32 | Mean embedding cosine | Exact catalog identity |
|---|---:|---:|---:|---:|---:|
| Raw embedding baseline | 90.63% | — | — | 1.0000 | 100.00% |
| Fixed MRL128 → KC → ridge | 59.38% | 68.75% | 50.00% | 0.7434 | 25.00% |
| Centered MRL128 → KC → ridge | 64.06% | 87.50% | 40.63% | 0.7430 | 18.75% |
| PCA128-rank31 → KC → ridge | **93.75%** | **100.00%** | **87.50%** | 0.7383 | 10.94% |

PCA는 이 authored set의 category에서 fixed mapping 두 개와 raw-embedding baseline을 모두 넘었다. 하지만 이는 각 문장을 충실히 복원했다는 뜻이 아니다. PCA의 exact catalog identity는 가장 낮았고 mean cosine도 조금 낮았다. Low-rank transform이 case-specific detail을 버리면서 four-category 구분에 유용한 차이를 강조했다.

영어/한국어 gap도 중요하다. Centering은 fixed보다 영어 category를 높였지만 한국어는 낮췄다. PCA는 두 언어 모두 높았으나 한국어 32행은 영어 scenario의 번역 pair이므로 독립 replication이 아니다.

64개 row는 authored scenario family 8개에 속한다. Family-cluster bootstrap 10,000회에서 PCA-minus-fixed category-accuracy 차이는 34.38 percentage point였고 percentile interval은 25.00~42.19 point였다. 이 interval은 작은 authored family 8개를 resampling했을 때의 기술적 민감도다. Population confidence interval, 외부 generalization 추정, 독립 수집 family의 대체물이 아니다.

Source: [final.json](../../results/precise-bridge/final.json).

## Native reward-trained policy

이는 semantic ridge calibration과 target과 training process가 모두 다르다. Mapping과 seed 701–703마다 fresh native engine이 64 epoch 동안 sampled action 2,048개를 batch 8로 받아 optimizer update 256회를 수행했다. Correct sampled action에는 scalar `+1`, 그 외에는 `-1`만 받았다. Teacher-action update는 없다. Seed 안에서 mapping들은 epoch permutation과 action uniform을 공유했지만 sampled action은 달라질 수 있었다.

| Mapping | Final greedy accuracy, seed 3개 평균 | 영어 평균 | 한국어 평균 | Mean correct-class probability | Seed별 accuracy |
|---|---:|---:|---:|---:|---|
| Fixed MRL128 | 41.15% | 44.79% | 37.50% | 0.2554 | 34.38%, 48.44%, 40.63% |
| Centered MRL128 | 58.85% | 73.96% | 43.75% | 0.2627 | 59.38%, 59.38%, 57.81% |
| PCA128-rank31 | **92.19%** | **98.96%** | **85.42%** | 0.2772 | 93.75%, 92.19%, 90.63% |

PCA policy 결과는 별도 objective를 사용했지만 semantic reconstruction과 같은 방향이었다. 그래도 작은 deterministic software 연구다. Seed 3개는 sampling과 learned KC→MBON weight를 바꾸지만 data, topology, transform을 공유한다.

Greedy accuracy를 calibrated probability나 stochastic deployment success로 읽으면 안 된다. PCA policy의 mean correct-class probability도 0.277로 four-class uniform 0.25에 가깝다. 작은 logit 차이가 correct argmax를 자주 만들었다. Calibration, rejection, physical reward success는 측정하지 않았다.

Source: [native-final.json](../../results/precise-bridge/native-final.json).

### 사후 untrained-policy 대조군

Locked final 평가 뒤, deterministic fresh untrained engine 하나를 training이나 tuning 없이 같은 final set에서 평가했다. Accuracy는 fixed 25.00%, centered 18.75%, PCA 35.94%였고, seed 3개의 trained mean은 41.15%, 58.85%, 92.19%였다. Native PCA policy 결과가 초기 random policy만이 아니라 KC→MBON reward training에 의존했다는 좁은 주장을 지지한다.

이 대조군은 final 결과를 본 뒤 추가했고, 독립 seed 3개가 아니라 deterministic initialization 하나만 사용했으며 locked final report는 바꾸지 않았다. PCA의 초기 accuracy가 chance보다 높다는 점도 trained와 untrained 값을 모두 보고해야 하는 이유다.

Source: [untrained-final-posthoc.json](../../results/precise-bridge/untrained-final-posthoc.json).

## Port-placement sensitivity

Seed 811–815의 고정 permutation 5개가 signed feature value 256개를 PN input port에 재할당했다. Padding port 63개와 모든 PN→KC edge는 그대로 유지했다. Permutation마다 train32와 validation12로 자체 semantic ridge를 fit했다. 다섯 개 모두 final 전에 freeze했고 final 결과로 고르지 않았다.

| Transform | Unpermuted final | Permutation 평균 | Permutation 범위 | 해석 |
|---|---:|---:|---:|---|
| Fixed | 59.38% | 64.38% | 59.38–73.44% | 원래 fixed placement만 유리하지 않았음 |
| Centered | 64.06% | 64.38% | 56.25–73.44% | Placement variance가 unpermuted score와 비슷함 |
| PCA | **93.75%** | 79.38% | 65.63–90.63% | PCA는 여전히 유용했지만 이점 일부가 coordinate placement에 의존 |

PCA permutation은 25 percentage point 범위로 변했다. 다섯 개 중 unpermuted 93.75%에 도달한 것은 없었지만 permutation 5개로 전체 null distribution을 추정하거나 특별한 anatomy-alignment mechanism을 증명할 수 없다. Transform은 engineered language axis를 non-exchangeable PN root에 할당하므로 port placement는 실제 modeling choice이자 confound이며 복원된 PN semantics가 아니다.

Permutation 값은 같은 dependent authored text 64개를 반복 변환한 것이며 새 dataset 5개가 아니다.

Source: [permutation-final.json](../../results/precise-bridge/permutation-final.json).

## Wiring manifest

[wiring.json](../../results/precise-bridge/wiring.json)은 native input index 319개 전체에 대해 PN root ID, PN→KC edge count, engineered coordinate/sign assignment를 export한다. Input 순서는 TSV sorting이 아니라 native checkpoint에서 가져온다. Index 0–127은 positive coordinate, 128–255는 대응 negative coordinate, 256–318은 padding이다.

Root ID와 edge count는 connectome-derived다. MRL/PCA coordinate와 sign 할당은 engineered다. 특히 PCA axis 0이 특정 PN root에 놓였다고 해서 그 PN이 해당 semantic dimension이나 odor identity를 운반한다는 증거는 아니다.

## Protocol lock과 reproducibility replay

Schema-1 [protocol-lock.json](../../results/precise-bridge/protocol-lock.json)은 다음 layer와 hash를 기록한다.

1. train32, validation12, authored final64 data
2. frozen EmbeddingGemma encoder setting과 holdout embedding
3. fixed, centered, PCA port transform
4. fixed PN→KC topology, semantic decoder, native checkpoint, permutation artifact
5. validation selection rule과 single final-evaluation contract
6. software 및 environment version

[evaluation-lock.json](../../results/precise-bridge/evaluation-lock.json)은 protocol hash를 final JSON 전체, holdout embedding, provenance와 결합한다. Locked artifact가 바뀌면 verify가 거부한다.

여기서 replay는 exact reproducibility이며 experience replay나 생물학적 replay mechanism이 아니다.

- Semantic replay는 transform과 decoder 세 개를 다시 fit하여 tensor artifact 6개와 저장된 final prediction을 exact하게 재현했다.
- Native replay는 CPU training을 독립 반복하여 mapping tensor 3개, checkpoint tensor 9개, 모든 metric과 prediction을 exact하게 재현했다.
- Extended replay는 원 train32/validation12 input에서 permutation decoder 15개를 다시 fit하여 저장 tensor array 전체를 exact하게 재현했고, timing을 제외한 permutation final result와 native final run/aggregate도 다시 확인했다. 원 locked artifact는 바뀌지 않았다.
- Fresh front-door 전체 실행은 isolated directory에서 EmbeddingGemma holdout을 다시 encode하고 development, freeze, evaluation, verification, replay를 반복했다. 새로 생성한 embedding을 포함해 Safetensors artifact 34개의 모든 array가 exact하게 일치했고 semantic, native, permutation final prediction도 exact했다.

Sources: [semantic-replay.json](../../results/precise-bridge/semantic-replay.json), [native-replay.json](../../results/precise-bridge/native-replay.json), [extended-replay.json](../../results/precise-bridge/extended-replay.json), [full-reproduction.json](../../results/precise-bridge/full-reproduction.json).

## Layer 및 command API

각 command는 서로 다른 experiment layer에 대응한다.

Fresh end-to-end run은 front-door wrapper로 실행한다.

```bash
cd /Users/a405394/fly/nobi
.venv/bin/python examples/bio_bridge/run_neural_bridge.py all \
  --run-dir results/my-neural-run \
  --model /path/to/google_embeddinggemma-300m
```

Artifact path의 portability를 위해 `--run-dir`은 저장소 내부의 fresh empty directory여야 한다. `--model` 경로는 바꿀 수 있지만 wrapper가 training-feature provenance의 weight revision과 비교하여 hash mismatch를 거부한다. Staged lifecycle이 필요하면 `develop`, `freeze`, `encode`, `evaluate`, `verify`, `replay` phase도 개별 실행할 수 있다.

각 layer를 점검할 때는 아래 fine-grained command도 계속 사용할 수 있다.

```bash
cd /Users/a405394/fly/nobi

# Import/encoder: 이미 작성하고 lock한 holdout encoding.
.venv/bin/python examples/bio_bridge/encode_precise_holdout.py

# Ports, fixed circuit, semantic decoder, validation-only ridge selection.
.venv/bin/python examples/bio_bridge/precise_bridge.py

# 독립 native reward-readout development.
.venv/bin/python examples/bio_bridge/precise_native.py --development

# 고정 port-placement sensitivity development.
.venv/bin/python examples/bio_bridge/precise_permutation.py --development

# Final을 읽기 전에 전체 schema freeze 및 verify.
.venv/bin/python examples/bio_bridge/precise_pipeline.py freeze
.venv/bin/python examples/bio_bridge/precise_pipeline.py verify

# 단 한 번의 final evaluation: semantic, native, permutation 전체 실행.
.venv/bin/python examples/bio_bridge/precise_pipeline.py evaluate

# Final 뒤 reproducibility replay.
.venv/bin/python examples/bio_bridge/precise_pipeline.py replay
.venv/bin/python examples/bio_bridge/precise_native.py --replay
.venv/bin/python examples/bio_bridge/replay_precise_artifacts.py

# Descriptive summary와 figure.
.venv/bin/python examples/bio_bridge/summarize_precise.py

# 사후 diagnostic이며 locked final report를 바꾸지 않음.
.venv/bin/python examples/bio_bridge/check_precise_untrained.py
.venv/bin/python examples/bio_bridge/export_precise_wiring.py
```

현재 result directory는 이미 freeze 및 evaluate되었다. 같은 곳에서 `evaluate`를 다시 실행하면 의도적으로 거부한다. 현재 artifact에는 `verify`와 replay를 사용한다. 전체 sequence를 깨끗하게 재현하려면 fresh isolated result directory 또는 checkout과 동일한 local model revision을 사용해야 한다.

최종 validation suite는 bio-bridge test 67개와 기타 repository test 30개, 총 97개가 통과했다.

## 해석과 한계

결과는 이 과제의 retained engineered interface로 PCA128-rank31을 지지한다. Validation, final semantic retrieval, 독립 reward-trained native policy 평가에서 모두 가장 높았다. 그러나 결론 범위는 좁다.

- Transform fitting에는 inherited development text 32개만 사용했다.
- Final text 64개는 이 campaign에서 작성한 bilingual scenario pair 32개다.
- Port permutation은 coordinate-to-PN placement가 결과에 상당한 영향을 줌을 보였다.
- Semantic cosine, exact identity, category retrieval, reward-policy accuracy는 서로 다른 target이다.
- Policy seed 3개와 permutation 5개는 반복 software condition이며 독립 biological sample이 아니다.
- Fly neural recording, sensory stimulus, behavioral trial, causal biological perturbation이 없다.

별도의 neural-link/BCI stability 확장은 forthcoming이다. 그 미래 결과를 이 문서에 포함하거나 암시하지 않는다.

측정된 주장은 train-only rank-31 PCA interface가 한 locked authored-family set에서 선언한 MRL mapping 두 개보다 task-relevant semantic structure를 잘 보존하고 reward learning을 더 잘 지원했다는 것이다. 생물학적 sensory alignment 결과가 아니다.
