# 신경 기록 안정성: 제한된 공학적 유사 실험

Neuralink의 2019년 논문은 유연한 전극 스레드, 다채널 동시 기록, 로봇 삽입, 온보드 증폭·디지털화를 설명한다. 이 저장소는 해당 하드웨어를 구현하거나 평가하지 않는다([Musk & Neuralink, 2019](https://pubmed.ncbi.nlm.nih.gov/31642810/)).

Degenhart 등은 안정적인 전극을 이용해 기록 시점 사이의 저차원 신경 매니폴드를 정렬하고, 원숭이 BCI에서 기준선 변화·유닛 탈락·튜닝 변화를 시험했다([Degenhart et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7822646/)). NoMAD는 비선형 잠재 동역학을 이용해 이후 기록을 기준 세션에 정렬했다([Karpowicz et al., 2025](https://www.nature.com/articles/s41467-025-59652-y)).

`neural_link_stability.py`는 훨씬 단순하다. 고정된 공학적 KC rate 벡터에 합성 gain/offset 및 채널 탈락을 적용하고, 기존 학습 split의 소수 paired anchor로 affine ridge 보정을 학습한다. 시간 동역학, factor analysis, 안정 전극 식별, 이식 기록, spike 처리, 비지도 세션 정렬을 구현하지 않는다. 따라서 Degenhart/NoMAD 재현이나 Neuralink 통합이 아니다.

NPZ/Safetensors importer는 고유한 정수 trial ID인 `sample_ids`, 초 단위의 유한한 `time`, `neural_features`, feature 순서에 대응하는 고유 정수 `channel_ids`, 선택적 정수 class ID인 `targets`를 받는다. target은 보정에 사용하지 않는다. Safetensors에는 `novi.recording` version-2 metadata도 필요하다. 부동소수점 ID, 중복 ID/채널, 비유한 값, shape 불일치, 알 수 없는 필드, 채널 identity 또는 순서 불일치를 거부한다. 보정 artifact는 anchor basis, dual coefficient, 실제 ridge 값, 채널 ID를 저장하며 재로드 후 추론이 정확히 같아야 한다.

외부 workflow는 `sample_ids`로 paired row를 정렬하고, 기본값으로 timestamp를 정확히 비교한다(명시적 tolerance도 가능). 채널 ID와 순서가 같아야 하며 row 순서나 feature 수만으로 대응 관계를 추정하지 않는다.

```bash
python neural_link_stability.py fit --observed train-observed.safetensors --reference train-reference.safetensors --calibration affine.safetensors
python neural_link_stability.py apply --calibration affine.safetensors --input later-session.safetensors --output corrected.safetensors
python neural_link_stability.py verify --calibration affine.safetensors --input later-session.safetensors --output corrected.safetensors --manifest corrected.safetensors.manifest.json
```

각 단계는 hash와 차원을 JSON manifest에 기록한다. `import-demo.json`은 합성 KC gain/offset 기록에서 동일한 경로를 실행한다. 순서가 섞인 train anchor 32개를 ID로 맞추고 old validation을 보정한 뒤, 재로드 출력과 메모리 직접 추론을 비교한다. timestamp는 이 소프트웨어 시험을 위한 합성 초 단위 값이며, channel ID는 고정된 graph checkpoint에 저장된 실제 `hidden_ids`를 checkpoint feature 순서로 사용한다.

drift, ridge, anchor 수, sham pairing은 validation 전에 고정했다. old train만 anchor로 사용하고 old validation만 평가한다. 결과는 작성된 embedding 입력에 대한 기술적 소프트웨어 시험이며 신경과학·임상 증거가 아니다.

anchor 32개에서 전체 [stability report](../../results/neural-link-bridge/stability.json)의 결과는 다음과 같다.

| drift | cosine: 무보정 | 보정 | shuffled sham | 정확도: 무보정 | 보정 | shuffled sham |
|---|---:|---:|---:|---:|---:|---:|
| clean | 1.000 | 0.815 | 0.674 | 0.833 | 0.833 | 0.333 |
| gain + offset | 0.693 | 0.810 | 0.654 | 0.833 | 0.833 | 0.333 |
| dropout 25% | 0.840 | 0.811 | 0.664 | 0.833 | 0.833 | 0.333 |
| dropout 50% | 0.681 | 0.802 | 0.650 | 0.833 | 0.750 | 0.417 |

보정은 gain/offset과 dropout 50%의 cosine을 높였지만 dropout 25%에서는 조금 낮췄다. clean에서도 ridge map이 identity가 아니므로 cosine이 낮아졌다. 정확도 향상은 없었고 dropout 50%에서는 오히려 낮아졌다. shuffled sham이 더 나쁜 점은 이 simulation 안에서 올바른 pairing의 가치를 지지하지만, 정확도의 음성 결과 때문에 일반적인 안정화 효과를 주장할 수 없다.

저장소 root에서 고정된 study와 전체 import demo를 재현할 수 있다.

```bash
.venv/bin/python examples/bio_bridge/neural_link_stability.py study --output results/neural-link-bridge/stability.json
.venv/bin/python examples/bio_bridge/demo_neural_recordings.py --out results/neural-link-bridge/import-demo.json
```

생성된 [import demo report](../../results/neural-link-bridge/import-demo.json)는 hash, manifest, 거부된 channel-order mismatch, serialized output과 직접 inference의 정확한 비교를 포함한다.
