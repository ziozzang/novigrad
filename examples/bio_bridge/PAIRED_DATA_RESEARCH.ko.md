# 향후 신경–임베딩 학습에 사용할 공개 초파리 대응 자료

확인일: 2026-09-17. 첫 단계로 작은 **실측 신경 신호 보정 과제**와 별도의 **FlyVis 시간축 시뮬레이션 과제**를 권한다. 실제 뇌–행동 정렬은 개체별 파일과 시간 동기화를 확인한 뒤 Aimon의 영역별 시계열 또는 하행 뉴런 기록을 우선 검토한다. 이번 조사에서 생물학적 정렬 모델을 구현하거나 학습하지 않았다.

정적 connectome은 배선 자료이며 동기화된 자극·활동·행동 자료가 아니다. FlyVis와 FlyGym의 대응 신호는 시뮬레이션에서 생성된다. 실제 calcium/voltage 기록에는 측정 과정과 시간 지연을 고려해야 한다. 아래 자료에 초파리의 자연어 생각이 담겨 있는 것은 아니다. 텍스트 임베딩을 쓴다면 실험 자극이나 관측 행동의 설명을 표현한다는 범위를 명시해야 한다.

## 후보 비교

| 자료 | 대응 관계 | 형식·접근·라이선스 | 적합한 용도와 한계 |
|---|---|---|---|
| FlyVis | 시각 입력 ↔ 시뮬레이션 뉴런 반응 | 공식 PyTorch 구현, 코드 MIT | 시간축 모델 실험에 유리하다. 생성된 반응은 동물에서 측정한 활동이 아니다. 코드와 pretrained weight를 따로 고정한다. |
| FlyGym / NeuroMechFly | 시뮬레이션 감각·신체·제어 궤적; 논문 자료에 실측 보행 운동학도 포함 | 코드 Apache-2.0; 논문 데이터 DOI `10.7910/DVN/3MCEYR` | 체화된 폐루프 실험용이다. 실제 전뇌 대응 기록은 아니다. 현행 2.x와 논문 당시 1.x API가 호환되지 않는다. 데이터 API가 여기서는 403이라 파일 크기·데이터 라이선스는 미확인이다. |
| Aimon CRCNS fly-1 | 실제 전뇌 형광과 행동 또는 자극 조건 | 주로 NIfTI, CC BY 4.0, DOI `10.6080/K01J97ZN`; 계정 필요; 전체 약 965 GB | 생물학적 관련성은 높지만 전체 다운로드부터 시작하기에는 크다. 모든 기록에 세 모달리티가 함께 있다고 가정하면 안 된다. |
| Aimon 2023 영역별 자료 | 실제 영역 활동과 자발적·강제 보행 분석 | 시계열 DOI `10.5061/dryad.3bk3j9kpb`; 작은 파생 배열·표는 `Aimon2022` GitHub, 저장소 MIT | 더 작은 실제 활동–행동 경로로 유망하다. 그림용 요약값을 원시 trial로 취급하지 않는다. Dryad 전체 파일 목록은 이번에 확보하지 못했다. |
| 하행 뉴런 네트워크 | 실제 신경·행동 시계열과 광유전학 조건 | 코드 MIT; 개체별 처리 데이터 Dataverse, population imaging DOI `10.7910/DVN/INYAYV` | 자극→활동→행동 분석 후보. 원시 영상은 요청 방식이며 공개 처리 배열이 현실적이다. 데이터 라이선스·압축 파일 크기는 추가 확인 대상이다. |
| 상행 뉴런 screen | 동기화 형광·행동·구형 트레드밀 속도 | 코드 Apache-2.0; Dataverse `AN` 모음 | 실제 활동–행동 결과의 독립 재현 후보. 원시 카메라·two-photon 파일은 공개 모음에 없으며 genotype·개체·획득 ID와 파일 라이선스를 확인해야 한다. |
| Xiao 등 ORN 대응 기록 | 냄새 농도와 동시 spike·calcium 측정 | Dryad DOI `10.5061/dryad.9p8cz8x0j`, 목록상 총 442.09 KB, XLSX·README; Dryad 데이터 CC0 | 가장 작은 실측 파일럿 후보. 행동 대응이나 전뇌 자료는 아니다. |

원 출처: [FlyVis 저자 저장소](https://github.com/TuragaLab/flyvis), [사용자 자극 튜토리얼](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/), [FlyGym 저자 저장소와 API 변경](https://github.com/NeLy-EPFL/flygym/), [NeuroMechFly 논문](https://www.nature.com/articles/s41592-024-02497-y), [CRCNS fly-1](https://crcns.org/data-sets/ia/fly-1/about-fly-1), [Aimon 논문 자료 공개 항목](https://elifesciences.org/articles/85202v1), [Aimon 코드](https://github.com/sophie63/Aimon2022), [하행 뉴런 논문](https://www.nature.com/articles/s41586-024-07523-9), [하행 분석 안내](https://raw.githubusercontent.com/NeLy-EPFL/dn_networks/main/dn_experiments/README.md), [상행 뉴런 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10076225/), [ORN 자료](https://datadryad.org/dataset/doi:10.5061/dryad.9p8cz8x0j), [Dryad CC0 정책](https://blog.datadryad.org/2023/09/22/for-data-creators-top-5-reasons-we-cant-publish-your-dataset-yet/).

## 확인한 형식과 가져오기 경계

**FlyVis:** `BoxEye` 튜토리얼은 `[sequence, frame, height, width]`를 받아 `[sequence, frame, 1, 721]`로 렌더링하고 `sequences.h5` 저장 예제를 제공한다. `random_walk_of_blocks`로 외부 영상 다운로드 없이 자극을 만들 수 있다. 향후 저장할 metadata는 자극 seed, 전체 sequence ID, 프레임 시각, 대비·속도, 모델/ensemble 구성원, 뉴런 type·좌표다. Ensemble 구성원은 동물이 아니다. 모델은 과제 최적화 모델이므로 모든 실측 뉴런 반응에 직접 학습했다고 설명하지 않는다. [공식 튜토리얼](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/).

**Aimon:** CRCNS 기록은 보통 1–2분이고 조건당 약 다섯 마리이며 modality와 frame rate가 다르다. [자료 설명](https://crcns.org/data-sets/ia/fly-1/about-fly-1). 후속 저장소의 notebook에는 `FlyID`, `expID`, `GAL4`, `FR`, `WalkRegressor`, `TurnRegressor`, 조건별 시간 범위가 존재한다. MATLAB의 영역별 `TS`와 보행·회전 regressor를 불러온다. 다만 저자 로컬 드라이브 경로와 옛 pandas API가 있어 곧바로 이식 가능한 schema라고 보장할 수 없다. [DB 생성 코드](https://github.com/sophie63/Aimon2022/blob/944c8367d5eed1a4ef477c702e31f31997762d34/MakeDBOfExperiements.ipynb), [영역 코드](https://github.com/sophie63/Aimon2022/blob/944c8367d5eed1a4ef477c702e31f31997762d34/MakeDBFunctionalRegions.ipynb).

작은 두 배열만 메모리로 내려받아 `allow_pickle=False`로 확인했다. `CompTrialsAllFreq.npy`는 193,080 bytes, float64 `(271, 89)`이고 `FuncRPANTrials.npy`는 52,328 bytes, float64 `(87, 75)`다. 차원만으로 시간·동물 축의 의미를 확정하지 않았다. GitHub 목록에서 `GoodICsdf.pkl` 240,944 bytes, `Regionsdf.pkl` 8,738,870 bytes도 확인했으나 다운로드하거나 unpickle하지 않았다. 그림 요약 행을 독립 trial 표본으로 만들면 안 된다. 원시 시계열의 공개 범위를 더 확인하는 metadata 우선 접근이 적절하다.

**하행 뉴런:** 저자 `loaddata.py`는 trial별 `processed/beh_df.pkl`과 `v_forw` 같은 속도 항목을 사용한다. Imaging 표와 획득 동기화를 확인해야 하며 행 번호만 같다고 결합하면 안 된다. README는 population imaging, headless behavior, supplementary 자료를 구분한다. 서로 다른 실험을 가상의 동시 전신 기록으로 합치지 않는다. [고정 버전 loader](https://github.com/NeLy-EPFL/dn_networks/blob/b9e66203ab825187e5d9bd651a63e46c2d5f0ab0/dn_experiments/loaddata.py).

**ORN 소형 파일럿:** 비서식 Figure 2 XLSX는 35.32 KB로 표시된다. 자료 카드에 fly·ROI, 냄새 희석 농도, peak 형광, spike rate, calcium kinetics가 설명되어 있다. Figure 4에는 연령과 평균 시계열이 있다. 서로 다른 실험의 비슷한 fly 열 이름을 억지로 대응시키지 말고 실제 paired measurement만 사용한다. Sheet별 ID가 로컬일 수 있고 평균 trace로는 동물 단위 시계열 검증을 할 수 없다. [자료 schema](https://datadryad.org/dataset/doi:10.5061/dryad.9p8cz8x0j).

## 후속 benchmark 제안

**A. 언어 정렬 전에 실제 신호 보정.** 작은 ORN workbook으로 calcium·농도·허용된 획득 공변량에서 측정 spike peak를 예측한다. 선형 보정, 작은 비선형 모델, embedding 조건 모델을 비교한다. 같은 동물의 ROI·농도·반복은 모두 같은 split에 두고 동물별 외부 검증, 학습 동물 내부의 하이퍼파라미터 선택을 수행한다. Sheet 간 fly ID의 지속성을 먼저 점검한다. 동물별 MAE·상관과 동물 단위 불확실성을 보고한다. 이미 계산된 회귀계수·p값·평균·시험 데이터 최대값으로 정규화된 값은 입력에서 제외한다. 이는 측정 인터페이스 실험이지 행동·생각 해독이 아니다. 개체 수가 적으므로 큰 인코더 전체 학습보다 nested leave-one-fly-out과 작은 모델이 적합하다.

**B. 실제 신경↔행동 시간축 연결.** 동기화와 개체 ID를 확보할 수 있다면 Aimon 영역 시계열을 우선하고, 아니면 DN population imaging에 실제 대응하는 행동 표를 선택한다. 활동 구간→행동 상태·속도와 행동·자극→활동을 양방향으로 비교한다. 외부 시험은 개체 분리, 내부 검증은 획득 세션 분리로 한다. 시간 경계 주변에서는 최대 모델 문맥 길이와 학습한 calcium 반응 구간을 합친 만큼 window를 제거한다. 정규화·영역 정렬·embedding·calcium kernel은 학습 자료로만 적합한다. 인접 프레임이나 같은 개체의 trial을 무작위로 나누지 않는다. 자발적/강제/광유전학 조건은 별도 분포 이동 시험으로 둔다. 신경만·행동 이력만·자극만·짝 교란 기준선을 비교한다. 행동 설명문은 허용된 관측값으로만 만들고 예측 대상 활동값에서 만들지 않는다. 개체 간 뉴런 일대일 대응이 없다면 영역/type pooling과 결측 mask를 명시한다.

**C. FlyVis/FlyGym 인과 공학 시뮬레이션.** 제한된 움직이는 경계·블록 영상에서 대비·속도를 바꿔 FlyVis 반응을 만들고, 필요하면 별도 FlyGym 감각·제어·자세·시각 궤적을 생성한다. 전체 장면·궤적 seed와 자극군으로 split하고 겹치는 window를 분리하지 않는다. 새 속도·대비 조합, 모델·arena 파라미터를 각각 보류한다. 이는 시뮬레이션 분포 이동이지 새 동물 일반화가 아니다. 학습 resampler/prefix와 같은 예산 linear·pooled 모델로 자극 embedding↔활동 복원을 비교한다. 자극 caption은 물리 파라미터에서 만들되 neural-only decoder에는 그 파라미터를 주지 않는다. Controller 행동이 simulator에 실제 영향을 줄 때만 폐루프라 부르고, 과제 성공·개입 효과를 embedding 검색과 분리해 평가한다. 버전·시간 간격·렌더 설정·checkpoint·초기화 상태를 고정한다. MPS/CUDA/CPU 실행성과 속도는 이번에 시험하지 않았다.

## 확인한 버전과 다음 단계

공식 GitHub API에서 확인한 source commit과 코드 라이선스:

| 저장소 | Commit | 코드 라이선스 |
|---|---|---|
| `TuragaLab/flyvis` | `92b3845cc426dd309a1a0e1b3890156c42e14021` | MIT |
| `NeLy-EPFL/flygym` | `38c8ec61034cd59bc5ba0de20688d4a3c0000d60` | Apache-2.0 |
| `sophie63/Aimon2022` | `944c8367d5eed1a4ef477c702e31f31997762d34` | MIT |
| `NeLy-EPFL/dn_networks` | `b9e66203ab825187e5d9bd651a63e46c2d5f0ab0` | MIT |
| `NeLy-EPFL/Ascending_neuron_screen_analysis_pipeline` | `1dd7cbdc347f5484e22ff4ab1a5f4f132b397c76` | Apache-2.0 |

코드 라이선스가 별도 데이터·모델 weight에 자동 적용되는 것은 아니다. 가져올 때 DOI/version, 파일 라이선스·checksum, 개체/세션/trial ID, 시간 단위, 동기화 변환, 결측, 전처리를 적합한 split, 자극 출처를 기록한다. 신뢰할 수 있는 legacy pickle/MAT는 분리된 변환 단계에서 명시적 배열과 metadata로 바꾸고 임의 pickle을 무심코 실행하지 않는다.

다음의 작은 작업은 ORN workbook과 Aimon 영역 파일 목록 점검이다. CRCNS 전체를 내려받는 것이 아니다. 이번 조사에서는 패키지 설치·대용량 다운로드·예측·학습을 하지 않았다. 전송한 수치 자료는 위의 작은 Aimon 배열 두 개뿐이며 메모리에서 확인했고 학습 파일로 저장하지 않았다.
