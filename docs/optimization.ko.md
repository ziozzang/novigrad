# 학습 성능을 높이는 메커니즘

[English](optimization.md)

Novigrad는 생물학적 연결 구조와 공학적으로 설계한 인코딩·학습·출력 처리를 구분한다. 새 연결이나 사전 학습 비전 모델을 추가하지 않고, 동일한 회로에서 어떤 변경이 성능을 높이는지 실험한다.

## 반복과 미니배치

`--epochs`는 전체 학습 데이터를 몇 번 반복할지 정한다. `--patience`는 검증 성능이 더 좋아지지 않을 때 조기 종료한다. 저장하는 모델은 마지막 반복이 아니라 검증 성능이 가장 좋은 시점이다.

`--batch-size`는 가중치를 고정한 상태에서 몇 개 샘플의 기울기를 모아 평균한 뒤 한 번 갱신할지 정한다. 전체 데이터 반복과는 다른 개념이다. 배치 크기 1은 기존 온라인 학습 경로를 그대로 사용하며, 마지막에 남은 작은 배치도 적용한다. 누적 버퍼를 재사용하므로 매 샘플마다 메모리를 할당하지 않는다.

학습률은 평균 기울기 한 번의 갱신에 적용된다. 온라인 학습률 0.0001과 배치 16의 학습률 0.0016을 비교하는 것은 한 epoch의 누적 갱신량을 대략 맞추기 위한 실험 설정이다. 엔진이 학습률을 자동으로 바꾸지는 않는다.

```sh
.venv/bin/python scripts/experiment_vision.py baseline
.venv/bin/python scripts/experiment_vision.py no-homeostasis --homeostasis false
.venv/bin/python scripts/experiment_vision.py longer75 --homeostasis false --epochs 75 --patience 10
.venv/bin/python scripts/experiment_vision.py batch16 --homeostasis false --epochs 75 --patience 10 --batch-size 16 --lr .0016
.venv/bin/python scripts/report_vision_experiments.py
```

## 비교하는 변경

- **출력 가중치 총량 제한:** MBON마다 입력 가중치 절댓값의 합을 매번 1 이하로 축소하는 정규화를 켜거나 끈다. 끄더라도 연결 구조·시냅스 부호·개별 가중치 한도는 유지한다.
- **출력 증폭:** softmax에 전달하는 증거의 크기를 조정한다. 기울기 크기도 달라지므로 학습률을 함께 명시해야 한다.
- **학습 반복 수:** 덜 학습된 모델이 더 많은 데이터 반복에서 좋아지는지 검증한다.
- **활성 뉴런 비율:** 경쟁에서 남기는 KC의 비율을 바꾸어 희소성과 정보 보존의 균형을 비교한다.
- **입력 어댑터:** PCA 차원·whitening을 바꾸되, 학습 이미지만 사용해 맞춘다.
- **미니배치:** 여러 샘플의 기울기를 모은 뒤 제약을 한 번 적용하는 효과를 측정한다.

범용 라이브러리에는 `accumulate_reward`, `apply_batch`, `discard_batch` API가 있다. 미적용 배치의 체크포인트 저장, 누적 중 출력 매핑 변경, 온라인 갱신과 누적 배치의 혼용을 거부한다. 기존 Safetensors 모델은 계속 로딩된다.

## 검증 원칙

먼저 고정 seed와 검증 세트로 비교한다. 설정을 선택한 뒤 세 seed와 라벨 섞기 대조군을 학습하고, 기존 시험 결과와 겹치지 않는 새 평가 데이터에서 기존 모델과 비교한다. 정확도 외에도 교차 엔트로피·가중치 분포·처리 시간·모델 크기를 기록한다.

CAPTCHA 전체 정답은 네 자리 모두 맞은 비율이다. 글꼴 분포와 고정된 셀 위치는 여전히 실험 범위의 한계다. 숫자가 높아지더라도 임의의 웹 CAPTCHA 해결이나 실제 뇌의 학습 법칙을 입증한 것은 아니다.
