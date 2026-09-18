# FlyGym Mac CPU 설치·물리 실행 확인

[English](README.md)

설치와 물리 step 실행 가능성을 확인한 것입니다. 언어모델 브리지·운동 학습·안정 보행·생물학적 정합성 실험이 아닙니다. 기존 프로젝트 환경은 변경하지 않았습니다.

- 별도 환경: Python 3.12.11, FlyGym 2.1.0, MuJoCo 3.9.0, macOS arm64.
- 공식 hybrid controller와 MixedTerrainWorld, 위치 제어 자유도 42개, adhesion 사용, controller seed 0.
- 렌더러·Warp·GPU·언어모델 없이 0.0001초 timestep 1,000회, 모의 시간 0.1초.
- 설치 시간을 제외한 import·구성·warmup 29.92초, step 실행 0.5743초, 약 1,741 step/s와 실시간 대비 0.174배. 짧은 단일 측정이며 최적화된 성능 벤치마크가 아닙니다.
- 흉부 위치 변화 [0.5268, 0.3176, -0.2516] mm. 매 step 위치가 유한함을 검사했지만 직립 유지·안정 보행 성공을 뜻하지 않습니다.
- 패키지 자산 79개의 해시를 기록했고 별도 대형 메시 다운로드는 없었습니다. 공식 제어기가 패키지의 사전 작성 보행 자료를 사용했으며 새 정책을 학습하지 않았습니다.

## 재현

```sh
uv venv --python 3.12.11 /tmp/novigrad-flygym-reproduction
uv pip install --python /tmp/novigrad-flygym-reproduction/bin/python \
  -r results/robot-simulator-smoke/requirements-freeze.txt
/tmp/novigrad-flygym-reproduction/bin/python examples/bio_bridge/flygym_smoke.py \
  --out /tmp/novigrad-flygym-reproduction-result --seconds .1
```

`results.json`에 버전, 설치 배포판 METADATA/RECORD 및 패키지 자산의 해시, 위치 표본, 시간이 있습니다. RECORD 해시는 설치 목록의 지문이며 pip의 `--require-hashes`용 wheel 다운로드 잠금이 아닙니다. `manifest.json`은 실행 결과·의존성 목록·재현 스크립트를 묶습니다.

같은 Mac과 설치 환경에서 새 프로세스로 한 번 더 실행했습니다. step 수, timestep, 제어 자유도, 초기 위치, 저장한 모든 위치 표본과 최종 변위가 정확히 같았습니다. 시간 측정값은 비교에서 제외했습니다. `replay-verification.json`에 기록했으며 다른 하드웨어 재현성이나 장기 안정성을 보장하지 않습니다.

## 원본 자료와 라이선스

[공식 설치](https://neuromechfly.org/installation/), [hybrid controller 예제](https://neuromechfly.org/tutorials/4c_hybrid_controller/), [고정 버전 소스](https://github.com/NeLy-EPFL/flygym/tree/v2.1.0)를 사용했습니다. FlyGym과 MuJoCo 배포판은 Apache-2.0이며 Novigrad의 MIT와 별개입니다. MuJoCo에는 별도 제3자 라이선스 고지도 있습니다. 이 디렉터리는 자체 재현 스크립트와 출처 기록만 포함하며 외부 wheel·메시·정책·보행 자료를 재배포하지 않습니다.
