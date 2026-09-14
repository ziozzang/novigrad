# Python 네이티브 바인딩

Rust 엔진을 Python 프로세스에서 직접 호출합니다. Mac에서 저장소 루트 기준:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python examples/python/basic.py
```

`from novigrad import Engine`으로 불러온 뒤 `Engine.load(path)`, `infer(rates)`, `infer_batch(rows)`, `learn(rows, actions, rewards)`, `save(path)`를 사용합니다. 입력 순서는 `input_ids`이며, 저장은 기본적으로 기존 파일을 덮어쓰지 않습니다. 학습 배치는 전체 입력을 검증한 뒤 평균 업데이트 한 번을 적용합니다. 예제는 새 회로를 만들고 두 패턴을 학습한 다음 Safetensors 재로딩 결과가 같은지 확인합니다.

선택적인 GPU 경로는 `pip install -e '.[mlx]'` 후 `from novigrad.mlx import MlxEngine`으로 사용합니다. `MlxEngine.load(path).infer_batch(rows)`는 Metal에서 배치 추론을 수행합니다. 학습은 CPU `Engine`에서 실행하고 저장 후 GPU 모델을 다시 로드해야 합니다. MLX-C 직접 바인딩과 GPU 학습은 구현하지 않았습니다.

M2 Ultra 측정에서 단일 입력은 CPU가 더 빠르고, 배치 256은 Metal이 약 8.6배 빨랐습니다. Python 리스트 변환과 GPU 동기화/결과 반환을 포함한 측정입니다. 전력 효율이나 모든 모델의 속도를 보장하지 않습니다.

[상세 API와 주의점](README.md) · [성능 실험 보고서](../../results/PYTHON_PERFORMANCE_REPORT.ko.md)
