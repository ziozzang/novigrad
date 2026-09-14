# 소프트 실시간 런타임

`novi::runtime::Runtime`은 Safetensors 체크포인트를 로드한 뒤 고정 입력 포트의 rate 벡터를 반복 처리합니다. `tick(input, None)`은 시냅스를 바꾸지 않고 확률과 선택 행동을 계산합니다. `tick(input, Some((action, reward)))`은 그 입력에 대한 보상을 즉시 반영합니다. 입력 순서는 체크포인트의 `input_ids()`와 같아야 합니다.

```sh
cargo build --release --bin novi_rt
./target/release/novi_rt model.safetensors input.tsv --ticks 1000 --period-us 10000
```

`input.tsv`는 줄마다 `port_id rate`를 적거나, `novi_engine infer`처럼 `port_id:rate` 필드를 한 줄에 탭 또는 공백으로 구분할 수 있습니다. 생략한 포트는 0입니다. 프로그램은 타이밍 측정 전에 입력을 파싱하고, 루프에서는 계산 시간 초과와 예약 시각 초과를 따로 셉니다. 지난 주기는 몰아서 실행하지 않고 건너뜁니다. 이 기능은 **소프트 실시간** 측정용입니다. macOS 스케줄러에서 실행되므로 하드 실시간 마감이나 RTOS 기능을 보장하지 않습니다. 자세한 API와 지표는 [영문 문서](runtime.md)를 참고하세요.
