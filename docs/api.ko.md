# 로컬 HTTP API

`novi_api`는 이름을 지정한 범용 rate 엔진 체크포인트를 HTTP로 제공한다. 입력은 모델의 입력 포트 순서에 맞춘 음이 아닌 실수 벡터이며, 출력은 행동 확률과 0부터 시작하는 행동 인덱스다. 이미지·오디오·로봇·게임·에이전트 어댑터가 이 벡터를 만들 수 있다. API 자체는 이미지 전처리나 행동의 의미 해석을 하지 않는다.

인증과 TLS가 없는 localhost 프로토타입이며 루프백 주소에만 바인딩할 수 있다. 포트에 접근할 수 있는 로컬 프로세스는 활성화된 기능을 사용할 수 있다. 기본 모드는 읽기 전용이고, 학습과 체크포인트 저장은 시작 옵션으로 명시적으로 활성화한다.

## 빌드와 실행

`api`는 선택적 Cargo 기능이다. 일반 `cargo build --release`는 HTTP 서버나 그 전용 의존성을 빌드하지 않는다.

```sh
cargo build --release --features api --bin novi_api
cargo run --release --bin novi_engine -- train \
  examples/generic/input_edges.tsv examples/generic/plastic_edges.tsv \
  examples/generic/train.tsv /tmp/novi-demo.safetensors 2 100 7
./target/release/novi_api --model demo=/tmp/novi-demo.safetensors
```

기본 주소는 `127.0.0.1:8080`이다. `--bind 127.0.0.1:9000` 또는 `--bind '[::1]:9000'`으로 바꿀 수 있다. `--model 이름=체크포인트`를 반복하면 여러 모델을 독립적으로 불러온다. 이름은 ASCII 영문자·숫자·밑줄·하이픈으로 구성된 1–64자이며 중복은 거부한다. 모델은 시작할 때 불러오며, HTTP로 임의 경로나 모델 설정을 지정할 수 없다. 전체 옵션은 `--help`로 확인한다.

## 경로와 입력

| 메서드와 경로 | 요청 | 응답 |
| --- | --- | --- |
| `GET /health` | 없음 | 상태, 모델 수, 학습 여부, 요청 제한 |
| `GET /v1/models` | 없음 | 모델 이름, 입력·출력 포트 수, 은닉 뉴런 수, 행동 수 |
| `GET /v1/models/{name}` | 없음 | 모델 정보, `input_ids`, `output_ids`, `output_actions`, `output_gains`, 설정 |
| `POST /v1/models/{name}/infer` | `inputs`, 선택적 `input_ids` | 행별 `probabilities`, 탐욕적 `actions` |
| `POST /v1/models/{name}/learn` | `samples`, 선택적 `input_ids` | 적용한 배치 크기와 갱신 의미; 학습 옵션 필요 |
| `POST /v1/models/{name}/checkpoint` | `{}` | 고정된 저장 파일명; 학습 옵션 필요 |

상세 메타데이터의 `output_ids`, `output_actions`, `output_gains`는 같은 인덱스의 출력 뉴런에 대한 외부 행동 그룹과 배율이다. 이 정보만으로 해부학적 운동 기능을 주장하지 않는다. 입력·출력 ID는 **10진수 문자열**이다. 서버로 다시 보낼 때도 JSON 숫자로 바꾸지 않는다. 일부 뉴런 ID는 JavaScript에서 정확히 표현할 수 있는 정수 범위를 넘는다. `input_ids`를 제공하면 메타데이터의 전체 목록과 순서까지 정확히 일치해야 한다.

```sh
curl http://127.0.0.1:8080/v1/models/demo
curl -H 'Content-Type: application/json' \
  -d '{"input_ids":["1","2"],"inputs":[[1,0],[0,1]]}' \
  http://127.0.0.1:8080/v1/models/demo/infer
```

위 예제 모델의 입력 포트는 두 개다. 다른 모델은 메타데이터에서 포트 수와 ID를 읽는다. 모든 행에는 정확히 `input_ports`개의 유한하고 음이 아닌 값이 필요하다. 서버는 모든 행을 검증한 후 추론한다. 가장 큰 확률이 같으면 작은 인덱스를 선택한다. 추론은 대기 중인 일시적 피드백 상태를 정리하며 가중치를 변경하지 않는다.

표준 라이브러리만 사용하는 Python 예제:

```python
import json
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8080/v1/models/demo"

def request(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as response:
        return json.load(response)

model = request(base)
rates = [0.0] * model["input_ports"]
rates[0] = 1.0
result = request(base + "/infer", {
    "input_ids": model["input_ids"], "inputs": [rates],
})
print(result["actions"][0], result["probabilities"][0])
```

## 학습과 저장

다음 두 옵션을 함께 사용해야 변경이 활성화된다.

```sh
./target/release/novi_api --model demo=/tmp/novi-demo.safetensors \
  --allow-training --checkpoint-dir /tmp/novi-api-checkpoints
curl -H 'Content-Type: application/json' \
  -d '{"samples":[{"inputs":[1,0],"action":0,"reward":1},{"inputs":[0,1],"action":1,"reward":-0.5}]}' \
  http://127.0.0.1:8080/v1/models/demo/learn
curl -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8080/v1/models/demo/checkpoint
```

각 샘플은 입력 벡터, `0..actions-1` 범위의 행동, 유한한 부호 있는 스칼라 보상을 제공한다. 서버는 **전체 배치를 변경 전에 검증**한다. 모든 샘플을 배치 시작 시점의 가중치로 다시 계산하고, 보상 기울기를 누적한 뒤 평균 기울기를 한 번 적용한다. 뒤쪽 샘플에 오류가 있어도 앞쪽 샘플만 일부 학습되는 일이 없다. 누적 기울기가 유한하지 않으면 가중치 적용 없이 버린다. 보상이 0인 샘플도 배치 크기에 포함되며 기울기 기여는 0이다.

`/infer`는 가장 확률이 큰 행동을 반환한다. 에이전트는 반환된 확률에서 직접 행동을 표본 추출해 외부 환경에서 실행하고, 관측한 보상과 해당 입력·행동을 `/learn`에 보낼 수도 있다. 학습 요청은 실행 시점의 가중치로 입력을 다시 계산한다. 이전 추론의 활성 상태를 재사용하거나 요청을 식별하지 않으며, 사이에 일어난 정책 갱신을 보정하지 않는다. 따라서 이는 표본 피드백 또는 지정 목표 학습이고, 트랜잭션 방식의 지연 보상 프로토콜은 아니다. 여러 클라이언트가 같은 모델을 학습하면 상호작용 순서를 조율해야 한다.

학습 결과는 `/checkpoint`가 성공하기 전까지 메모리에만 존재한다. `demo` 모델은 시작 옵션의 디렉터리 안에 `demo.safetensors`로 원자적으로 저장·교체한다. HTTP 요청은 경로나 파일명을 선택할 수 없으며, 응답에는 서버의 전체 경로가 나오지 않는다. 출력 디렉터리와 파생 파일명이 원본과 같도록 설정하지 않는 한 불러온 원본은 바뀌지 않는다. 저장 파일은 기존 `nobi.plastic` 스키마를 유지하며 `novi_engine`, `novi_rt`, 새 API 프로세스에서 읽을 수 있다.

## 제한과 동시성

JSON 본문은 1 MiB, 추론·학습 배치는 256행까지다. 알 수 없는 필드는 거부한다. 검증 실패는 HTTP 400, JSON 자료형 오류는 422, 본문 크기 초과는 413, 읽기 전용 변경 시도는 403, 없는 모델은 404, 작업 슬롯 부족은 503을 반환한다. 애플리케이션 오류는 `{"error":"..."}` 형식이다. 없는 경로나 지원하지 않는 메서드 등의 프레임워크 오류 본문은 다를 수 있다.

모델별 mutex를 사용하므로 같은 모델의 요청은 순차적으로 처리하고 서로 다른 모델은 별도 작업 스레드에서 실행할 수 있다. 추론·학습·파일 저장은 [Tokio blocking 작업](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html)으로 처리하며, 실행 중이거나 모델 잠금을 기다리는 작업은 최대 8개다. 상태 확인은 모델 잠금을 기다리지 않는다. HTTP 경로와 JSON 본문 제한은 [Axum](https://docs.rs/axum/latest/axum/struct.Router.html)을 사용한다.

클라이언트 연결이 끊겨도 이미 시작한 학습이나 저장은 취소되지 않는다. 요청에 멱등성 키가 없으므로 응답을 받지 못했다고 학습을 무조건 재시도하면 중복 적용될 수 있다. 하드 실시간 보장, 원격 접근 제어, 영속 작업 큐, 다중 프로세스 일관성은 제공하지 않는다.

`cargo test --features api --test api_cli`는 합성 두 포트 모델로 실제 TCP 요청을 검증한다. `2^53`보다 큰 입력 ID도 포함하며 이미지 홀드아웃은 평가하지 않는다.

[English API documentation](api.md)
