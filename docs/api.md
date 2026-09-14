# Local HTTP API

`novi_api` serves named generic rate-engine checkpoints over HTTP. An input is a dense vector of nonnegative rates in the model's input-port order; an output is a probability distribution and a zero-based action index. Image, audio, robot, game, and agent adapters can supply these rates. The API does not perform image preprocessing or interpret action meanings.

This is a localhost prototype with no authentication or TLS. It accepts only loopback bind addresses. Any local process able to reach the port can use enabled operations. Inference is read-only by default; learning and checkpoint writes require explicit startup flags.

## Build and start

The `api` Cargo feature is optional. Ordinary `cargo build --release` does not build this server or activate its HTTP dependencies.

```sh
cargo build --release --features api --bin novi_api
cargo run --release --bin novi_engine -- train \
  examples/generic/input_edges.tsv examples/generic/plastic_edges.tsv \
  examples/generic/train.tsv /tmp/novi-demo.safetensors 2 100 7
./target/release/novi_api --model demo=/tmp/novi-demo.safetensors
```

The default address is `127.0.0.1:8080`. Use `--bind 127.0.0.1:9000` or `--bind '[::1]:9000'` to change it. Repeat `--model name=CHECKPOINT` to load multiple independent models. Names must contain 1–64 ASCII letters, digits, underscores, or hyphens. Duplicate names are rejected. Models are loaded at startup; HTTP cannot load arbitrary paths or change configuration. `--help` prints the full CLI.

## Routes

| Method and route | Request | Response |
| --- | --- | --- |
| `GET /health` | None | Status, model count, training flag, limits |
| `GET /v1/models` | None | Model names, input/output port counts, hidden neuron counts, action counts |
| `GET /v1/models/{name}` | None | Model summary, ordered `input_ids`, `output_ids`, `output_actions`, `output_gains`, configuration |
| `POST /v1/models/{name}/infer` | `inputs`, optional `input_ids` | `probabilities` and greedy `actions`, one per row |
| `POST /v1/models/{name}/learn` | `samples`, optional `input_ids` | Applied batch size and update semantics; training flag required |
| `POST /v1/models/{name}/checkpoint` | `{}` | Fixed checkpoint filename; training flag required |

The detailed metadata exposes the checkpoint’s output decoder: `output_ids`, `output_actions`, and `output_gains` are aligned by position. These describe the external action grouping and gain for each output neuron; they do not establish an anatomical motor function. Input and output IDs are **decimal strings**, including when supplied back to the server. Some neuron IDs exceed JavaScript's exact integer range. Do not convert them to JSON numbers. Optional `input_ids` must match the entire ordered metadata list exactly.

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/models/demo
curl -H 'Content-Type: application/json' \
  -d '{"input_ids":["1","2"],"inputs":[[1,0],[0,1]]}' \
  http://127.0.0.1:8080/v1/models/demo/infer
```

The example model has two ports. For another model, obtain its IDs and dimensions from metadata. Each row must contain exactly `input_ports` finite, nonnegative numbers. All rows are validated before inference. Ties in the greedy action use the lowest index. Inference clears transient pending feedback and does not modify weights.

A minimal Python client uses only the standard library:

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

## Learning and persistence

Restart with both flags to enable mutation:

```sh
./target/release/novi_api --model demo=/tmp/novi-demo.safetensors \
  --allow-training --checkpoint-dir /tmp/novi-api-checkpoints
curl -H 'Content-Type: application/json' \
  -d '{"samples":[{"inputs":[1,0],"action":0,"reward":1},{"inputs":[0,1],"action":1,"reward":-0.5}]}' \
  http://127.0.0.1:8080/v1/models/demo/learn
curl -H 'Content-Type: application/json' -d '{}' \
  http://127.0.0.1:8080/v1/models/demo/checkpoint
```

Each sample supplies a dense input vector, an action in `0..actions-1`, and a finite signed scalar reward. The server validates the **entire batch before mutation**, recomputes every sample at the current batch-start weights, accumulates its reward gradient, and applies the mean gradient once. Invalid later rows cannot leave earlier rows partially trained. Nonfinite accumulated gradients are discarded without applying weights. A zero reward consumes a sample and contributes zero gradient to the batch mean.

`/infer` returns greedy actions; an agent can instead sample an action from the returned probabilities, execute it externally, observe reward, and submit that input/action/reward to `/learn`. Learning recomputes the supplied input at the weights current when the learning request runs. It does not reuse a saved activation from the earlier inference request, identify an earlier request, or correct for intervening policy updates. This is sampled-feedback or supplied-target training, not a transactional delayed-reward protocol. Keep each agent's interaction loop coordinated if other clients also train the same model.

Learning changes only memory until `/checkpoint` succeeds. For model `demo`, checkpoint writes atomically replace `demo.safetensors` inside the startup-configured directory. The request cannot choose a filename or directory, and the response does not expose a server filesystem path. The original loaded checkpoint is unaffected unless the configured output directory and derived filename deliberately name that same file. Saved files keep the existing `nobi.plastic` schema and load with `novi_engine`, `novi_rt`, or a new API process.

## Limits and concurrency

JSON bodies are limited to 1 MiB and each infer/learn batch to 256 rows. Unknown fields are rejected. Validation errors use HTTP 400, malformed typed JSON may use 422, oversized bodies use 413, read-only mutation attempts use 403, unknown models use 404, and exhausted worker slots use 503. Application errors carry `{"error":"..."}`. Framework errors such as an unmatched route or unsupported method may have a different body.

Each model has its own mutex, so its requests serialize while different models can execute on separate workers. CPU inference, learning, and checkpoint I/O run via [Tokio's blocking task pool](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html); up to eight jobs may be running or waiting for model locks. Health checks bypass model locks. The HTTP layer uses [Axum's router](https://docs.rs/axum/latest/axum/struct.Router.html) and [JSON body limit](https://docs.rs/axum/latest/axum/extract/struct.DefaultBodyLimit.html).

Client disconnects do not cancel an already-started learning or checkpoint operation. Requests have no idempotency key: if a response is lost, do not blindly retry training because it may already have applied. There are no hard realtime guarantees, remote access controls, persistent job queues, or multi-process consistency mechanisms.

Run the real TCP integration tests with `cargo test --features api --test api_cli`. They use synthetic two-port models, including input IDs above `2^53`, and do not evaluate image holdouts.

[한국어 API 문서](api.ko.md)
