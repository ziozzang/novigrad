# FunctionGemma → typed API → novi

A local, executable research prototype: original BF16 FunctionGemma 270M generates calls, a strict dispatcher validates them, and an HTTP API invokes the existing Gemma/novi reward policy. A separate LoRA adapter improves domain routing. The original checkpoint is never overwritten or additionally quantized.

## What actually ran

| Evaluation | Original | Adapted |
|---|---:|---:|
| Explicit development calls | 8/8 | 8/8 |
| Fresh English/Korean calls, fixed before training | 6/16 | **13/16** |
| Four explicit goals through actual HTTP and novi | 4/4 | 4/4 |

Accuracy means exact function **and** argument equality, not merely parseable output. The fresh set was not used for training or checkpoint selection; there is one predetermined 80-step run, one training seed and a tiny hand-authored dataset. It is a useful integration result, not a broad reliability claim. The same author/code module defines training and fresh sets; there is no independently authored blind test or repository-history proof of predeclaration. Some fresh prompts remain lexically similar to the domain training templates despite no exact overlap.

The first baseline reported 0/8 because our parser rejected a legitimate terminal `<start_function_response>` token. The raw model calls were already correct. The fixed parser accepts that boundary but rejects any generated function-response body, extra calls, or trailing text. Do not attribute this improvement to the neural model. The official format identifies the response-start token as an inference stopping boundary. [Google format documentation](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)

An earlier 16-case challenge scored 7/16 with original descriptions, 9/16 with expanded bilingual descriptions, and **6/16 with few-shot examples**. The few-shot variant copied a demonstrated 30° angle into unrelated requests and was rejected as the default. Those repeated challenge evaluations are exploratory; they are distinct from the later fresh set. Negative comparator code and raw outputs are retained for reproducibility.

LoRA trains only 368,640 parameters (rank 8, alpha 16, Q/V projections) on 76 separately authored examples, 80 steps × batch 2, learning rate 2e−4. Only response-token cross-entropy is optimized; prompt/padding tokens are masked. The base stays BF16 and frozen; adapter parameters may use FP32. The adapter changes inference behavior, so adapted inference is not numerically identical to the original model. Training took 12.52 seconds in this run on M2 Ultra; no energy measurement was taken.

Fresh-case median generation latency was 0.299 seconds original and 0.276 seconds adapted, excluding model load. This is short-call generation including prefill/synchronization, not a long-text decode benchmark. The separate DeepSeek 30-TPS objective is not validated by these results.

## API contract

| Tool | Arguments | Effect |
|---|---|---|
| `set_goal` | `goal`: water / food / warmth / rest | Sets a high-level need; rejected while an action is pending. |
| `observe_heading` | finite `degrees` in −180…180 | Records a **user-reported, unverified** value. It is not a trusted sensor; the current novi action backend ignores heading, and the navigation benchmark is separate. |
| `choose_action` | none | Runs the native novi policy once; returns action ID, meaning and probabilities. Requires a goal and no pending action. |
| `get_status` | none | Reads state without changing it. |

FunctionGemma chooses only among these declared tools. Arbitrary code evaluation is never used. Unknown functions, wrong fields, ranges, malformed calls and multiple calls are rejected. A semantically wrong but schema-valid call can still pass: the model remains an imperfect interpreter.

`POST /v1/call` accepts `{"name":"set_goal","arguments":{"goal":"water"}}`. `GET /v1/status` returns current state. Host-only `POST /v1/environment/outcome` accepts `action_id` and reward in [−1,1] and requires a separate Bearer capability from `NOVI_ENVIRONMENT_TOKEN`. This capability is not put in model prompts or tool schemas. Successfully completed outcomes reject duplicate delivery within the in-memory session. A learner that mutates state and then raises can leave partial changes; retries are not transactionally safe. Session state remains pending if the learner raises; the backend's learning atomicity remains its own responsibility.

The server binds only 127.0.0.1, handles one request at a time and owns one session. There is no durable crash recovery, multi-user authentication for ordinary calls, continuous batching, cancellation or restart replay. Other local processes can issue ordinary tool calls. Pending actions require a trusted outcome before a new choice. Do not describe this example as a production service.

The live demo checks semantic consistency between the requested need and returned action meaning, not an external environmental outcome. Its four episodes learn sequentially and are not independent evaluations of the initial checkpoint. FunctionGemma does not invent the objective or grade its own success. Canonical need embeddings are cached at startup; once the enum is known, this second encoder stage adds no new language information. It reuses the existing learned novi policy to demonstrate the interface, not an advantage over directly decoding the enum.

## Reproduce

From the repository root, first prepare the [Gemma bridge](../gemma_bridge/README.md) and its `novi-101` bundle, then:

```sh
.venv/bin/python -m pip install -r requirements-functiongemma.txt
.venv/bin/python examples/function_bridge/evaluate.py --variant base --out results/function-bridge/baseline-rerun.json
.venv/bin/python examples/function_bridge/finetune.py
.venv/bin/python examples/function_bridge/evaluate.py --variant base --split fresh --adapter results/function-bridge/lora --out results/function-bridge/fresh-lora-rerun.json
.venv/bin/python examples/function_bridge/live.py --adapter results/function-bridge/lora
.venv/bin/python examples/function_bridge/server.py --embedding-model /path/to/google_embeddinggemma-300m
.venv/bin/python -m unittest discover -s examples/function_bridge -p 'test_*.py' -v
.venv/bin/python examples/function_bridge/navigation.py
```

Evaluation/live scripts accept `--model` or `--function-model` for a different original local FunctionGemma directory. `finetune.py` uses the recorded `/Users/a405394/models/google_functiongemma-270m-it` path; change its `model_path` if relocating. No automatic model download or license acceptance occurs. Model and adapter weights retain their applicable model terms; our source license does not relicense them. Large weights are not committed. The generated adapter uses Safetensors and its metadata/configuration are retained.

## Robustness limits

The adapted model plus validator rejected an out-of-range angle and a compound request in the four-case robustness probe (2/4 correct rejection). For requests to self-award reward or delete a checkpoint, the model instead emitted `get_status`. Those calls cannot perform the requested mutation, but they are still semantic errors rather than successful refusal. More negative/ambiguous examples, explicit abstention, fresh evaluation and broader seeds are required before autonomous real-world use.

## Research cases beyond tool calls

[English research catalog](CASES_RESEARCH.md) / [한국어](CASES_RESEARCH.ko.md) connects goal switching, cue dropout, cue conflict, hunger/aversion and reversal to primary studies. Six software navigation conditions ran across three seeds, with an analytic true-heading control and ablations. The estimator is a persistent circular state variable, not reconstructed FC2/EPG/PFL neurons; the LLM and novi are not in that simulation loop. Actual anatomy and candidate IDs remain in the [earlier audit](../gemma_bridge/RESEARCH.md).

Raw evaluation outputs, training settings, fresh/training cases, live HTTP episodes, model hashes and navigation trajectories are under [`results/function-bridge`](../../results/function-bridge). The run stops at its four-iteration research cap; it does not certify general tool-use reliability or biological fidelity.
