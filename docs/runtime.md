# Soft real-time runtime

Novigrad's `novi::runtime::Runtime` wraps a loaded Safetensors `Engine` for repeated fixed-port inference and optional reward feedback. The caller provides one nonnegative, finite rate per checkpoint `input_id`, in that exact order. `tick(input, None)` returns an action and probabilities without changing synapses; `tick(input, Some((action, reward)))` applies online feedback to the same trial. The runtime keeps its output vector and engine scratch space for reuse. Construction requires no pending trial or unapplied gradient batch. A tick validates its input and feedback before changing engine state. `save_checkpoint()` persists the current trained weights in the existing schema-4 format.

```rust
use novi::{plastic::Engine, runtime::Runtime};
use std::time::Duration;

let engine = Engine::load_checkpoint("model.safetensors")?;
let mut runtime = Runtime::new(engine, Duration::from_millis(10))?;
let rates = vec![0.0; runtime.input_ids().len()];
let report = runtime.tick(&rates, None)?;
println!("action={} probability={:?}", report.selected_action, runtime.output());
```

The allocation for `rates` happens in the caller outside the tick. The runtime wrapper allocates its output buffer once at construction. `TickReport::compute_deadline_missed` compares one tick's observed validation, forward, and optional feedback time with the configured period; `Runtime::compute_deadline_misses()` accumulates this count. Neither measurement includes operating-system sleep or external image preprocessing.

The standalone runner accepts a saved model and one sparse input. `INPUT.tsv` may contain `port_id rate` pairs, one per line, or one tab/space-separated row of `port_id:rate` fields like `novi_engine infer`. Unspecified ports are zero. IDs must belong to the checkpoint and cannot be duplicated. All parsing and port mapping happen before the timed loop.

```sh
cargo build --release --bin novi_rt
./target/release/novi_rt model.safetensors input.tsv --ticks 1000 --period-us 10000
```

Its JSON separates **compute deadline misses** (tick compute time exceeds the period), **scheduled finish misses** (wall-clock finish exceeds that slot's deadline), **late starts**, and **skipped periods**. When a slot is overdue, the runner advances to a future slot instead of processing a burst of backlogged ticks. The runner is inference-only; online feedback is available through the Rust API. This is a soft real-time measurement interface. macOS scheduling, page faults, I/O, and power management prevent a hard deadline guarantee, and this is not an RTOS or kernel scheduler.

The graph, checkpoint and rate vector are generic: an upstream image, sensor, or other adapter can supply the same fixed input-port contract. The runner itself does not perform image decoding or segmentation.
