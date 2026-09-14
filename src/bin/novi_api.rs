//! Optional, loopback-only HTTP interface to named generic rate-engine models.
use axum::{
    extract::{rejection::JsonRejection, DefaultBodyLimit, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use novi::plastic::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{self, Write},
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};
use tokio::sync::Semaphore;

const BODY_LIMIT: usize = 1024 * 1024;
const MAX_BATCH: usize = 256;
const MAX_JOBS: usize = 8;
const USAGE: &str = "Usage: novi_api --model NAME=CHECKPOINT [--model NAME=CHECKPOINT ...]
                [--bind 127.0.0.1:8080] [--allow-training --checkpoint-dir DIRECTORY]

Optional build: cargo build --release --features api --bin novi_api
NAME: 1-64 ASCII letters, digits, underscores or hyphens; unique per process.
Only loopback bind addresses are accepted. No authentication: local prototype only.
Models and checkpoint directory are configured at startup, never through HTTP.
Default is read-only. Training and checkpoint writes require both training flags.
Requests: at most 1 MiB JSON and 256 input rows or learning samples.
Routes: GET /health; GET /v1/models; GET /v1/models/{name};
        POST /v1/models/{name}/infer, /learn, /checkpoint.
See docs/api.md for request formats and sampled-feedback semantics.";

#[derive(Clone)]
struct AppState {
    models: Arc<BTreeMap<String, Mutex<Engine>>>,
    jobs: Arc<Semaphore>,
    checkpoint_dir: Option<PathBuf>,
    model_count: usize,
}

struct ApiError(StatusCode, String);
impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, message.into())
    }
    fn internal(message: impl Into<String>) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, message.into())
    }
    fn readonly() -> Self {
        Self(StatusCode::FORBIDDEN, "server is read-only".into())
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}
type ApiResult = Result<Json<Value>, ApiError>;

fn payload<T>(value: Result<Json<T>, JsonRejection>) -> Result<T, ApiError> {
    value
        .map(|Json(body)| body)
        .map_err(|e| ApiError(e.status(), e.body_text()))
}

async fn blocking<F>(state: AppState, work: F) -> ApiResult
where
    F: FnOnce(&BTreeMap<String, Mutex<Engine>>) -> Result<Value, ApiError> + Send + 'static,
{
    // Bound blocking jobs before spawning; waiting clients do not occupy unlimited threads.
    let permit = state.jobs.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            StatusCode::SERVICE_UNAVAILABLE,
            "server busy; try later".into(),
        )
    })?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work(&state.models).map(Json)
    })
    .await
    .map_err(|_| ApiError::internal("model worker failed"))?
}

fn model<'a>(
    models: &'a BTreeMap<String, Mutex<Engine>>,
    name: &str,
) -> Result<MutexGuard<'a, Engine>, ApiError> {
    models
        .get(name)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "unknown model".into()))?
        .lock()
        .map_err(|_| ApiError::internal("model unavailable"))
}

fn description(name: &str, engine: &Engine, detailed: bool) -> Value {
    let config = engine.config();
    let mut value = json!({"id": name, "input_ports": engine.input_ids().len(),
        "hidden_neurons": engine.hidden_ids().len(), "output_ports": engine.output_ids().len(),
        "actions": config.actions});
    if detailed {
        value["input_ids"] = json!(engine
            .input_ids()
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>());
        value["output_ids"] = json!(engine
            .output_ids()
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>());
        value["output_actions"] = json!(engine.output_actions());
        value["output_gains"] = json!(engine.output_gains());
        value["config"] = json!({"actions": config.actions, "learning_rate": config.learning_rate,
            "logit_gain": config.logit_gain, "active_fraction": config.active_fraction,
            "weight_limit": config.weight_limit, "homeostasis": config.homeostasis});
    }
    value
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(
        json!({"status": "ok", "models": state.model_count, "training_enabled": state.checkpoint_dir.is_some(),
        "max_body_bytes": BODY_LIMIT, "max_batch_rows": MAX_BATCH}),
    )
}

async fn list_models(State(state): State<AppState>) -> ApiResult {
    blocking(state, |models| {
        let entries = models
            .keys()
            .map(|name| Ok(description(name, &*model(models, name)?, false)))
            .collect::<Result<Vec<_>, ApiError>>()?;
        Ok(json!({"models": entries}))
    })
    .await
}

async fn get_model(State(state): State<AppState>, Path(name): Path<String>) -> ApiResult {
    blocking(state, move |models| {
        Ok(description(&name, &*model(models, &name)?, true))
    })
    .await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InferRequest {
    inputs: Vec<Vec<f32>>,
    input_ids: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LearnSample {
    inputs: Vec<f32>,
    action: usize,
    reward: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LearnRequest {
    samples: Vec<LearnSample>,
    input_ids: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointRequest {}

fn validate_ids(engine: &Engine, ids: &Option<Vec<String>>) -> Result<(), ApiError> {
    if let Some(ids) = ids {
        if ids.len() != engine.input_ids().len()
            || ids
                .iter()
                .zip(engine.input_ids())
                .any(|(supplied, expected)| supplied != &expected.to_string())
        {
            return Err(ApiError::bad(
                "input_ids must exactly match the ordered decimal strings from model metadata",
            ));
        }
    }
    Ok(())
}

fn validate_count(count: usize) -> Result<(), ApiError> {
    if count == 0 || count > MAX_BATCH {
        return Err(ApiError::bad(format!(
            "batch must contain 1..={MAX_BATCH} rows"
        )));
    }
    Ok(())
}

fn validate_input(input: &[f32], dimension: usize) -> Result<(), ApiError> {
    if input.len() != dimension || input.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(ApiError::bad(
            "every input row must have input_ports finite, nonnegative rates",
        ));
    }
    Ok(())
}

async fn infer(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: Result<Json<InferRequest>, JsonRejection>,
) -> ApiResult {
    let body = payload(body)?;
    blocking(state, move |models| {
        let mut engine = model(models, &name)?;
        validate_ids(&engine, &body.input_ids)?;
        validate_count(body.inputs.len())?;
        for input in &body.inputs {
            validate_input(input, engine.input_ids().len())?;
        }
        let mut probabilities = Vec::with_capacity(body.inputs.len());
        let mut actions = Vec::with_capacity(body.inputs.len());
        for input in &body.inputs {
            let p = engine.forward(input).to_vec();
            let mut action = 0;
            for index in 1..p.len() {
                if p[index] > p[action] {
                    action = index;
                }
            }
            probabilities.push(p);
            actions.push(action);
            engine.clear_pending();
        }
        Ok(json!({"model": name, "probabilities": probabilities, "actions": actions}))
    })
    .await
}

async fn learn(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: Result<Json<LearnRequest>, JsonRejection>,
) -> ApiResult {
    let body = payload(body)?;
    let enabled = state.checkpoint_dir.is_some();
    blocking(state, move |models| {
        let mut engine = model(models, &name)?;
        if !enabled { return Err(ApiError::readonly()); }
        validate_ids(&engine, &body.input_ids)?;
        validate_count(body.samples.len())?;
        // Validate the entire request before any forward call or gradient accumulation.
        for sample in &body.samples {
            validate_input(&sample.inputs, engine.input_ids().len())?;
            if sample.action >= engine.config().actions || !sample.reward.is_finite() {
                return Err(ApiError::bad("each action must be in range and each reward finite"));
            }
        }
        for sample in &body.samples {
            engine.forward(&sample.inputs);
            if let Err(error) = engine.accumulate_reward(sample.action, sample.reward) {
                engine.discard_batch();
                return Err(ApiError::bad(error));
            }
        }
        if let Err(error) = engine.apply_batch() {
            engine.discard_batch();
            return Err(ApiError::bad(error));
        }
        Ok(json!({"model": name, "samples": body.samples.len(), "updated": true,
            "semantics": "mean reward gradient; all supplied inputs recomputed at current batch-start weights"}))
    }).await
}

async fn checkpoint(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: Result<Json<CheckpointRequest>, JsonRejection>,
) -> ApiResult {
    payload(body)?;
    let directory = state.checkpoint_dir.clone();
    blocking(state, move |models| {
        let engine = model(models, &name)?;
        let directory = directory.ok_or_else(ApiError::readonly)?;
        // No path is supplied by HTTP. Validated startup names derive fixed filenames.
        let filename = format!("{name}.safetensors");
        let save = || -> Result<(), String> {
            let temporary =
                tempfile::NamedTempFile::new_in(&directory).map_err(|e| e.to_string())?;
            engine.save_checkpoint(temporary.path())?;
            std::fs::File::open(temporary.path())
                .and_then(|file| file.sync_all())
                .map_err(|e| e.to_string())?;
            temporary
                .persist(directory.join(&filename))
                .map_err(|e| e.to_string())?;
            Ok(())
        };
        save().map_err(|error| {
            eprintln!("novi_api checkpoint failed: {error}");
            ApiError::internal("checkpoint write failed")
        })?;
        Ok(json!({"model": name, "checkpoint": filename, "saved": true}))
    })
    .await
}

fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{name}", get(get_model))
        .route("/v1/models/{name}/infer", post(infer))
        .route("/v1/models/{name}/learn", post(learn))
        .route("/v1/models/{name}/checkpoint", post(checkpoint))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(state)
}

struct Options {
    models: Vec<(String, PathBuf)>,
    bind: SocketAddr,
    checkpoint_dir: Option<PathBuf>,
}

fn options() -> Result<Option<Options>, String> {
    let mut args = std::env::args().skip(1);
    let mut models = Vec::new();
    let mut bind: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let mut allow_training = false;
    let mut checkpoint_dir = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(None);
            }
            "--model" => {
                let value = args.next().ok_or("--model requires NAME=CHECKPOINT")?;
                let (name, path) = value
                    .split_once('=')
                    .ok_or("--model requires NAME=CHECKPOINT")?;
                if name.is_empty()
                    || name.len() > 64
                    || !name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
                    || path.is_empty()
                {
                    return Err("invalid model name or empty checkpoint path".into());
                }
                if models.iter().any(|(existing, _)| existing == name) {
                    return Err("duplicate model name".into());
                }
                models.push((name.to_owned(), PathBuf::from(path)));
            }
            "--bind" => {
                bind = args
                    .next()
                    .ok_or("--bind requires IP:PORT")?
                    .parse()
                    .map_err(|_| "invalid bind address")?
            }
            "--allow-training" => allow_training = true,
            "--checkpoint-dir" => {
                checkpoint_dir = Some(PathBuf::from(
                    args.next().ok_or("--checkpoint-dir requires DIRECTORY")?,
                ))
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    if models.is_empty() {
        return Err("at least one --model NAME=CHECKPOINT is required".into());
    }
    if !bind.ip().is_loopback() {
        return Err("only loopback bind addresses are supported (no authentication)".into());
    }
    if allow_training != checkpoint_dir.is_some() {
        return Err("--allow-training and --checkpoint-dir must be supplied together".into());
    }
    Ok(Some(Options {
        models,
        bind,
        checkpoint_dir,
    }))
}

async fn run() -> Result<(), String> {
    let Some(options) = options()? else {
        return Ok(());
    };
    let mut models = BTreeMap::new();
    for (name, path) in options.models {
        models.insert(name, Mutex::new(Engine::load_checkpoint(path)?));
    }
    let checkpoint_dir = options
        .checkpoint_dir
        .map(|directory| {
            std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
            directory.canonicalize().map_err(|e| e.to_string())
        })
        .transpose()?;
    let state = AppState {
        model_count: models.len(),
        models: Arc::new(models),
        jobs: Arc::new(Semaphore::new(MAX_JOBS)),
        checkpoint_dir,
    };
    let listener = tokio::net::TcpListener::bind(options.bind)
        .await
        .map_err(|e| e.to_string())?;
    println!(
        "listening=http://{}",
        listener.local_addr().map_err(|e| e.to_string())?
    );
    io::stdout().flush().map_err(|e| e.to_string())?;
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| e.to_string())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("novi_api: {error}\n{USAGE}");
        std::process::exit(1);
    }
}
