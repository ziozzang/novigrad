use novi::plastic::Engine;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "novi_cli_{}_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        name
    ))
}

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/generic")
        .join(name)
}

fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_novi_engine"))
        .args(args)
        .output()
        .expect("run novi_engine")
}

#[test]
fn train_eval_infer_and_frozen_checkpoint() {
    let checkpoint = temp("model.safetensors");
    let input = example("input_edges.tsv");
    let plastic = example("plastic_edges.tsv");
    let train = example("train.tsv");
    let eval = example("eval.tsv");
    let infer = example("infer.tsv");
    let result = run(&[
        "train",
        path(&input),
        path(&plastic),
        path(&train),
        path(&checkpoint),
        "2",
        "100",
        "7",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("training_after"));
    let before = fs::read(&checkpoint).unwrap();
    let result = run(&["eval", path(&checkpoint), path(&eval)]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("evaluation samples=2 greedy=1.0000"));
    let result = run(&["infer", path(&checkpoint), path(&infer)]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let lines = String::from_utf8(result.stdout).unwrap();
    assert_eq!(lines.lines().count(), 2);
    let result = run(&["bench", path(&checkpoint), "10"]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("iterations=10"));
    assert_eq!(fs::read(&checkpoint).unwrap(), before);
    fs::remove_file(checkpoint).unwrap();
}

#[test]
fn malformed_and_unknown_inputs_are_rejected() {
    let checkpoint = temp("model.safetensors");
    let sample = temp("bad.tsv");
    let input = example("input_edges.tsv");
    let plastic = example("plastic_edges.tsv");
    let train = example("train.tsv");
    assert!(run(&[
        "train",
        path(&input),
        path(&plastic),
        path(&train),
        path(&checkpoint)
    ])
    .status
    .success());
    fs::write(&sample, "0\t999:1.0\n").unwrap();
    let unknown = run(&["eval", path(&checkpoint), path(&sample)]);
    assert!(!unknown.status.success());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("root ID absent"));
    fs::write(&sample, "0\t1:NaN\n").unwrap();
    let malformed = run(&["eval", path(&checkpoint), path(&sample)]);
    assert!(!malformed.status.success());
    assert!(String::from_utf8_lossy(&malformed.stderr).contains("finite and nonnegative"));
    fs::remove_file(checkpoint).unwrap();
    fs::remove_file(sample).unwrap();
}

#[test]
fn init_builds_five_action_opponent_checkpoint_without_overwriting() {
    let input = temp("init-input.tsv");
    let plastic = temp("init-plastic.tsv");
    let checkpoint = temp("init-model.safetensors");
    fs::write(&input, "1 10 1 1\n2 20 1 1\n").unwrap();
    let edges = (100..110)
        .map(|post| format!("10 {post} 1 1\n20 {post} 1 1\n"))
        .collect::<String>();
    fs::write(&plastic, edges).unwrap();
    let initialized = run(&["init", path(&input), path(&plastic), path(&checkpoint)]);
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stderr)
    );
    let engine = Engine::load_checkpoint(&checkpoint).unwrap();
    assert_eq!(engine.config().actions, 5);
    assert_eq!(engine.config().learning_rate, 0.01);
    assert_eq!(engine.config().logit_gain, 6.0);
    assert_eq!(engine.config().active_fraction, 0.2);
    assert!(!engine.config().homeostasis);
    assert_eq!(engine.output_ids().len(), 10);
    assert_eq!(engine.output_actions(), &[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
    assert_eq!(
        engine.output_gains(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    );
    let saved = fs::read(&checkpoint).unwrap();
    let repeated = run(&["init", path(&input), path(&plastic), path(&checkpoint)]);
    assert!(!repeated.status.success());
    assert!(String::from_utf8_lossy(&repeated.stderr).contains("already exists"));
    assert_eq!(fs::read(&checkpoint).unwrap(), saved);
    fs::remove_file(input).unwrap();
    fs::remove_file(plastic).unwrap();
    fs::remove_file(checkpoint).unwrap();
}
