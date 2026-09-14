#![cfg(feature = "api")]

use novi::plastic::{Engine, PlasticConfig};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::Duration,
};
use tempfile::TempDir;

struct Fixture {
    root: TempDir,
    checkpoint: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input.tsv");
        let plastic = root.path().join("plastic.tsv");
        fs::write(
            &input,
            "9007199254740993\t10\t1\t1\n18446744073709551614\t20\t1\t1\n",
        )
        .unwrap();
        fs::write(
            &plastic,
            "10\t100\t1\t1\n20\t100\t1\t1\n10\t200\t1\t1\n20\t200\t1\t1\n",
        )
        .unwrap();
        let engine = Engine::load(&input, &plastic, PlasticConfig::default()).unwrap();
        let checkpoint = root.path().join("original.safetensors");
        engine.save_checkpoint(&checkpoint).unwrap();
        Self { root, checkpoint }
    }
    fn saved(&self) -> PathBuf {
        self.root.path().join("saved/demo.safetensors")
    }
}

struct Server {
    child: Child,
    address: String,
}
impl Server {
    fn start(fixture: &Fixture, training: bool) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_novi_api"));
        command
            .arg("--model")
            .arg(format!("demo={}", fixture.checkpoint.display()))
            .arg("--model")
            .arg(format!("second={}", fixture.checkpoint.display()))
            .args(["--bind", "127.0.0.1:0"])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if training {
            command
                .arg("--allow-training")
                .arg("--checkpoint-dir")
                .arg(fixture.root.path().join("saved"));
        }
        let mut child = command.spawn().unwrap();
        let mut startup = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut startup)
            .unwrap();
        let address = startup
            .trim()
            .strip_prefix("listening=http://")
            .expect("server did not start")
            .to_owned();
        Self { child, address }
    }
    fn raw(&self, method: &str, route: &str, body: &str) -> (u16, String) {
        let mut socket = TcpStream::connect(&self.address).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(socket, "{method} {route} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", self.address, body.len()).unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        let (headers, body) = response.split_once("\r\n\r\n").expect("HTTP response");
        let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, body.to_owned())
    }
    fn request(&self, method: &str, route: &str, body: Value) -> (u16, Value) {
        let (status, response) = self.raw(method, route, &body.to_string());
        (
            status,
            serde_json::from_str(&response).expect("JSON response"),
        )
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn weights(path: &Path) -> Vec<f32> {
    Engine::load_checkpoint(path).unwrap().weights().collect()
}

#[test]
fn metadata_inference_and_readonly_default() {
    let fixture = Fixture::new();
    let server = Server::start(&fixture, false);
    let (status, health) = server.request("GET", "/health", json!({}));
    assert_eq!(status, 200);
    assert_eq!(health["training_enabled"], false);
    let (status, listing) = server.request("GET", "/v1/models", json!({}));
    assert_eq!(status, 200);
    assert_eq!(listing["models"].as_array().unwrap().len(), 2);
    let (status, metadata) = server.request("GET", "/v1/models/demo", json!({}));
    assert_eq!(status, 200);
    assert_eq!(
        metadata["input_ids"],
        json!(["9007199254740993", "18446744073709551614"])
    );
    assert_eq!(metadata["actions"], 2);
    assert_eq!(metadata["output_ids"], json!(["100", "200"]));
    assert_eq!(metadata["output_actions"], json!([0, 1]));
    assert_eq!(metadata["output_gains"], json!([1.0, 1.0]));
    let inputs = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]];
    let mut direct = Engine::load_checkpoint(&fixture.checkpoint).unwrap();
    let expected: Vec<_> = inputs
        .iter()
        .map(|row| direct.forward(row).to_vec())
        .collect();
    let (status, inference) = server.request(
        "POST",
        "/v1/models/demo/infer",
        json!({"inputs": inputs, "input_ids": metadata["input_ids"]}),
    );
    assert_eq!(status, 200);
    let actual: Vec<Vec<f32>> = serde_json::from_value(inference["probabilities"].clone()).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(inference["actions"].as_array().unwrap().len(), 2);
    assert_eq!(
        server
            .request(
                "POST",
                "/v1/models/demo/learn",
                json!({"samples": [{"inputs": [1,0], "action": 0, "reward": 1}]})
            )
            .0,
        403
    );
    assert_eq!(
        server
            .request("POST", "/v1/models/demo/checkpoint", json!({}))
            .0,
        403
    );
    assert_eq!(server.request("GET", "/v1/models/absent", json!({})).0, 404);
    assert_eq!(
        server
            .request(
                "POST",
                "/v1/models/absent/infer",
                json!({"inputs": [[1,0]]})
            )
            .0,
        404
    );
}

#[test]
fn whole_batch_validation_and_checkpoint_reload() {
    let fixture = Fixture::new();
    let original = fs::read(&fixture.checkpoint).unwrap();
    let server = Server::start(&fixture, true);
    let valid = json!({"inputs": [1, 0], "action": 0, "reward": 1});
    for invalid in [
        json!({"inputs": [0,1], "action": 2, "reward": 1}),
        json!({"inputs": [-1,0], "action": 1, "reward": 1}),
        json!({"inputs": [1], "action": 0, "reward": 1}),
    ] {
        let (status, _) = server.request(
            "POST",
            "/v1/models/demo/learn",
            json!({"samples": [valid, invalid]}),
        );
        assert_eq!(status, 400);
        assert_eq!(
            server
                .request("POST", "/v1/models/demo/checkpoint", json!({}))
                .0,
            200
        );
        assert_eq!(
            weights(&fixture.saved()),
            weights(&fixture.checkpoint),
            "invalid second sample partially mutated weights"
        );
    }
    let mut expected = Engine::load_checkpoint(&fixture.checkpoint).unwrap();
    for (input, action, reward) in [([1.0, 0.0], 0, 1.0), ([0.0, 1.0], 1, -0.5)] {
        expected.forward(&input);
        expected.accumulate_reward(action, reward).unwrap();
    }
    expected.apply_batch().unwrap();
    assert_eq!(
        server
            .request(
                "POST",
                "/v1/models/demo/learn",
                json!({"samples": [valid, {"inputs": [0,1], "action": 1, "reward": -0.5}]})
            )
            .0,
        200
    );
    let (status, saved) = server.request("POST", "/v1/models/demo/checkpoint", json!({}));
    assert_eq!(status, 200);
    assert_eq!(saved["checkpoint"], "demo.safetensors");
    assert_eq!(
        weights(&fixture.saved()),
        expected.weights().collect::<Vec<_>>()
    );
    assert_ne!(weights(&fixture.saved()), weights(&fixture.checkpoint));
    assert_eq!(fs::read(&fixture.checkpoint).unwrap(), original);
    let (status, independent) = server.request(
        "POST",
        "/v1/models/second/infer",
        json!({"inputs": [[1,0]]}),
    );
    assert_eq!(status, 200);
    let mut baseline = Engine::load_checkpoint(&fixture.checkpoint).unwrap();
    let p: Vec<Vec<f32>> = serde_json::from_value(independent["probabilities"].clone()).unwrap();
    assert_eq!(p[0], baseline.forward(&[1.0, 0.0]));
}

#[test]
fn malformed_requests_limits_and_no_http_paths() {
    let fixture = Fixture::new();
    let server = Server::start(&fixture, true);
    for body in [
        json!({"inputs": []}),
        json!({"inputs": [[1]]}),
        json!({"inputs": [[-1,0]]}),
        json!({"inputs": [[1,0]], "input_ids": ["18446744073709551614", "9007199254740993"]}),
        json!({"inputs": vec![vec![0,0];257]}),
    ] {
        assert_eq!(server.request("POST", "/v1/models/demo/infer", body).0, 400);
    }
    assert_eq!(server.request("POST", "/v1/models/demo/infer", json!({"inputs": [[1,0]], "input_ids": [9007199254740993_u64, 18446744073709551614_u64]})).0, 422);
    assert_eq!(
        server
            .request(
                "POST",
                "/v1/models/demo/checkpoint",
                json!({"path": "../outside.safetensors"})
            )
            .0,
        422
    );
    assert!(!fixture.root.path().join("outside.safetensors").exists());
    assert_eq!(
        server.raw("POST", "/v1/models/demo/infer", "{broken").0,
        400
    );
    assert_eq!(
        server
            .raw(
                "POST",
                "/v1/models/demo/infer",
                &" ".repeat(1024 * 1024 + 1)
            )
            .0,
        413
    );
    assert_eq!(server.request("GET", "/health", json!({})).0, 200);
}

#[test]
fn cli_rejects_unsafe_names_and_nonlocal_bind() {
    let binary = env!("CARGO_BIN_EXE_novi_api");
    let help = Command::new(binary).arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8(help.stdout)
        .unwrap()
        .contains("--model NAME=CHECKPOINT"));
    for args in [
        vec![],
        vec!["--model", "../escape=x"],
        vec!["--model", "ok=x", "--bind", "0.0.0.0:8080"],
        vec!["--model", "ok=x", "--allow-training"],
        vec!["--model", "ok=x", "--checkpoint-dir", "x"],
        vec!["--model", "ok=x", "--model", "ok=y"],
    ] {
        assert!(!Command::new(binary)
            .args(args)
            .output()
            .unwrap()
            .status
            .success());
    }
}
