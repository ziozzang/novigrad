use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::{fs, path::PathBuf, process::Command};

#[test]
fn tune_never_writes_predictions_and_full_reports_frozen_baseline() {
    let root = std::env::temp_dir().join(format!("nobi_classifier_{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let input = root.join("input.tsv");
    let plastic = root.join("plastic.tsv");
    let features = root.join("features.safetensors");
    fs::write(&input, "1 10 1 1\n2 20 1 1\n").unwrap();
    fs::write(
        &plastic,
        "10 100 1 1\n10 200 1 -1\n20 100 1 -1\n20 200 1 1\n",
    )
    .unwrap();
    let ids: Vec<u8> = [1u64, 2].into_iter().flat_map(u64::to_le_bytes).collect();
    let x: Vec<u8> = [1f32, 0., 0., 1., 1., 0., 0., 1.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let y: Vec<u8> = [0i64, 1, 0, 1]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect();
    let tensors = [
        ("input_ids", Dtype::U64, vec![2], &ids),
        ("train_inputs", Dtype::F32, vec![4, 2], &x),
        ("validation_inputs", Dtype::F32, vec![4, 2], &x),
        ("test_inputs", Dtype::F32, vec![4, 2], &x),
        ("train_labels", Dtype::I64, vec![4], &y),
        ("validation_labels", Dtype::I64, vec![4], &y),
        ("test_labels", Dtype::I64, vec![4], &y),
    ];
    let views: Vec<_> = tensors
        .iter()
        .map(|(n, t, s, b)| (*n, TensorView::new(*t, s.clone(), b).unwrap()))
        .collect();
    serialize_to_file(views, None, &features).unwrap();
    let invoke = |phase: &str, out: &PathBuf| {
        let result = Command::new(env!("CARGO_BIN_EXE_train_classifier"))
            .args([
                features.as_os_str(),
                input.as_os_str(),
                plastic.as_os_str(),
                out.as_os_str(),
            ])
            .args(["--actions", "2", "--epochs", "2", "--phase", phase])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    };
    let tune = root.join("tune");
    invoke("tune", &tune);
    let tune_json = fs::read_to_string(tune.join("metrics.json")).unwrap();
    assert!(tune_json.contains("\"test_accuracy\": null"));
    assert!(!tune.join("predictions.tsv").exists());
    let full = root.join("full");
    invoke("full", &full);
    let full_json = fs::read_to_string(full.join("metrics.json")).unwrap();
    assert!(!full_json.contains("\"test_accuracy\": null"));
    assert!(!full_json.contains("\"baseline_test_accuracy\": null"));
    assert_eq!(
        fs::read_to_string(full.join("predictions.tsv"))
            .unwrap()
            .lines()
            .count(),
        5
    );
    let inference = root.join("infer.safetensors");
    let views = vec![
        (
            "input_ids",
            TensorView::new(Dtype::U64, vec![2], &ids).unwrap(),
        ),
        (
            "inputs",
            TensorView::new(Dtype::F32, vec![4, 2], &x).unwrap(),
        ),
    ];
    serialize_to_file(views, None, &inference).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_classify_features"))
        .args([full.join("model.safetensors"), inference.clone()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let inferred = String::from_utf8(output.stdout).unwrap();
    let expected = fs::read_to_string(full.join("predictions.tsv")).unwrap();
    for (a, b) in inferred.lines().skip(1).zip(expected.lines().skip(1)) {
        let a: Vec<_> = a.split('\t').collect();
        let b: Vec<_> = b.split('\t').collect();
        assert_eq!(&a[0..2], &[b[0], b[2]]);
        assert_eq!(&a[2..], &b[3..]);
    }
    let reversed: Vec<u8> = [2u64, 1].into_iter().flat_map(u64::to_le_bytes).collect();
    let bad = root.join("bad.safetensors");
    let views = vec![
        (
            "input_ids",
            TensorView::new(Dtype::U64, vec![2], &reversed).unwrap(),
        ),
        (
            "inputs",
            TensorView::new(Dtype::F32, vec![4, 2], &x).unwrap(),
        ),
    ];
    serialize_to_file(views, None, &bad).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_classify_features"))
        .args([full.join("model.safetensors"), bad])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("input_ids mismatch"));
    fs::remove_dir_all(root).unwrap();
}
