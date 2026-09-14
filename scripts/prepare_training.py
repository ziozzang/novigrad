#!/usr/bin/env python3
"""Extract annotated FlyWire v783 olfactory feedforward edges for training."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def annotations(path):
    classes = {"ALPN": set(), "Kenyon_Cell": set(), "MBON": set()}
    subclasses = {name: Counter() for name in classes}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            name = row["cell_class"]
            if name in classes:
                root = int(row["root_id"])
                if root in classes[name]:
                    raise ValueError(f"duplicate {name} root_id {root}")
                classes[name].add(root)
                subclasses[name][row["cell_sub_class"] or "(blank)"] += 1
    if any(not values for values in classes.values()):
        raise ValueError("one or more cell classes missing")
    return classes, subclasses


def extract(directory):
    ann_path = directory / "neurons_783.tsv"
    conn_path = directory / "Connectivity_783.parquet"
    classes, subclasses = annotations(ann_path)
    targets = {
        "pn_kc": ("ALPN", "Kenyon_Cell"),
        "kc_mbon": ("Kenyon_Cell", "MBON"),
    }
    counters = {key: {"edges": 0, "synapses": 0, "signs": Counter(), "sign_synapses": Counter(),
                      "source_roots": set(), "target_roots": set()} for key in targets}
    paths = {key: directory / f"{key}.tsv" for key in targets}
    temp_paths = {key: path.with_suffix(".tsv.part") for key, path in paths.items()}
    source = pq.ParquetFile(conn_path)
    columns = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"]
    value_sets = {name: pa.array(sorted(roots), type=pa.int64()) for name, roots in classes.items()}
    files = {key: path.open("w", newline="") for key, path in temp_paths.items()}
    try:
        writers = {key: csv.writer(stream, delimiter="\t", lineterminator="\n") for key, stream in files.items()}
        for batch in source.iter_batches(batch_size=262144, columns=columns):
            pre, post, count, sign = (batch.column(i) for i in range(4))
            for key, (pre_class, post_class) in targets.items():
                mask = pc.and_(pc.is_in(pre, value_set=value_sets[pre_class]),
                               pc.is_in(post, value_set=value_sets[post_class]))
                if not pc.any(mask).as_py():
                    continue
                selected = [pc.filter(col, mask).to_pylist() for col in (pre, post, count, sign)]
                stats = counters[key]
                for row in zip(*selected):
                    a, b, n, polarity = row
                    if n <= 0 or polarity not in (-1, 0, 1):
                        raise ValueError(f"invalid source edge: {row}")
                    writers[key].writerow(row)
                    stats["edges"] += 1
                    stats["synapses"] += n
                    stats["signs"][polarity] += 1
                    stats["sign_synapses"][polarity] += n
                    stats["source_roots"].add(a)
                    stats["target_roots"].add(b)
    finally:
        for stream in files.values():
            stream.close()
    for key in targets:
        temp_paths[key].replace(paths[key])
    circuits = {}
    for key, (pre_class, post_class) in targets.items():
        stats = counters[key]
        circuits[key] = {
            "file": paths[key].name, "sha256": sha256(paths[key]),
            "source_cell_class": pre_class, "target_cell_class": post_class,
            "connection_rows": stats["edges"], "synapses": stats["synapses"],
            "polarity_connection_rows": {str(i): stats["signs"][i] for i in (-1, 0, 1)},
            "polarity_synapses": {str(i): stats["sign_synapses"][i] for i in (-1, 0, 1)},
            "connected_source_roots": len(stats["source_roots"]),
            "connected_target_roots": len(stats["target_roots"]),
            "excluded_annotated_source_roots_without_matching_edge": len(classes[pre_class] - stats["source_roots"]),
            "excluded_annotated_target_roots_without_matching_edge": len(classes[post_class] - stats["target_roots"]),
        }
    manifest = {
        "dataset": "FlyWire FAFB v783; Shiu model connectivity derivative",
        "sources": [
            {"file": ann_path.name, "sha256": sha256(ann_path)},
            {"file": conn_path.name, "sha256": sha256(conn_path)},
        ],
        "selection": "Exact annotation cell_class ALPN→Kenyon_Cell and Kenyon_Cell→MBON; all matching direct connectivity rows retained with dataset polarity.",
        "format": "Headerless TSV: pre root_id, post root_id, synapse count, Excitatory sign (-1/0/+1).",
        "annotated_cell_class_roots": {name: len(roots) for name, roots in classes.items()},
        "annotation_subclasses": {name: dict(sorted(counts.items())) for name, counts in subclasses.items()},
        "excluded_edges": "All source rows outside the two exact class-pair selections; no edge or synapse count threshold.",
        "circuits": circuits,
    }
    manifest_path = directory / "training_circuit.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(circuits, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    extract(args.data_dir)


if __name__ == "__main__":
    main()
