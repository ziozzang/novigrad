#!/usr/bin/env python3
"""Download pinned FlyWire v783 derivatives, verify, stream-convert using Arrow.

Python is an offline data tool only. Rust simulation needs no Python runtime.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

MODEL = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/"
ANNOTATIONS = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/"
SOURCES = [
    ("Connectivity_783.parquet", MODEL + "Connectivity_783.parquet", "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
    ("Completeness_783.csv", MODEL + "Completeness_783.csv", "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    ("neurons_783.tsv", ANNOTATIONS + "supplemental_files/Supplemental_file1_neuron_annotations.tsv", "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be"),
]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(directory, name, url, expected):
    path = directory / name
    if path.exists():
        if sha256(path) != expected:
            raise ValueError(f"checksum mismatch: {path}; remove it explicitly before retry")
        return
    partial = path.with_suffix(path.suffix + ".part")
    print(f"downloading {name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if sha256(partial) != expected:
        raise ValueError(f"checksum mismatch: {partial}")
    partial.replace(path)


def convert(directory):
    with (directory / "Completeness_783.csv").open() as stream:
        reader = csv.reader(stream)
        next(reader)
        ids = pa.array([int(row[0]) for row in reader], type=pa.int64())
    if len(set(ids.to_pylist())) != len(ids):
        raise ValueError("duplicate neuron IDs")
    source = pq.ParquetFile(directory / "Connectivity_783.parquet")
    columns = ["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory", "Excitatory x Connectivity"]
    out_path = directory / "edges_783.tsv"
    temporary = out_path.with_suffix(".tsv.part")
    rows = synapses = 0
    schema = pa.schema([(name, pa.int64()) for name in ["pre_id", "post_id", "syn_count", "sign"]])
    with pacsv.CSVWriter(str(temporary), schema, write_options=pacsv.WriteOptions(include_header=False, delimiter="\t")) as writer:
        for batch in source.iter_batches(batch_size=262144, columns=columns):
            values = {name: batch.column(i) for i, name in enumerate(columns)}
            if any(column.null_count for column in values.values()):
                raise ValueError("null in connectivity data")
            for side in ["Presynaptic", "Postsynaptic"]:
                mapped = pc.take(ids, values[side + "_Index"], boundscheck=True)
                if not pc.all(pc.equal(mapped, values[side + "_ID"])).as_py():
                    raise ValueError(f"{side} index/root_id mapping mismatch")
            count, sign = values["Connectivity"], values["Excitatory"]
            if not pc.all(pc.greater(count, 0)).as_py():
                raise ValueError("nonpositive synapse count")
            if not pc.all(pc.is_in(sign, value_set=pa.array([-1, 0, 1]))).as_py():
                raise ValueError("unexpected polarity")
            if not pc.all(pc.equal(pc.multiply(count, sign), values["Excitatory x Connectivity"])).as_py():
                raise ValueError("signed connectivity mismatch")
            writer.write_batch(pa.record_batch([values["Presynaptic_ID"], values["Postsynaptic_ID"], count, sign], schema=schema))
            rows += len(batch)
            synapses += pc.sum(count).as_py()
    temporary.replace(out_path)
    manifest = {
        "dataset": "FlyWire FAFB v783; Shiu model derivative",
        "source_files": [{"file": name, "url": url, "sha256": digest} for name, url, digest in SOURCES],
        "neurons_in_completeness": len(ids), "connection_rows": rows,
        "summed_synapse_counts": synapses, "output_sha256": sha256(out_path),
        "conversion": "all rows retained; original root IDs/counts/polarity; source indices validated against Completeness CSV",
        "runtime_transform": "Rust normalizes each source's signed outgoing counts by their absolute sum; nodes with no edges omitted",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        fetch(args.data_dir, *source)
    convert(args.data_dir)


if __name__ == "__main__":
    main()
