#!/usr/bin/env python3
"""Audit annotated FlyWire v783 paths relevant to a proposed Gemma bridge.

This is a sparse edge-list audit.  It reports annotation matches and direct
synapse counts; it does not infer function, signal flow, or physiological gain.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_annotations(path):
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            root = int(row["root_id"])
            if root in rows:
                raise ValueError(f"duplicate annotated root_id: {root}")
            rows[root] = row
    return rows


def member_sets(rows):
    # These are exact matches to supplied annotation fields, not morphology
    # reconstructed or inferred by this script.
    return {
        "MBON": {root for root, row in rows.items() if row["cell_class"] == "MBON"},
        "FB_tangential": {
            root for root, row in rows.items()
            if row["cell_class"] == "CX" and row["cell_sub_class"] == "tangential"
        },
        "PFL2": {
            root for root, row in rows.items()
            if row["cell_type"] == "PFL2" or row["hemibrain_type"] == "PFL2"
        },
        "PFL3": {
            root for root, row in rows.items()
            if row["cell_type"] == "PFL3" or row["hemibrain_type"] == "PFL3"
        },
        "EPG": {
            root for root, row in rows.items()
            if row["cell_type"] == "EPG" or row["hemibrain_type"] == "EPG"
        },
        "FC2": {
            root for root, row in rows.items()
            if row["cell_type"] in {"FC2A", "FC2B", "FC2C"}
        },
        "descending": {
            root for root, row in rows.items() if row["super_class"] == "descending"
        },
    }


def type_counts(ids, rows):
    return dict(sorted(Counter(rows[root]["cell_type"] or "(blank)" for root in ids).items()))


def scan_edges(path, groups):
    pfl = groups["PFL2"] | groups["PFL3"]
    transitions = {
        "MBON_to_FB_tangential": (groups["MBON"], groups["FB_tangential"]),
        "MBON_to_PFL": (groups["MBON"], pfl),
        "FB_tangential_to_PFL": (groups["FB_tangential"], pfl),
        "FB_tangential_to_PFL2": (groups["FB_tangential"], groups["PFL2"]),
        "FB_tangential_to_PFL3": (groups["FB_tangential"], groups["PFL3"]),
        "EPG_to_PFL": (groups["EPG"], pfl),
        "EPG_to_PFL2": (groups["EPG"], groups["PFL2"]),
        "EPG_to_PFL3": (groups["EPG"], groups["PFL3"]),
        "EPG_to_FC2": (groups["EPG"], groups["FC2"]),
        "FC2_to_PFL3": (groups["FC2"], groups["PFL3"]),
        "PFL_to_descending": (pfl, groups["descending"]),
        "PFL2_to_descending": (groups["PFL2"], groups["descending"]),
        "PFL3_to_descending": (groups["PFL3"], groups["descending"]),
    }
    found = {name: [] for name in transitions}
    scanned = 0
    with path.open(newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for fields in reader:
            if len(fields) != 4:
                raise ValueError(f"expected four edge fields at row {scanned + 1}")
            pre, post, count, sign = map(int, fields)
            scanned += 1
            if count <= 0 or sign not in (-1, 0, 1):
                raise ValueError(f"invalid edge at row {scanned}: {fields}")
            for name, (sources, targets) in transitions.items():
                if pre in sources and post in targets:
                    found[name].append((pre, post, count, sign))
    return scanned, found


def edge_summary(edges):
    signs = Counter(edge[3] for edge in edges)
    sign_synapses = Counter()
    for _, _, count, sign in edges:
        sign_synapses[sign] += count
    return {
        "connection_rows": len(edges),
        "synapses": sum(edge[2] for edge in edges),
        "connected_sources": len({edge[0] for edge in edges}),
        "connected_targets": len({edge[1] for edge in edges}),
        "polarity_rows": {str(sign): signs[sign] for sign in (-1, 0, 1)},
        "polarity_synapses": {str(sign): sign_synapses[sign] for sign in (-1, 0, 1)},
    }


def chained_paths(found, rows, limit):
    first = found["MBON_to_FB_tangential"]
    second_by_pre = defaultdict(list)
    third_by_pre = defaultdict(list)
    for edge in found["FB_tangential_to_PFL"]:
        second_by_pre[edge[0]].append(edge)
    for edge in found["PFL_to_descending"]:
        third_by_pre[edge[0]].append(edge)

    total = 0
    examples = []
    pfl_counts = Counter()
    for mbon, fb, n1, s1 in first:
        for _, pfl, n2, s2 in second_by_pre.get(fb, ()):
            for _, dn, n3, s3 in third_by_pre.get(pfl, ()):
                total += 1
                pfl_counts[rows[pfl]["cell_type"]] += 1
                if len(examples) < limit:
                    examples.append({
                        "root_ids": [mbon, fb, pfl, dn],
                        "cell_types": [rows[x]["cell_type"] or "(blank)" for x in (mbon, fb, pfl, dn)],
                        "direct_synapse_counts": [n1, n2, n3],
                        "edge_polarities": [s1, s2, s3],
                    })
    return {
        "path_instances": total,
        "path_instances_by_PFL_type": dict(sorted(pfl_counts.items())),
        "examples_truncated_to": limit,
        "examples": examples,
        "interpretation": "Each instance is a join of three observed direct edges. Synapse counts belong to individual edges and are not an end-to-end effective weight.",
    }


def candidate_ports(groups, rows, annotation_path, edge_path):
    proposed_roles = {
        "EPG": "proposed_context_port",
        "FC2": "proposed_intermediate_port",
        "PFL2": "proposed_output_port",
        "PFL3": "proposed_output_port",
    }
    ports = {}
    for group, role in proposed_roles.items():
        entries = []
        for root in sorted(groups[group]):
            row = rows[root]
            entries.append({
                "root_id": root,
                "cell_type": row["cell_type"] or None,
                "hemibrain_type": row["hemibrain_type"] or None,
                "side": row["side"] or None,
            })
        ports[group] = {
            "role": role,
            "status": "proposed_not_calibrated",
            "count": len(entries),
            "neurons": entries,
        }
    return {
        "scope": "Candidate root-ID groups for engineering experiments; not a calibrated biological interface.",
        "sources": {
            "annotations": {"path": str(annotation_path), "sha256": sha256(annotation_path)},
            "edges": {"path": str(edge_path), "sha256": sha256(edge_path)},
        },
        "selection": {
            "EPG": "Exact supplied cell_type or hemibrain_type value EPG.",
            "FC2": "Exact supplied cell_type family members FC2A, FC2B, and FC2C.",
            "PFL2": "Exact supplied cell_type or hemibrain_type value PFL2.",
            "PFL3": "Exact supplied cell_type or hemibrain_type value PFL3.",
        },
        "encoding": "None assigned. Root order and annotation side do not encode direction, heading, phase, action, or signed numeric value.",
        "ports": ports,
        "warning": "Roles are proposed engineering labels only. Structural annotations and synapse counts do not establish function or calibration.",
    }


def audit(annotation_path, edge_path, example_limit):
    rows = load_annotations(annotation_path)
    groups = member_sets(rows)
    scanned, found = scan_edges(edge_path, groups)
    return {
        "scope": "Read-only structural audit of local FlyWire FAFB v783-derived annotations and sparse connectivity.",
        "sources": {
            "annotations": {"path": str(annotation_path), "sha256": sha256(annotation_path)},
            "edges": {"path": str(edge_path), "sha256": sha256(edge_path), "rows_scanned": scanned},
            "edge_format": "Headerless TSV: pre root_id, post root_id, synapse count, Excitatory sign (-1/0/+1).",
        },
        "annotation_matching": {
            "method": "Exact supplied field-value filters: MBON=cell_class MBON; FB tangential=cell_class CX plus cell_sub_class tangential; PFL2/PFL3/EPG=cell_type or hemibrain_type exact value; FC2=cell_type in the explicit family {FC2A, FC2B, FC2C}; descending=super_class descending.",
            "warning": "Exact describes matching against annotation columns only. These are supplied annotation matches, not identities independently inferred from morphology. FB_tangential is a broad CX/tangential candidate filter, including ambiguous or blank types, not independent validation of FB anatomy.",
            "groups": {
                name: {"annotated_roots": len(ids), "cell_types": type_counts(ids, rows)}
                for name, ids in groups.items()
            },
        },
        "direct_connectivity": {name: edge_summary(edges) for name, edges in found.items()},
        "MBON_FB_tangential_PFL_descending_chains": chained_paths(found, rows, example_limit),
        "limitations": [
            "Connectivity is structural only and provides no functional, causal, behavioral, or prefrontal-homology evidence.",
            "The polarity field is inherited from the processed connectivity source; it is not a measured edge-specific postsynaptic response.",
            "The local data contain neuron-level aggregate edges, not synapse coordinates or VNC motor connectivity.",
            "Absence of a selected path can reflect the exact annotation-based selection and does not prove biological absence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/neurons_783.tsv")
    parser.add_argument("--edges", type=Path, default=ROOT / "data/edges_783.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/gemma-bridge/anatomy.json")
    parser.add_argument("--ports-output", type=Path, default=ROOT / "results/gemma-bridge/candidate-ports.json")
    parser.add_argument("--example-limit", type=int, default=100)
    args = parser.parse_args()
    if args.example_limit < 0:
        parser.error("--example-limit must be nonnegative")
    result = audit(args.annotations, args.edges, args.example_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    rows = load_annotations(args.annotations)
    ports = candidate_ports(member_sets(rows), rows, args.annotations, args.edges)
    args.ports_output.parent.mkdir(parents=True, exist_ok=True)
    args.ports_output.write_text(json.dumps(ports, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "ports_output": str(args.ports_output),
        "annotation_counts": {k: v["annotated_roots"] for k, v in result["annotation_matching"]["groups"].items()},
        "direct_connectivity": result["direct_connectivity"],
        "chain_count": result["MBON_FB_tangential_PFL_descending_chains"]["path_instances"],
    }, indent=2))


if __name__ == "__main__":
    main()
