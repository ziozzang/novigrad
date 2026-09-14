# Data provenance

[`../scripts/prepare_data.py`](../scripts/prepare_data.py) downloads public sources pinned to commits and verifies SHA-256 hashes. Large data files are excluded from Git. [the recorded manifest](../results/connectome-data-manifest.json) records sources and derived artifacts.

- Original connectome: [FlyWire FAFB v783](https://zenodo.org/records/10676866).
- Data used in this run: [Shiu group's model-oriented v783 connectivity data](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960), specifically `Connectivity_783.parquet` and `Completeness_783.csv`. That repository provides an MIT license.
- Neuron type and neurotransmitter annotations: [FlyWire annotations](https://github.com/flyconnectome/flywire_annotations/tree/8587524c1748ce5ef2080822a2fc890fc03bf597). The annotation files are retained; v1 execution uses the Shiu table's `Excitatory` value for polarity.
- Source papers: [Dorkenwald et al., 2024](https://doi.org/10.1038/s41586-024-07558-y) and [Schlegel et al., 2024](https://doi.org/10.1038/s41586-024-07686-5).

The original Zenodo Feather file could not be downloaded in this environment because the connection stalled. The files retained here are **processed connectivity tables distributed by the paper's authors**, not asserted to be identical to the original Feather file. We do not assume that the full public v783 data and this derived table have the same neuron count. Structural imagery, raw imaging, and individual synapse coordinates are not included.

Preparation preserves every connectivity row and validates root IDs and the original-index mapping. Headerless TSV edges have four fields: `pre_id`, `post_id`, `syn_count`, and `sign`. The Rust loader creates neurons only when they occur in an edge. Weights are normalized by the source neuron's sum of absolute weights; they are not physiological current units.

The project's source-code license does not relicense upstream data or annotations. Follow each source's license and citation guidance when redistributing them.
