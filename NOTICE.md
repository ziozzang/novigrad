# Data and third-party notices

The MIT license at the repository root covers Novigrad source code. It does not
relicense upstream datasets, annotations, fonts, or their rights. Retain this
notice with released model bundles, which preserve FlyWire neuron IDs and
connectivity-derived initial structure/counts.

- **FlyWire FAFB v783**: Dorkenwald et al., *Neuronal wiring diagram of an adult brain*,
  Nature (2024), https://doi.org/10.1038/s41586-024-07558-y . Original archive:
  https://zenodo.org/records/10676866 . Consult the original archive's data terms.
- **Model-ready connectivity derivative**: Philip Shiu and collaborators,
  https://github.com/philshiu/Drosophila_brain_model . That repository includes an
  MIT license; Novigrad downloads its pinned v783 data, not its execution code.
  Original data rights and citation requirements continue to apply.
- **Neuron annotations**: Schlegel et al., *Whole-brain annotation and
  multi-connectome cell typing of Drosophila*, Nature (2024),
  https://doi.org/10.1038/s41586-024-07686-5 and
  https://github.com/flyconnectome/flywire_annotations . See upstream data terms.
- **Fashion-MNIST**: Han Xiao, Kashif Rasul, Roland Vollgraf / Zalando Research,
  https://github.com/zalandoresearch/fashion-mnist . The upstream repository
  supplies its dataset under MIT: https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE .
  Raw training/test archives are downloaded separately and excluded from Git.
- **Synthetic character images**: locally rendered for this experiment using
  installed macOS fonts. Font software is neither copied nor distributed. The
  dataset manifest records font paths and hashes; sample renderings and training
  results do not grant a license to the original font software.
- **Software dependencies**: Rust safetensors and transitive dependencies are
  governed by their own licenses, listed in their packages and Cargo.lock.
  Python preprocessing uses NumPy, PyArrow, Pillow, and safetensors under their
  respective upstream licenses. No pretrained OCR or vision model is used.

Dataset commits, checksums, extraction rules, and model lineage are recorded in
the preparation scripts, generated manifests, and model bundle manifests. The
project makes no claim that connectome-constrained models reproduce a whole fly
brain or outperform conventional vision systems.
