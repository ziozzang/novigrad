# Precise fitting of an embedding-to-fly bridge

## Question and boundary

The engineering question is whether a frozen language embedding can be mapped into the existing 319 input ports so that the fixed PN→KC topology preserves useful semantic structure. This is a representation-fitting problem. It is not a recovery of the fly's sensory code, because no language observation is paired with PN activity from a fly.

The experiment may preserve real connectome edges while learning an artificial coordinate system upstream of them. “Topology preserved” and “biological sensory alignment learned” are different claims:

```text
anatomical claim: the selected PN→KC and KC→MBON edges came from a connectome
software claim: a train-fitted transform assigned language coordinates to those input ports
biological alignment claim: requires matched fly stimuli and neural recordings; unavailable here
```

The allowed conclusion is therefore comparative: one frozen, train-only transform preserves held-out semantic information better than another in this engineered bridge.

## Biological constraints from primary studies

### PN→KC convergence is expansive, sparse, and neither perfectly random nor semantic

[Turner, Bazhenov, and Laurent (2008)](https://doi.org/10.1152/jn.01283.2007) recorded odor responses from Drosophila Kenyon cells in vivo. KC responses were substantially sparser and more selective than their projection-neuron inputs. This supports measuring active fraction and pattern overlap after the embedding-to-port transform. It does not specify how a sentence embedding should occupy PN identities.

[Honegger, Campbell, and Turner (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3180869/) imaged populations of more than 100 KCs and found distinct, sparse odor patterns across diverse stimuli and concentrations. Recent odor history affected sparseness. A software comparison should therefore report representation geometry and sparsity, not classification accuracy alone. Language paraphrases remain unlike odor concentration series.

[Caron et al. (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4148081/) traced glomerular inputs to 200 adult KCs and found that most KCs sampled different, apparently random combinations of glomeruli. This supports a mixing/expansion analogy. It does not imply that arbitrary language dimensions correspond to named PNs or that PCA axes should be called glomeruli.

[Eichler et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5806122/) reconstructed a larval mushroom body. Most KCs combined apparently random inputs, while a subset had stereotyped single-PN input and non-olfactory streams showed structure. The “random projection” description is therefore an approximation with stage and modality limits.

[Zheng et al. (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9413950/) analyzed an adult PN→KC connectome and found above-chance convergence among primarily food-responsive PN types. Observed structure sometimes reduced generic discrimination relative to randomized networks while benefiting particular food-related input. This directly cautions against equating anatomical connectivity with a universally optimal embedding projection.

[Li et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7909955/) used the adult hemibrain mushroom-body connectome to show mixed olfactory input together with subtype biases and segregated visual and thermo-hygro streams. The topology constrains which edges exist, but modality assignments and activity distributions remain biologically meaningful. Reassigning learned PCA coordinates to PN root IDs is an engineered port convention, not learned anatomy.

### Representation alignment is defined by paired observations and independent evaluation

[Haxby et al. (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3201764/) learned orthogonal transformations from paired fMRI response trajectories to form a common representational space, then applied the learned maps to independent experiments. Its useful lesson is procedural: an alignment is learned from matched observations and evaluated outside the fitting data. Here the only paired observations are language embeddings and simulated hidden rates, not language and fly neural activity.

[Raghu et al. (2017)](https://proceedings.neurips.cc/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf) introduced SVCCA to compare learned representations after dimensional reduction and CCA. The work emphasizes effective dimensionality and invariance to affine coordinate changes. It motivates reporting rank, singular values, and held-out representation similarity instead of treating a larger output coordinate count as greater information.

These alignment papers provide computational methodology, not evidence that a fly and a language model share a semantic space.

## Audit of the proposed transform comparison

The proposed candidates are:

1. fixed 128-dimensional MRL mapping;
2. centered 128-dimensional MRL mapping;
3. a train-only PCA transform stored in 128 coordinates, signed-split into 256 ports and padded to 319.

The train/validation/test plan uses 32 old training texts, 12 validation texts for transform selection, and one untouched authored semantic-family set for final evaluation. Native graph topology remains unchanged. Downstream measures are semantic embedding reconstruction and a separately reward-trained native readout. No supervised class probe is part of this round.

### The implemented PCA128 has rank 31

Let the train embedding matrix be `X ∈ R^(32×D)` and let `mu` be its training mean:

```text
X_c = X - 1 mu^T
X_c = U S V^T
z(x) = (x - mu) V_k
```

After centering 32 observations,

```text
rank(X_c) <= min(D, 32 - 1) = 31.
```

A requested 128-coordinate projection can contain at most 31 data-supported principal directions. The implementation therefore stores a `768×128` projection whose first 31 columns contain fitted PCs and whose remaining 97 columns are exactly zero. Signed splitting does not create 128 informative dimensions: it creates at most 62 nonzero signed ports from 31 informative scalar coordinates.

The condition should therefore be described by both **stored width** and **fitted rank**: `PCA128-rank31`. Its zero columns and 63 ordinary padding ports must remain exactly zero. Reports should not imply that all 128 PCA coordinates, 256 signed ports, or 319 engine inputs contain independently fitted information.

PCA128-rank31 is a compact interface experiment, not a way to “use all 319 ports.” Estimating more than 31 informative directions requires more independent fitting observations or an external unsupervised corpus declared before evaluation. Using evaluation texts to fit that corpus would leak distribution information.

### Centering and normalization must be part of the frozen protocol

MRL truncation, centering, projection, normalization, and signed splitting do not commute. Predeclare one exact order per condition. A defensible specification is:

```text
fixed MRL128:
    q = L2_normalize(full_embedding[:128])

centered MRL128:
    q = L2_normalize(full_embedding[:128] - train_mean_128)

PCA rank r:
    q = L2_normalize((full_embedding - train_mean_full) @ V_r)

signed ports:
    p = concat(max(q, 0), max(-q, 0), zero_padding)
```

Fit every mean and PCA basis using the 32 training rows only. Save the basis, mean, singular values, normalization order, model revision, and hashes. Apply them unchanged to validation and test. Stabilize each PC's sign with a declared deterministic convention, such as forcing its largest-magnitude loading positive; sign does not change PCA geometry, but stable signs make port assignments and artifacts reproducible.

Centering can make a zero or near-zero vector for a sample close to the training mean. Define the zero-norm behavior before running. Report pre-normalization norms so normalization does not hide collapse.

### Port identity is an engineered assignment

A rotation or PCA projection changes which scalar reaches each PN root ID. The real edge table remains fixed, but the input meaning assigned to each root changes. The experiment therefore fits an **interface to a fixed anatomy**, not the anatomy itself.

This distinction matters because the PN→KC graph is not exchangeable in detail: PN degree, synapse count, sign, and KC targets vary, and adult studies find sampling biases. A PCA axis placed on a high-degree PN can affect more KCs than the same axis on a low-degree PN. Any gain can arise from matching dataset variance to graph degree rather than discovering a fly sensory basis.

Required port-assignment controls are:

- at least five fixed random permutations of the same transformed coordinates across eligible PN ports;
- a degree-stratified permutation that preserves the distribution of PN out-degree assigned to each coordinate magnitude rank;
- a fixed random orthogonal rotation with the same rank and norm as PCA;
- exact checks that the 97 unsupported PCA columns and 63 padding ports remain zero;
- the existing fixed MRL mapping as the preregistered reference.

Do not select the best random seed and report it as a comparator. Report the full random-control distribution.

## Evaluation design

### Splits and selection

The split must be by semantic family, not by individual sentence. All translations and close paraphrases of one underlying scenario belong to the same group. The proposed sequence is sound only if enforced as follows:

1. **Train 32:** fit means, PCA, semantic ridge reconstruction, and the separately defined reward learner.
2. **Validation 12:** select one transform and any declared rank or normalization choice.
3. **Final authored-family test:** run once after the choice and report every case.

Hash and freeze the final authored family before validation results are inspected. The author must not see model outputs or validation failure modes before writing it. One authored family remains a narrow test and cannot support a broad generalization claim. Prefer several independently authored scenario families, with English and Korean grouped together, and report family-level results rather than treating translations as independent samples.

If the old 32/12 cases influenced earlier architecture or active-fraction decisions, label them inherited development data. The final family is the only confirmatory set for this round.

### Two downstream tasks, two different claims

**Semantic reconstruction** learns a ridge map from simulated activity `H_train` back to the original embedding `E_train`:

```text
B = argmin_B ||H_train B - E_train||_F^2 + lambda ||B||_F^2
E_hat = H_test B
```

Nearest-description category and embedding cosine measure how much supplied input geometry remains linearly accessible. This is not intent decoding. Fit `lambda` on training alone or include it in validation selection; never use final-test reconstruction to choose it.

**Native reward learning** trains the existing KC→MBON action readout from sampled actions and scalar rewards under each mapping, with the same seeds, examples, update budget, and optimizer settings. It measures whether that representation supports the repository's action-learning procedure. It does not reconstruct semantics and is not a supervised class probe.

Do not merge these results. A transform can preserve generic embedding geometry yet harm the four-class boundary, or do the reverse.

### Frozen policy versus retrained readout

Changing the port transform changes the distribution seen by the native policy. Two comparisons answer different questions:

- **Frozen native readout:** compatibility with the existing checkpoint.
- **New readout trained with an identical budget per transform:** learnability under that representation.

The frozen result can favor the original mapping because that mapping trained the checkpoint. The retrained result can favor a new mapping but introduces optimization variance. Report both if feasible, with no checkpoint selection on the final test. Do not call a frozen-readout loss evidence that the representation contains less information.

## Primary metrics and controls

Predeclare one primary metric for each non-combinable target: final-family semantic category accuracy for reconstruction, and final-family greedy action accuracy for the independently reward-trained native readout. Do not average them into one score.

For each transform report:

- mathematical rank and the full singular-value spectrum from training;
- explained variance on training and reconstruction residual on validation/test;
- input norm before and after normalization;
- number of positive and negative signed ports and zero-port fraction;
- KC active count/fraction, dead-case rate, and per-row maximum before normalization;
- within-class and between-class cosine distributions at input and KC stages;
- ridge reconstruction cosine and nearest-description category;
- reward-trained native action accuracy/probabilities and confusion matrix;
- all random port permutations and optimization seeds.

Useful negative controls:

1. zero-reward or shuffled-reward controls for native learning;
2. shuffled `(hidden, embedding)` pairs for reconstruction;
3. constant-mean reconstruction;
4. identity or raw-embedding prototype baseline before the graph;
5. random orthogonal rank-31 projection;
6. permuted port assignments with preserved vector norms;
7. untrained downstream readout to locate information upstream of learning.

### Small-sample stability

With 32 fitting rows, PCA directions can be unstable even though the subspace dimension is exactly bounded. Bootstrap the training cases by semantic family, refit the PCA basis, and compare projection matrices or principal angles. Individual PC correlations are not reliable when singular values are close. Report downstream variation across bootstrap fits without treating the resamples as independent datasets.

PCA is unsupervised with respect to class labels but still fitted to the training inputs. It may capture author, language, sentence length, or template variance. Regress or stratify these nuisance variables and compare language-held-out results. A high semantic score after family leakage would not validate the bridge.

## Layered architecture and replay

A scientifically interpretable implementation should freeze and audit one layer at a time:

```text
Layer 0  frozen EmbeddingGemma vector
Layer 1  train-only interface transform: fixed, centered, PCA, or control
Layer 2  signed ports assigned to fixed PN root IDs
Layer 3  fixed connectome PN→KC expansion plus declared sparsification
Layer 4a semantic reconstruction probe (measurement only)
Layer 4b separately reward-trained KC→MBON/native action readout (behavioral policy)
Layer 5  validation-only selection and hash-locked final evaluation
```

Each artifact should declare the hashes of all upstream artifacts. Do not fit a downstream decoder and then back-propagate its final result into PCA or port assignment. A gain at Layer 4a is retained input information; a gain at Layer 4b is reward-trained task performance. Neither makes Layer 1 anatomical.

In this experiment, **replay means reproducibility replay**. It refits the semantic decoder and reruns saved inference from frozen inputs, then demands exact tensor artifacts and predictions. The native replay independently repeats the seeded CPU reward-training procedure and compares its development report and checkpoints. It is not an experience buffer and it does not add training examples after development.

The command contract should expose the layers explicitly: holdout import/encoding, train-only port fitting, circuit/decoder development, protocol freeze and verification, one final evaluation, then replay. Hash locks must reject changed sources, data, transforms, checkpoints, or final outputs. This is provenance verification, not fly memory replay, sleep consolidation, or a biological KC trace.

## Decision table

| Outcome | Interpretation | Action |
|---|---|---|
| PCA128-rank31 beats fixed and centered MRL on validation and the frozen final family; random rotations/permutations do not | Dataset-fitted low-rank coordinates improve this software interface | Retain as an engineered candidate; do not call axes PN semantics |
| PCA wins validation but not final family | Selection overfit or authored-family shift | Discard the generalization claim; preserve as exploratory |
| Centered MRL matches PCA | Mean removal, not principal-axis ordering, explains the gain | Prefer the simpler centered transform |
| Random rotation or port permutation matches PCA | Orientation/port assignment is not specifically supported | Do not attribute benefit to PCA alignment |
| Unsupported PCA columns or padding become nonzero | Serialization, transform, or indexing artifact | Debug before interpretation |
| Reconstruction improves but reward-trained policy does not | More input geometry is accessible without demonstrated task benefit | Keep reconstruction as descriptive only |
| Frozen policy falls but equally trained new readout improves | Distribution mismatch with the old checkpoint | Treat as interface migration, not information loss |

## Acceptance criteria

A precise positive result requires a train-only transform artifact, explicit rank no greater than 31 for 32 centered training cases, validation-only model choice, a pre-authored untouched family test, complete baselines, and stable direction across family/bootstrap analyses. Report exact case counts and uncertainty descriptively; three seeds do not replace independent datasets.

Even a clean positive result demonstrates improved fitting between frozen language features and a fixed fly-derived topology. Claiming biological sensory alignment would require simultaneous stimuli and PN/KC recordings, known cell registration, train-fitted maps applied to held-out flies, and causal tests showing that the aligned activity predicts or changes fly behavior.
