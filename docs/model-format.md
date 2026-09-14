# Novigrad v1 Safetensors model

Checkpoints use the official Safetensors format. The current `__metadata__` values are `format = nobi.plastic` and `version = 4`. Version 4 identifies the file schema changed during development; the product scope remains v1. Schema 3 remains readable with unit output gains; earlier experimental schemas are rejected.

The crate and CLI are now named `novi` and `novi_engine`. Existing on-disk identifiers (`nobi.plastic`, `nobi.image_rate.v1`, `nobi.image_bundle.v1`, and `nobi.classifier_dataset.v1`) are retained unchanged for checkpoint and image-bundle compatibility. Existing models require no conversion.

| Tensor | dtype | Meaning |
| --- | --- | --- |
| `input_ids` | U64 | Sorted external input root IDs |
| `hidden_ids` | U64 | Union of hidden IDs referenced by either layer |
| `output_ids` | U64 | Sorted output root IDs |
| `input_pre`, `input_post` | U64 | Input/hidden indices of fixed-layer edges |
| `input_count` | U64 | Original synapse counts of fixed-layer edges |
| `input_sign` | I8 | Fixed-edge sign: -1, 0, or +1 |
| `plastic_pre`, `plastic_post` | U64 | Hidden/output indices of plastic-layer edges |
| `plastic_count` | U64 | Original synapse counts of plastic-layer edges |
| `plastic_sign` | I8 | Preserved sign of each plastic edge |
| `plastic_weight` | F32 | Current learned weight of each plastic edge |
| `output_actions` | U64 | Action index assigned to each output neuron |
| `output_gains` | F32 | Fixed external readout gains in [-1,1]; not biological KC synapse signs |

All 14 tensors are one-dimensional. Edge order is preserved so summation and resumed execution can be reproduced on the same platform. Input weights are reconstructed deterministically from original counts and signs. Metadata also records `actions`, `learning_rate`, `logit_gain`, `active_fraction`, `weight_limit`, and `homeostasis`.

Real neuron IDs are never converted to F32. Transient activity and eligibility before reward, the external input generator's RNG, and the epoch cursor are not stored. Saving is allowed only after `reward` or `clear_pending` completes a trial. Evaluation and inference do not require PyTorch or pickle.
