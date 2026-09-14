# Discovering better learning mechanisms

[한국어](optimization.ko.md)

Novigrad separates biological connectivity from engineered input encoding, optimization and output decoding. A useful optimization should explain which component changes, preserve the declared circuit constraints, and survive a held-out comparison.

The experiment tools compare mechanisms while keeping the FlyWire input/plastic edge tables and synapse signs fixed. They do not substitute a pretrained vision model or add dense neural layers.

## Repeatable experiments

```sh
.venv/bin/python scripts/experiment_vision.py baseline
.venv/bin/python scripts/experiment_vision.py no-homeostasis --homeostasis false
.venv/bin/python scripts/experiment_vision.py longer75 --homeostasis false --epochs 75 --patience 10
.venv/bin/python scripts/experiment_vision.py batch16 --homeostasis false --epochs 75 --patience 10 --batch-size 16 --lr .0016
.venv/bin/python scripts/report_vision_experiments.py
```

Each named experiment trains one fixed seed on both tasks and writes its complete configuration, epoch curves and validation metrics under `results/improvement/NAME/`. Two task processes run concurrently on CPU; this limits parallel work on a Mac. Feature matrices and adapters are reused when their configuration matches. Test metrics remain null in this phase. The same `train_classifier` can be invoked directly for arbitrary feature datasets and circuits.

`--epochs` controls complete passes over the training data. `--patience` allows early stopping when validation accuracy, with cross-entropy as a tie-breaker, stops improving. The final checkpoint comes from validation, not necessarily the last epoch.

`--batch-size` controls how many sample gradients are accumulated at unchanged weights before applying an update. A larger batch is different from another pass over the dataset. The mean gradient is applied once, then synapse signs, individual magnitude limits and optional output normalization are enforced. Batch size 1 uses the existing online update path exactly. Partial batches at the end of an epoch are applied. Accumulation reuses a buffer rather than allocating per sample.

Learning rate is per mean-gradient update. The batch16 example explicitly multiplies the online learning rate by 16 to roughly preserve the total update scale per epoch. That is an experimental choice, not automatic behavior or a guarantee that every batch size should use linear scaling.

## Mechanisms under test

- **Output weight normalization:** `--homeostasis true` constrains every MBON's incoming absolute weight sum to at most 1 after each update. Disabling it preserves the edge topology, signs and per-edge magnitude bound, while allowing the total evidence carried by an output neuron to grow. Inspect accuracy, cross-entropy, output L1 mass and clipping fractions together.
- **Output gain:** raising `--gain` changes both softmax evidence and gradient scale. In the gain control, learning rate is reduced proportionally to avoid accidentally increasing both at once. This helps distinguish score scale from the effect of normalization.
- **Training duration:** additional epochs can help an undertrained model, but validation checkpoint selection guards against keeping a worse late epoch.
- **Hidden activity:** `--active` changes the fraction of winning hidden neurons. This trades sparse selectivity against retained information without adding connections.
- **Adapter capacity:** `--components` and `--whitening` change PCA compression and scaling, fitted using training images only.
- **Minibatch updates:** sample gradients are averaged before projection, reducing dependence on the immediately preceding sample. Measure this rather than assuming that larger batches improve accuracy or runtime.

The public library APIs are `accumulate_reward(action, reward)`, `apply_batch()` and `discard_batch()`. Saving a checkpoint with an unapplied batch is rejected; optimizer accumulation is transient. Decoder remapping during accumulation and mixing an open batch with online `reward()` are rejected. Existing completed-trial Safetensors checkpoints remain readable.

## Final confirmation

The v0.1.0 test results are already known. New configurations are therefore selected using validation only, then compared on previously unused official Fashion-MNIST test images and new synthetic CAPTCHA strings excluded from every original split.

```sh
.venv/bin/python scripts/prepare_vision_holdout.py
.venv/bin/python scripts/confirm_vision_improvement.py results/improvement/selection.json --out results/improvement/reproduction
.venv/bin/python scripts/report_vision_improvement.py --confirmation results/improvement/reproduction
```

The holdout generator refuses accidental overwrite. Confirmation fixes the configuration first, completes three training seeds and a shuffled-label control, then evaluates the held-out images. A released baseline model is evaluated on the same images. CAPTCHA exact success means all four digits are correct. Shared fonts and fixed cell positions remain limitations.

The independent evaluator also accepts another labeled dataset without passing labels into inference:

```sh
.venv/bin/python scripts/evaluate_image_model.py BUNDLE.json IMAGES.npz --out NEW_RESULT_DIR
```

This produces class accuracy, complete-string accuracy, cross-entropy, confidence, confusion matrices, model sizes and measured adapter/backend time. Timing includes one Rust process per batch; it is not a hardware NPU benchmark.

## Interpreting a gain comparison

Multiplying the logits of an already trained model by a positive constant does not change its argmax. The gain experiment retrains the circuit: even when `learning_rate × logit_gain` is held fixed, softmax probabilities change the error term and therefore which synaptic updates are emphasized. It is not a post-processing calibration trick. Output L1 normalization also changes the relative contribution of incoming weights after each update. These comparisons identify useful engineering choices; they do not prove that one biological process alone caused the original limitation.
