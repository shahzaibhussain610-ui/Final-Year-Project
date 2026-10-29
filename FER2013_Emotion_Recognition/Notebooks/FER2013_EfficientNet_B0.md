# FER2013 EfficientNet-B0 Notebook Guide

This guide documents the notebook [`FER2013_EfficientNet_B0.ipynb`](./FER2013_EfficientNet_B0.ipynb) from setup through inference. It describes the implementation and configuration in the notebook, explains how to run it, identifies the outputs it produces, and records important interpretation limits.

## 1. Purpose and task

The notebook builds an image classifier for the seven FER2013 facial-expression classes:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

The source images are pre-cropped 48 × 48 grayscale faces. The pipeline converts them to three-channel images at 224 × 224 for torchvision EfficientNet-B0. The classifier returns one class prediction and, through the inference helper, a probability for every class plus confidence and latency.

The primary experiment is EfficientNet-B0. The notebook also includes optional DeepFace baseline evaluation and hyperparameter-tuning machinery. Its opening text mentions ViT-B/16 as a separate experiment, but this guide focuses on the EfficientNet workflow documented by the main sections.

## 2. End-to-end workflow

Run the notebook from top to bottom in one kernel. The major stages are:

1. Load dependencies, locate the YAML configuration and select CPU or CUDA.
2. Seed Python, NumPy and PyTorch for reproducibility.
3. Inventory the image files and audit image readability, dimensions, modes, hashes and duplicates.
4. Explore class frequencies and representative images.
5. Preserve the official test partition and split the official training data into train and validation partitions.
6. Build image transforms, a dataset wrapper and data loaders.
7. Create EfficientNet-B0 and replace its ImageNet classifier with a seven-class head.
8. Train, select checkpoints and record history using validation metrics.
9. Optionally run validation-only tuning.
10. Evaluate the selected checkpoint on the held-out official test split; save metrics, confusion matrices and error rows.
11. Optionally evaluate DeepFace on the same official test images.
12. Verify checkpoint loading, demonstrate single-image inference and summarize results and limitations.

The official test set is meant for final reporting only. Training, scheduling, early stopping, checkpoint selection and tuning use the validation split.

## 3. Repository and required inputs

The notebook expects a FER2013 directory arranged as class subfolders under `train` and `test`, for example:

```text
FER2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   └── ...
└── test/
    ├── angry/
    ├── disgust/
    └── ...
```

The exact dataset root is configured in `FER2013_EfficientNet_B0.yaml`. The notebook finds that configuration relative to the current working directory or a `Notebooks` directory above it. It uses the configuration's parent-parent as the project root and creates these directories when needed:

```text
FER2013_Emotion_Recognition/
├── models/
└── results/
    ├── eda/
    ├── metrics/
    └── plots/
```

Required packages include PyTorch, torchvision, NumPy, pandas, Pillow, scikit-learn, matplotlib, seaborn, tqdm and PyYAML. DeepFace and its compatible backend are only needed for the optional baseline. Use an environment with versions compatible with each other; the notebook records Python, PyTorch and torchvision versions in its environment summary.

## 4. Configuration reference

The adjacent YAML file is the primary source for run settings. The checked-in configuration has the following values:

| Setting | Configured value | Meaning |
|---|---:|---|
| `seed` | 42 | Seeds Python, NumPy, PyTorch and data-loader shuffling. |
| `image_size` | 224 | Square model input resolution. |
| `validation_fraction` | 0.15 | Fraction of official training images held out for validation. |
| `batch_size` | 64 | Batch size configured for loaders. |
| `num_workers` | 0 | Data-loader worker processes. |
| `stage1_epochs` / `stage2_epochs` | 30 / 20 | Maximum epochs in the two training phases. |
| `optimizer` | AdamW | Optimizer family. |
| `learning_rate` / `fine_tune_learning_rate` | 0.001 / 0.0003 | Stage learning rates. |
| `weight_decay` | 0.0001 | Optimizer weight decay. |
| `pretrained` | false | Whether to request ImageNet pretrained weights. |
| `dropout` | 0.30 | Dropout probability in the replacement classifier head. |
| `unfreeze_layers` | 40 | Number of final feature children re-enabled in phase two. |
| `scheduler` | cosine | Learning-rate scheduler choice (`cosine` or `plateau`). |
| `min_learning_rate` | 0.00001 | Scheduler lower bound. |
| `label_smoothing` | 0.0 | Cross-entropy label smoothing. |
| `horizontal_flip` | 0.5 | Training horizontal-flip probability. |
| `rotation_degrees` | 10 | Training affine rotation range. |
| `translate_fraction` | 0.05 | Training affine translation fraction. |
| `brightness_factor` | 0.1 | Training brightness jitter. |
| `class_weighted_loss` / `balanced_sampling` | false / false | Optional imbalance mitigations; both are disabled in this configuration. |
| `patience` / `monitor` | 7 / `val_loss` | Early-stopping patience and validation criterion. |
| `save_best_model` / `run_training` | true / true | Save selected checkpoint and allow training. |
| `metrics` | accuracy, macro F1, per-class recall, confusion matrix | Configured evaluation outputs. |

The notebook validates many configuration values before training. The code also has independent controls for optional pretrained download, tuning trials and the DeepFace baseline. Review those cells before starting expensive work.

## 5. Runtime setup and reproducibility

The setup cell imports the Python and ML stack, reads YAML into a frozen `Config` dataclass, creates output directories, selects `cuda` when available and otherwise uses `cpu`, and seeds Python's `random`, NumPy, PyTorch and CUDA. cuDNN deterministic mode is enabled and benchmarking disabled.

Deterministic settings improve repeatability but do not guarantee bit-for-bit identical results across every hardware, CUDA, library or kernel combination. The notebook also prints software versions and resolved paths so a run can be described accurately.

## 6. Dataset audit and exploratory analysis

The inventory stage walks the official train/test folders and records file path, class label and source split. It checks that images decode, records dimensions and color modes, computes SHA-256 hashes, and flags exact duplicates. The audit is intended to expose malformed or surprising inputs before modeling. Duplicate files are reported, not removed, so the source benchmark remains intact.

EDA then summarizes per-class and per-split counts, plots the class distribution, displays a representative face for each emotion and samples images for visual inspection. A dataset report is written to `results/eda/dataset_report.json`.

The notebook's saved narrative describes a substantial imbalance (for example, many more `happy` samples than `disgust`). This matters because accuracy can hide poor performance on rare classes; macro-averaged metrics and per-class recall are needed alongside accuracy.

## 7. Split design and leakage control

The notebook keeps the official `test` folder separate. It uses `train_test_split` with stratification and the configured seed to carve validation data out of the official `train` folder. Validation is used for learning-rate decisions, early stopping, checkpoint selection and optional hyperparameter tuning. The test set is evaluated after those choices are complete.

One saved notebook run reports 28,709 official training images split into 24,402 training and 4,307 validation images, with 7,178 official test images. These counts describe the saved run; the actual counts depend on the dataset directory contents.

## 8. Preprocessing and data loading

`FERFrameDataset` opens an inventory row's image and returns `(image_tensor, class_index, path)`. Keeping the path lets later evaluation link mistakes back to source images.

All images are converted to RGB and resized to 224 × 224. Training applies random horizontal flip, small random affine rotation/translation and brightness jitter. Validation and test use deterministic conversion and resize only. Both pipelines normalize with the mean and standard deviation from torchvision's default EfficientNet-B0 weight transform. These statistics are used even if pretrained checkpoint weights are unavailable.

The train loader shuffles by default and receives a seeded generator. If `balanced_sampling` is enabled, it instead uses a `WeightedRandomSampler`. Validation and test loaders are ordered and deterministic. CUDA enables pinned-memory loading. A sanity-check batch displays shape, labels and normalized value range.

## 9. EfficientNet-B0 architecture

The backbone is torchvision's `efficientnet_b0`. It consists of the EfficientNet-B0 convolutional feature extractor and the standard pooling/classifier path. The original 1,000-class final linear layer is replaced with:

```text
Dropout(p=0.30) → Linear(1280, 7)
```

The seven logits correspond to the inventory's class order. A softmax is applied at inference to turn logits into probabilities; training uses logits directly with cross-entropy.

When pretrained use is requested, the helper first checks the local Torch checkpoint cache and then (if allowed) attempts to obtain the torchvision default ImageNet weights. A failure is caught and results in random initialization with a warning. With `pretrained: false`, as in the checked-in YAML, the model is constructed without pretrained weights regardless of cache availability.

The notebook provides `freeze_backbone` and `unfreeze_last_layers`. In a genuine pretrained experiment, stage one freezes convolutional features and trains the classifier head; stage two unfreezes the requested final feature blocks plus the head. When starting from scratch, stage one trains the full model because there is no useful pretrained representation to freeze. The recorded narrative reports about 4.0 million total parameters; exact trainable counts depend on the selected stage and configuration.

## 10. Training procedure

The `Trainer` owns the loss, optimizer, scheduler, epoch loop, validation, early stopping, checkpointing and history. It uses cross-entropy with optional class weights and configured label smoothing. Class weights are derived from training-set frequencies only when `class_weighted_loss` is enabled.

The configured optimizer is AdamW. The two stages use their respective configured learning rates. The scheduler supports cosine annealing or ReduceLROnPlateau, with the configured minimum rate. At each epoch the trainer records training and validation loss/accuracy and learning rate. The configured monitor (`val_loss` here) controls improvement and early stopping; patience is seven epochs.

The best validation checkpoint is written as `models/efficientnet_b0_fer2013.pt` (based on `model_name`). Training history is written to `results/metrics/training_history.json`. The checkpoint stores model state, class names/order, preprocessing information and configuration so inference can reconstruct the experiment.

The configured epoch maxima are 30 and 20, but early stopping can end a stage earlier. Training on CPU can take a long time. Inspect the training flag and runtime before starting; rerunning training overwrites the named checkpoint/history artifacts.

## 11. Training curves and optional tuning

The history visualization plots training and validation accuracy/loss and learning rate, and marks the stage boundary. It helps distinguish underfitting, overfitting and optimization problems.

The tuning helper starts a fresh model for each trial and selects by validation accuracy. Trial checkpoints and a CSV summary are saved under `models/` and `results/metrics/`. `TUNING_TRIALS` is empty in the notebook, so the saved state is `Not Yet Evaluated`. The official test data is not used for tuning.

## 12. Final test evaluation and error analysis

After training is complete, the notebook reloads the best validation checkpoint and evaluates the official test loader. It calculates:

- Accuracy: total correct predictions divided by test examples.
- Macro precision, recall and F1: unweighted average across the seven classes; useful for imbalance.
- Weighted precision, recall and F1: class metrics weighted by class support.
- Per-class precision, recall, F1 and support.
- Raw and row-normalized confusion matrices.
- Parameter count, parameter-file size estimate and average forward-pass latency.

The detailed metric JSON and confusion-matrix PNGs are written below `results/metrics/` or `results/plots/` as specified in the evaluation call; misclassified rows are saved as `efficientnet_misclassified.csv`. Error analysis ranks frequent actual-to-predicted confusions, sorts classes by recall and displays a gallery of misclassified faces.

The metric definitions are not interchangeable: accuracy can be dominated by common classes, while macro F1 gives every class equal influence. Always inspect per-class recall and the confusion matrix before describing the model as useful.

## 13. Optional DeepFace baseline

The baseline cell imports DeepFace at runtime and runs its emotion detector on the same official test paths. It maps a predicted emotion to a FER2013 label only when the names match exactly; failures and unmapped labels are recorded. It reports metrics for the successfully mapped/evaluated subset, so compare its evaluated sample count and failures as well as its scores. It can be slow and may require downloading model weights and installing compatible TensorFlow/DeepFace dependencies.

The cell has `RUN_DEEPFACE_BASELINE = True`, but it still depends on the preceding data cells and import/runtime availability. A saved notebook narrative says the baseline was not executed. Treat the baseline as unavailable unless a current run produced its metrics artifact.

## 14. Checkpoint verification and inference

Checkpoint verification reports its size and stored keys, rebuilds the architecture, loads the saved state and performs a dummy forward pass. This checks structural compatibility; it does not prove that model accuracy is good.

`predict_emotion(image_path)` loads the checkpoint, rebuilds the model, reproduces the recorded image preprocessing and returns:

```text
{
  "emotion": predicted class,
  "confidence": probability of predicted class,
  "probabilities": {each class: probability},
  "latency_ms": forward-pass latency
}
```

The helper can also display the input and probability bar chart. Its input should be a face image similar to the pre-cropped FER2013 data; this notebook does not include face detection or alignment for arbitrary full-scene photographs.

## 15. Outputs and artifact map

| Artifact | Purpose |
|---|---|
| `results/eda/dataset_report.json` | Dataset audit and class summary. |
| `models/efficientnet_b0_fer2013.pt` | Best validation-selected checkpoint and inference metadata. |
| `results/metrics/training_history.json` | Epoch-level losses, accuracies and learning rates. |
| `results/metrics/efficientnet.json` | Test metrics and classification report (name may follow the evaluation output directory). |
| `results/metrics/efficientnet_misclassified.csv` | Test examples the model misclassified. |
| Confusion-matrix PNG files | Raw and normalized test confusion matrices. |
| `results/metrics/hyperparameter_trials.csv` | Tuning results if trials are configured and run. |
| `results/metrics/deepface.json` | DeepFace baseline results if the optional evaluation completes. |

Exact output locations are set by the notebook's `METRICS_DIR`, `PLOTS_DIR` and evaluation call; confirm those cells if you change project layout.

## 16. Accuracy and validation results

The notebook has stored results from at least two executions. The tables below preserve those values and identify the execution context shown by each saved output. They must not be merged into one result set.

### Results shown in the final summary cell

| Metric | Saved value |
|---|---:|
| Best validation accuracy | **68.31%** |
| Official test accuracy | **67.94%** |
| Test macro F1 | **64.89%** |
| Test weighted F1 | **67.51%** |
| Training images | 24,402 |
| Validation images | 4,307 |
| Official test images | 7,178 |

These figures appear in the final summary cell's stored output. The saved output identifies the model as `efficientnet_b0_fer2013` and says pretrained initialization was not requested (`False`). The stored summary alone does not prove that its metrics, checkpoint and history all came from the same current run; verify the generated JSON/checkpoint artifacts before citing them.

### Earlier evaluation and checkpoint narrative

| Metric / observation | Earlier saved value |
|---|---:|
| Test accuracy | **25.38%** |
| Test macro F1 | **9.11%** |
| Test weighted F1 | **13.56%** |
| Best validation accuracy / checkpoint narrative | approximately **25.7%** |
| Interpretation | Predictions collapsed heavily toward `happy`; `disgust` was not correctly predicted. |

This earlier narrative attributes the poor result to random initialization, CPU-only short training and class imbalance. It is inconsistent with the final-summary metrics above, so treat it as a separate saved execution, not as a per-epoch explanation of the 68.31%/67.94% results.

### How validation is used

Validation is carved from the official training split with a stratified 15% split and seed 42. It is used to monitor `val_loss`, schedule learning rates, trigger early stopping, select the best checkpoint and score optional tuning trials. The held-out official test split is reserved for final evaluation. The configured maximum training schedule is 30 epochs for stage one and 20 for stage two, with patience 7; early stopping may stop sooner.

Validation accuracy and test accuracy measure different splits. The best validation accuracy is the selected checkpoint's validation score, while test metrics are calculated afterward on the official test set. The notebook's saved outputs currently contain two incompatible validation/test result sets; use a fresh complete execution and matching `training_history.json`, checkpoint and test metrics JSON as the authoritative run record.

### Result integrity note

The checked-in YAML sets `pretrained: false`, while some saved notebook prose describes an attempted pretrained download that failed. The notebook also contains different training/evaluation narratives and metric outputs. Before reporting any score, confirm the effective configuration, the run's checkpoint and history timestamps, and the corresponding `efficientnet.json` output. Do not present the two tables as one experiment or describe the saved values as validated transfer-learning performance without reconciling their provenance.

The main configuration disables class-weighted loss and balanced sampling. The optional DeepFace baseline and hyperparameter tuning are described as not completed in one saved run. Duplicate files are reported but not removed. These facts affect interpretation and reproducibility.

## 17. Recommended execution checklist

1. Confirm the YAML dataset path exists and has `train/<class>` and `test/<class>` directories.
2. Activate the intended Python environment and confirm compatible PyTorch/torchvision versions.
3. Decide explicitly whether the experiment is from scratch or uses ImageNet initialization; record the effective setting and weight-loading message.
4. Review compute availability, training flags, epochs and optional DeepFace/tuning switches.
5. Run all cells in order and inspect the dataset audit, split counts and class order.
6. Confirm training history and the best checkpoint were written during this execution.
7. Confirm the final test metrics and confusion matrices correspond to that checkpoint.
8. Record device, software versions, configuration, checkpoint identity and metrics together when writing a report.

## 18. Scope and limitations

This notebook classifies pre-cropped faces; it is not a complete camera-based emotion-recognition application. FER2013 labels are dataset annotations and predictions should be interpreted as expressions in this benchmark, not definitive measurements of a person's internal emotional state. Generalization beyond the dataset, demographic fairness, calibration and real-world image robustness are not established by this notebook alone.
