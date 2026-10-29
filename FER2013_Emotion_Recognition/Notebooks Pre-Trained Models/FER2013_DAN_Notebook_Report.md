# FER2013 Deep Attention Network Notebook Report

## 1. Scope and evidence

This report documents the workspace notebook
`FER2013_Deep_Attention_Netowork.ipynb` in this directory. The spelling
`Netowork` is retained because it is part of the existing filename. The report
uses:

- the notebook's 15 cells and current source;
- the downloaded upstream model definition at
  `models/dan_pretrained/dan.py`;
- the AffectNet-8 checkpoint at
  `models/dan_pretrained/affecnet8_epoch5_acc0.6209.pth`;
- the fine-tuned checkpoint path referenced by the notebook; and
- the saved training history and test-metric JSON files under
  `results/dan_pretrained/`.

The notebook currently has no saved cell outputs or execution counts. Therefore,
the training results in this report are read from the saved JSON artifacts; they
are not reproduced by the notebook file's outputs in its current state. Training
was not rerun to prepare this report.

There is no matching YAML configuration file in this directory. The active
configuration is the Python `CONFIG` dictionary in the notebook's configuration
cell. Editing a separate YAML file will not affect this notebook unless its
configuration-loading code is added.

## 2. Purpose and end-to-end workflow

The notebook fine-tunes the authors' pretrained DAN model on FER2013. It does not
train a wholly new architecture from random initialization:

1. Imports PyTorch, torchvision, NumPy, pandas, scikit-learn, Pillow, tqdm, and
   plotting libraries.
2. Defines paths and training settings in an in-notebook dictionary.
3. Downloads or reuses the upstream DAN Python source and AffectNet-8 checkpoint.
4. Reconstructs the official DAN architecture and loads the AffectNet weights.
5. Replaces the eight-class classifier with a seven-class FER2013 classifier,
   copying weights for the seven overlapping emotions.
6. Loads FER2013 from `train/<emotion>/` and `test/<emotion>/` image folders.
7. Stratifies a validation subset from the official training folder.
8. Fine-tunes the model, selects checkpoints by validation accuracy, and applies
   early stopping.
9. Reloads the best checkpoint and evaluates it against the official test
   loader.
10. Saves the fine-tuned checkpoint, history, metadata, and test metrics.

## 3. Configuration and paths

The notebook's `CONFIG` dictionary currently specifies:

| Setting | Current value | Where it is used |
|---|---:|---|
| Dataset directory | `D:\FYP\Datasets\Facial Datasets\FER2013` | Expected `train/` and `test/` folders |
| Checkpoint directory | `FER2013_Emotion_Recognition\models\dan_pretrained` | Upstream source, pretrained weights, fine-tuned checkpoint |
| Output directory | `FER2013_Emotion_Recognition\results\dan_pretrained` | Training history and test metrics |
| Image size | 224 | Resize dimensions for all splits |
| Batch size | 32 | DataLoader batches; capped at 32 when CUDA is active |
| Workers | 0 | DataLoader worker count |
| Validation fraction | 0.15 | Fraction of the training folder held for validation |
| Seed | 42 | Python, NumPy, and PyTorch random seeds |
| Epoch limit | 20 | Maximum fine-tuning epochs |
| Backbone learning rate | `1e-5` | `model.features` parameters |
| Attention learning rate | `1e-4` | Parameters whose names start with `cat_head` |
| Classifier learning rate | `1e-4` | Final `fc` and `bn` parameters |
| Weight decay | `1e-4` | AdamW optimizer |
| Early-stopping patience | 5 | Non-improving validation-accuracy epochs |
| Training switch | `true` | Controls whether the fine-tuning loop runs |

The official AffectNet checkpoint's Google Drive ID and filename, the upstream
source URL, four attention heads, eight source classes, and seven FER classes are
hard-coded in the notebook. Augmentation settings and scheduler settings are
also hard-coded rather than configurable.

The CUDA batch-size rule only lowers the configured value if it exceeds 32; it
does not currently change the default of 32.

## 4. Model architecture

### 4.1 Backbone and input representation

The notebook imports the upstream `DAN` class and constructs it as
`DAN(num_head=4, num_class=8, pretrained=False)`. In that implementation:

- The feature extractor is torchvision ResNet-18 with its average-pooling and
  classification layers removed.
- The resulting feature tensor has 512 channels. A 224-by-224 image yields a
  7-by-7 spatial feature map at the final ResNet-18 stage.
- The pretrained flag is false here specifically to prevent a separate
  backbone-weight download. The complete AffectNet DAN state dictionary is
  loaded immediately afterwards, including the ResNet-18 trunk.

The ResNet-18 feature trunk follows this spatial progression:

| ResNet-18 portion | Output channels | Spatial size for 224x224 input |
|---|---:|---:|
| 7x7 stem convolution, stride 2 | 64 | 112x112 |
| 3x3 max-pool, stride 2 | 64 | 56x56 |
| Residual layer 1 (2 BasicBlocks) | 64 | 56x56 |
| Residual layer 2 (2 BasicBlocks) | 128 | 28x28 |
| Residual layer 3 (2 BasicBlocks) | 256 | 14x14 |
| Residual layer 4 (2 BasicBlocks) | 512 | 7x7 |

The instantiated seven-class FER2013 model has **19,723,733 parameters**:

| Component | Parameters |
|---|---:|
| ResNet-18 feature trunk | 11,176,512 |
| Four cross-attention heads | 8,543,616 |
| Final `512 -> 7` linear classifier | 3,591 |
| Seven-output BatchNorm | 14 |
| **Total** | **19,723,733** |

Each cross-attention head contains 2,135,904 parameters.

The complete forward path is:

```text
RGB image (3x224x224)
  -> ResNet-18 feature trunk (512x7x7)
  -> four parallel CrossAttentionHead modules
       -> spatial attention mask multiplied with feature map
       -> channel attention and global pooled 512-vector
  -> stack four vectors and log-softmax across the head axis
  -> sum the four head outputs
  -> Linear(512, 7)
  -> BatchNorm1d(7)
  -> FER2013 logits
```

### 4.2 Cross-attention head

Each of the four heads independently applies spatial attention followed by
channel attention to the 512-channel ResNet feature map.

**Spatial attention**

1. A 1-by-1 convolution reduces channels from 512 to 256, followed by
   BatchNorm2d.
2. The reduced representation is passed through three parallel convolutions:
   3-by-3, 1-by-3, and 3-by-1. Each produces 512 channels and has BatchNorm2d.
3. The three outputs are summed and passed through ReLU.
4. Summing across channels forms a single spatial map.
5. This map multiplies the original 512-channel feature map elementwise.

This produces a spatially gated feature map while retaining the original channel
width.

**Channel attention**

1. Adaptive average pooling reduces each channel to one scalar.
2. A two-layer MLP maps `512 -> 32 -> 512`.
3. The hidden layer uses BatchNorm1d and ReLU; the output uses Sigmoid.
4. The sigmoid channel weights multiply the pooled channel vector.

The head returns its resulting 512-dimensional attended representation.

### 4.3 Four-head aggregation and classifier

The four attended vectors are stacked to shape `[batch, 4, 512]`. For multiple
heads, the upstream forward pass applies `log_softmax` across the head dimension
and sums the four head vectors. A linear layer maps the aggregate 512-vector to
the class logits, followed by BatchNorm1d over the class dimension.

The official forward method returns a tuple `(logits, feature_map, heads)`.
Training and evaluation intentionally use element zero, the logits.

### 4.4 Transfer of AffectNet outputs

The source checkpoint's class order is:

`neutral, happy, sad, surprise, fear, disgust, anger, contempt`

The FER2013 order expected by torchvision `ImageFolder` is:

`angry, disgust, fear, happy, neutral, sad, surprise`

The notebook maps `angry` to AffectNet `anger`, copies the matching seven rows
from the source classifier and its BatchNorm parameters/running statistics, and
drops `contempt`. The FER2013 head is therefore initialized with relevant
AffectNet class weights, not random classifier weights. All model parameters
remain trainable during fine-tuning.

## 5. Data and preprocessing

### Expected dataset layout

The loader requires:

```text
FER2013/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

The notebook checks that both split directories exist and that the train and
test class lists equal the expected FER2013 order. It uses the training-folder
labels for a stratified 85%/15% train/validation split with seed 42. The official
test folders are placed in a separate loader and are not passed to the training
loop, scheduler, checkpoint selection, or early stopping.

### Image transforms

All images are resized to 224-by-224, converted to tensors, and normalized by
ImageNet channel statistics:

- Mean: `(0.485, 0.456, 0.406)`
- Standard deviation: `(0.229, 0.224, 0.225)`

Training additionally uses random horizontal flip and random rotation up to 10
degrees. Validation and test use only deterministic resize/tensor/normalization.
The notebook does not configure color jitter, translation, crop, or brightness
augmentation.

## 6. Training procedure

The optimizer is AdamW with three parameter groups:

| Parameter group | Parameters selected | Learning rate |
|---|---|---:|
| Backbone | `model.features` | `1e-5` |
| Attention | names beginning `cat_head` | `1e-4` |
| Classifier | `model.fc` and `model.bn` | `1e-4` |

All groups use weight decay `1e-4`. The loss is cross-entropy with no class
weights. The notebook uses `ReduceLROnPlateau` with `mode='max'`, scheduler
patience 2, and factor 0.3, stepped on validation accuracy.

At every epoch the notebook computes training loss, validation accuracy, and
validation macro-F1. It selects and saves a checkpoint only when validation
accuracy strictly improves. Macro-F1 is logged but does not select the model or
drive early stopping. Early stopping terminates after five consecutive epochs
without a validation-accuracy improvement. The maximum configured epoch count
is 20.

The best checkpoint stores model weights, class labels, a source description,
configuration values, selected epoch, and validation accuracy/macro-F1. The
training-history JSON stores per-epoch train loss, validation accuracy,
validation macro-F1, and the three learning rates.

## 7. Saved training-history results

`results/dan_pretrained/training_history.json` contains 14 epoch records. The
best validation accuracy in that history is **70.07198% at epoch 9**. The best
validation macro-F1 in the logged records is **0.67247 at epoch 14**, which is
not the checkpoint-selection criterion.

| Epoch | Train loss | Validation accuracy | Validation macro-F1 | Backbone / attention / classifier LR |
|---:|---:|---:|---:|---|
| 1 | 1.1285 | 0.6654 | 0.6228 | `1e-5 / 1e-4 / 1e-4` |
| 2 | 1.0348 | 0.6773 | 0.6362 | `1e-5 / 1e-4 / 1e-4` |
| 3 | 0.9807 | 0.6814 | 0.6439 | `1e-5 / 1e-4 / 1e-4` |
| 4 | 0.9345 | 0.6866 | 0.6564 | `1e-5 / 1e-4 / 1e-4` |
| 5 | 0.9006 | 0.6917 | 0.6560 | `1e-5 / 1e-4 / 1e-4` |
| 6 | 0.8577 | 0.6926 | 0.6643 | `1e-5 / 1e-4 / 1e-4` |
| 7 | 0.8222 | 0.6917 | 0.6575 | `1e-5 / 1e-4 / 1e-4` |
| 8 | 0.7857 | 0.6979 | 0.6611 | `1e-5 / 1e-4 / 1e-4` |
| **9** | **0.7479** | **0.7007** | 0.6638 | `1e-5 / 1e-4 / 1e-4` |
| 10 | 0.7171 | 0.6898 | 0.6612 | `1e-5 / 1e-4 / 1e-4` |
| 11 | 0.6813 | 0.6984 | 0.6712 | `1e-5 / 1e-4 / 1e-4` |
| 12 | 0.6512 | 0.6940 | 0.6664 | `1e-5 / 1e-4 / 1e-4` |
| 13 | 0.6001 | 0.6984 | 0.6706 | `3e-6 / 3e-5 / 3e-5` |
| 14 | 0.5834 | 0.6986 | **0.6725** | `3e-6 / 3e-5 / 3e-5` |

The history ends at epoch 14. This is consistent with early stopping: after the
epoch-9 validation-accuracy peak, epochs 10 through 14 do not exceed that
accuracy, reaching the configured five-epoch patience. Training loss continues
to decrease, while validation accuracy mostly plateaus/fluctuates. Because
validation loss and training accuracy are not logged, the history is insufficient
to quantify a train/validation loss gap or diagnose overfitting conclusively.
No per-epoch duration is recorded.

## 8. Saved official-test results

`results/dan_pretrained/test_metrics.json` reports results on **7,178** test
images and references
`models/dan_pretrained/dan_fer2013_finetuned_best.pt`.

| Metric | Saved result |
|---|---:|
| Test accuracy | 0.68376 (68.38%) |
| Test macro precision | 0.65038 |
| Test macro recall | 0.64263 |
| Test macro-F1 | 0.64475 |
| Test weighted precision | 0.68239 |
| Test weighted recall | 0.68376 |
| Test weighted-F1 | 0.68138 |

| FER2013 class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Angry | 0.5974 | 0.6660 | 0.6298 | 958 |
| Disgust | 0.5283 | 0.5045 | 0.5161 | 111 |
| Fear | 0.5333 | 0.4219 | 0.4711 | 1,024 |
| Happy | 0.8729 | 0.8940 | 0.8833 | 1,774 |
| Neutral | 0.6354 | 0.6869 | 0.6602 | 1,233 |
| Sad | 0.5720 | 0.5958 | 0.5837 | 1,247 |
| Surprise | 0.8134 | 0.7292 | 0.7690 | 831 |

The weakest class F1 is fear (0.4711); happy is strongest (0.8833). Disgust has
only 111 test examples, so its score is comparatively uncertain. The saved
metrics JSON does not contain test loss, inference latency, or throughput, and
the notebook does not measure them.

These are recorded artifact results, not newly reproduced results. The notebook
source currently has no outputs. The project dataset has had duplicate-image
risks noted in prior audits; this notebook does not hash or remove duplicates
across its train, validation, and test folders, so the test result should not be
described as proven duplicate-free.

## 9. Reproducibility and implementation notes

- Random seeds are set for Python, NumPy, and PyTorch. CUDA seeds are set when
  CUDA is available. cuDNN deterministic mode is not explicitly enabled.
- No explicit `torch.Generator` is passed to the training DataLoader. The global
  PyTorch seed is set, but exact reproducibility may still vary by platform and
  runtime.
- The source file is downloaded from the moving GitHub `main` branch when absent.
  No source commit hash or source-file hash is recorded.
- The checkpoint download is accepted based on a minimum file size; the notebook
  does not verify a published cryptographic digest.
- The model architecture, class ordering, augmentation, scheduler mode/settings,
  and checkpoint filenames are partly hard-coded, not all tunable through a
  standalone configuration file.
- The test set is only evaluated after fitting in the code flow. Rerunning the
  test-evaluation cell repeatedly is still possible; test metrics should not be
  used to tune future model settings.
- `CONFIG['run_training'] = False` skips training, but the later test cell will
  evaluate an existing fine-tuned checkpoint if one is present. The output
  should therefore be interpreted as evaluation of that checkpoint, not a
  training run from the current configuration.

## 10. Artifact inventory

| Artifact | Location | Role |
|---|---|---|
| Notebook | `Notebooks Pre-Trained Models/FER2013_Deep_Attention_Netowork.ipynb` | Download, transfer, fine-tune, evaluate |
| Upstream source | `models/dan_pretrained/dan.py` | DAN architecture definition |
| AffectNet-8 weights | `models/dan_pretrained/affecnet8_epoch5_acc0.6209.pth` | Source pretrained model |
| FER2013 best checkpoint | `models/dan_pretrained/dan_fer2013_finetuned_best.pt` | Best validation-accuracy weights |
| Training history | `results/dan_pretrained/training_history.json` | 14 epoch records |
| Test metrics | `results/dan_pretrained/test_metrics.json` | Test metrics, class report, confusion matrix |
| This report | `Notebooks Pre-Trained Models/FER2013_DAN_Notebook_Report.md` | Architecture and pipeline documentation |

## 11. Overall assessment

This notebook implements transfer learning using the official four-head DAN
architecture and an AffectNet-8 checkpoint. It maps pretrained weights for all
seven FER2013 classes, fine-tunes the full network with parameter-group learning
rates, and selects the model using a stratified validation split rather than
test accuracy. The saved results report 68.38% test accuracy and 0.64475 macro
F1.

The core architecture and transfer mapping are explicit in the notebook and
vendored source. The main documentation/reproducibility limitation is that this
pretrained workflow's configuration is embedded in Python, not a matching YAML
file; several consequential settings remain hard-coded. The persisted metrics
provide evidence of a completed run, but the notebook itself does not preserve
cell outputs, and the saved test result has not been independently regenerated
for this report.
