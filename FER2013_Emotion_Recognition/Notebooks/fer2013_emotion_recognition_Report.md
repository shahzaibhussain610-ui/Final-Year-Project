# Research-Grade Facial Emotion Recognition Pipeline (FER2013) — Full Notebook Report

**Source Notebook:** `D:\FYP\FER2013_Emotion_Recognition\Notebooks\fer2013_emotion_recognition.ipynb`  
**Configuration:** `D:\FYP\FER2013_Emotion_Recognition\Notebooks\config.yaml`  
**Report Generated:** 2026-10-15  
**Notebook Structure:** 38 cells (markdown + code), fully executed with saved outputs

---

## 1. Project Overview & Research Objectives

A research-grade benchmarking pipeline for **Facial Emotion Recognition (FER)** on the **FER2013** dataset, covering 7 emotion classes: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral` (48×48 grayscale facial crops).

**Benchmark architectures declared:** EfficientNet-B3 (baseline), Vision Transformer ViT-B/16, ConvNeXt-Tiny, and DDAMFN (Dual-Direction Attention Mixed Feature Network — custom FER architecture).

**Core research guidelines:**

1. **Strict test isolation** — the official test set (`test/`) is quarantined until final evaluation; never used for hyperparameter search or checkpoint selection.
2. **Macro-F1 focus** — because `Disgust` is only ~1.5% of samples, accuracy alone is misleading; model selection prioritizes Macro-Averaged F1 and per-class recall.
3. **Two-phase transfer learning** — frozen-backbone classifier-head warm-up, then full fine-tuning with discriminative learning rates (`backbone_lr < head_lr`).
4. **Fair experimental protocol** — all architectures share identical stratified splits, augmentation, loss formulation, and seeds.
5. **Complete decoupling via `config.yaml`** — all hyperparameters and paths driven externally, no in-notebook edits needed.

---

## 2. Environment & Hardware (Actual Run)

| Item | Value |
|---|---|
| Operating System | Windows 10 (AMD64) |
| Python | 3.11.9 |
| PyTorch | **2.5.1+cpu** |
| CPU | 32 logical cores |
| System RAM | 127.69 GB |
| GPU | **None — entire training ran on CPU** |
| Mixed Precision (AMP) | Disabled (no CUDA) |

> ⚠️ **Caveat:** All training times below are CPU times (8–38 hours per model). Results would likely improve with GPU + AMP enabled.

Reproducibility: **seed = 42**, deterministic mode enabled (Python `random`, NumPy, PyTorch CPU/CUDA seeds, cuDNN deterministic).

---

## 3. Configuration (`config.yaml`)

Loaded from `Notebooks\config.yaml`. Output directories:

- Checkpoints: `Notebooks\checkpoints\`
- Results: `Notebooks\results\`
- Figures: `Notebooks\results\figures\`
- Export: `Notebooks\exported_models\`

Key global settings:

- Loss: **weighted Cross-Entropy** (`weighted_ce`) with **label smoothing 0.05** (FocalLoss implemented as an alternative but unused)
- Optimizer: **AdamW**, grad clip 1.0, **cosine scheduler** (min LR 1e-6), 1 warmup epoch
- Early stopping: patience 5, monitored on **val Macro-F1**
- Hyperparameter search: **DISABLED** (verified default research hyperparameters used)
- External dataset evaluation (RAF-DB etc.): **DISABLED**

---

## 4. Dataset Discovery, Schema Validation & Integrity

- **Source:** `D:\FYP\Datasets\Facial Datasets\FER2013`
- Auto-detected format: **image_folder** (`train/<emotion>/`, `test/<emotion>/`)
- **35,887 validated facial samples** (MD5 hashing for integrity/dedup checks)
- Native resolution: 48×48 grayscale, converted to **3-channel RGB** for pretrained backbones (custom `FERDataset`)

**Partitions:**

| Split | Count |
|---|---|
| Train (official) | 28,709 |
| Test (official) | 7,178 |
| Train (final, after stratified val split) | **24,402** |
| Validation (15% stratified from train) | **4,307** |
| Test (quarantined until final eval) | **7,178** |

---

## 5. Exploratory Data Analysis & Class Imbalance

- **Imbalance ratio: 16.55×** (largest vs. smallest training class) — flagged as "Severe Imbalance"
- Inverse-frequency **class weights** computed for cost-sensitive loss:

| Class | Weight | Class | Weight |
|---|---|---|---|
| Angry | 1.0265 | **Disgust** | **9.3962** |
| Fear | 1.0011 | Happy | 0.5684 |
| Sad | 0.8492 | Surprise | 1.2935 |
| Neutral | 0.8261 | | |

- Visualizations saved: `class_distribution.png`, `sample_faces.png`

---

## 6. Data Preprocessing & Augmentation (Expression-Preserving)

**Train pipeline:**
`Resize(224) → RandomHorizontalFlip(p=0.5) → RandomRotation(±10°) → RandomAffine(translate 0.06) → ColorJitter(brightness/contrast 0.85–1.15) → ToTensor → Normalize(ImageNet mean/std) → RandomErasing(p=0.1, scale 0.02–0.20)`

**Eval pipeline:** `Resize(224) → ToTensor → Normalize(ImageNet mean/std)`

Normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

---

## 7. Models & Architectures — Pretrained vs. Scratch

### Summary Table

| Model | Type | Params | Head params | **Weights** | Input | Batch | Epochs (freeze+FT) | backbone_lr | head_lr | WD | Dropout |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **EfficientNet-B3** | CNN (MBConv) | 10.71M | 10.8K | ✅ **Pretrained** (ImageNet `DEFAULT`) | 224 | 32 | 3+15 | 2e-5 | 2e-4 | 0.01 | 0.3 |
| **ViT-B/16** | Transformer | 85.80M | 5.4K | ✅ **Pretrained** (ImageNet `DEFAULT`, 330MB weights) | 224 | 16 | 3+15 | 1e-5 | 1.5e-4 | 0.05 | 0.1 |
| **ConvNeXt-Tiny** | Modernized ConvNet | 27.83M | 5.4K | ✅ **Pretrained** (ImageNet `DEFAULT`, 109MB weights) | 224 | 32 | 3+15 | 2e-5 | 2e-4 | 0.01 | 0.2 |
| **DDAMFN** (custom) | Attention FER CNN | 0.72M | 3.6K | ❌ **Scratch** | 224 | 32 | 2+15 | 1e-4 | 5e-4 | 1e-4 | 0.2 |

### Architecture Details

**EfficientNet-B3** — torchvision `efficientnet_b3`; classification head replaced with `Dropout(0.3) → Linear(in_features, 7)`. Backbone = `model.features`.

**ViT-B/16** — torchvision `vit_b_16`; head replaced with `Dropout(0.1) → Linear(768, 7)`. Backbone = conv_proj + encoder + class token.

**ConvNeXt-Tiny** — torchvision `convnext_tiny`; `classifier[2]` replaced with `Dropout(0.2) → Linear(768, 7)`.

**DDAMFN (Dual-Direction Attention Mixed Feature Network)** — implemented from scratch in-notebook:

- Stem: `Conv2d(3→32, 3×3, s2) → BN → PReLU`
- 6 × `InvertedResidualCA` blocks: 32→64(s2) → 64 → 128(s2) → 128 → 256(s2) → 256 (expand ratio 2, depthwise 3×3, PReLU)
- Each block contains a **CoordinateAttention** module: dual-direction (H/W) adaptive average pooling, joint embedding via concat, `1×1 conv → BN → Hardswish`, split into H/W attention branches, sigmoid-gated channel recalibration (`identity * a_h * a_w`)
- Head: `Conv2d(256→512, 1×1) → BN → PReLU → AdaptiveAvgPool(1,1) → Flatten → Dropout(0.2) → Linear(512, 7)`
- Only 0.72M parameters — mobile-friendly dedicated FER architecture

### ✅ Pretrained vs. Scratch Answer

- The three benchmarked models (**EfficientNet-B3, ViT-B/16, ConvNeXt-Tiny**) are all **ImageNet-pretrained via transfer learning** (weights `DEFAULT`, downloaded from `download.pytorch.org` during the run).
- **DDAMFN is the only from-scratch architecture** — and it was **disabled in `config.yaml`**, so it was **never trained** ("Skipping ddamfn (disabled in config.yaml)"). The scratch-vs-pretrained comparison is therefore missing from the actual results.

---

## 8. Training Engine — Two-Phase Transfer Learning

- **Phase 1 (epochs 1–3):** backbone frozen; only head trained with AdamW(head_lr) + CosineAnnealingLR
- **Phase 2 (epochs 4–18):** full unfreeze with **discriminative learning rates** (backbone_lr ≪ head_lr); AdamW over two param groups + CosineAnnealingLR
- **Loss:** `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`
- **Regularization:** gradient clipping (1.0), weight decay, dropout; strong weight decay (0.05) for ViT against overfitting
- **Checkpointing:** best model saved by **Val Macro-F1**; early stopping patience 5
- AMP GradScaler/autocast wired in but inactive on CPU
- Progress bars via `tqdm`; per-epoch logs of train loss, val loss, val accuracy, val Macro-F1

---

## 9. Training Results (Per-Model Epoch Trajectories)

### EfficientNet-B3 — 1,628.1 min (~27 h, CPU)

- Phase 1 (frozen head): Val Macro-F1 0.284 → 0.303
- Phase 2: steady climb (0.418 → 0.516 → 0.577 → 0.604) → **best Val Macro-F1 0.6074**, Val Acc 65.29% (epoch 18)

### ViT-B/16 — 2,298.1 min (~38 h, CPU)

- Phase 1: Val Macro-F1 0.381 → 0.424 (strongest head warm-up of all models)
- Phase 2: peaks at **Val Macro-F1 ≈ 0.6598** (epoch 14); mild late overfitting (train loss 0.62 vs. val loss 1.35)

### ConvNeXt-Tiny — 486.3 min (~8 h, CPU)

- Phase 1: 0.373 → 0.396
- Phase 2: fastest convergence → **best Val Macro-F1 0.6650** (epoch 14), Val Acc 70.23%
- **~3.4× faster than EfficientNet-B3 and ~4.7× faster than ViT at better accuracy**

### DDAMFN — **skipped** (disabled in config.yaml)

---

## 10. Final Test Set Evaluation — Official FER2013 Test Split (7,178 samples)

### Benchmark Comparison Table

| Architecture | Test Accuracy | Macro-P | Macro-R | **Macro-F1** | Weighted-F1 | ROC-AUC | Params | Size (MB) | Train time |
|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b3 | 64.77% | 0.6018 | 0.6354 | 0.6113 | 0.6437 | 0.9058 | 10.71M | 41.4 | 1628.1 min |
| vit_b_16 | 69.31% | 0.6475 | 0.6911 | 0.6581 | 0.6941 | 0.9042 | 85.80M | 327.4 | 2298.1 min |
| **convnext_tiny** 🏆 | **69.38%** | 0.6519 | 0.6907 | **0.6657** | 0.6928 | **0.9181** | 27.83M | 106.2 | 486.3 min |

**→ Selected primary model: ConvNeXt-Tiny** (highest Macro-F1: 0.6657). Exported to `results\model_comparison.csv`.

**Dataset scope:** evaluation is on the single dataset (FER2013 official test split); no external dataset was used (`external_dataset.enabled: false`).

### Per-Class Report — ConvNeXt-Tiny (Best Model)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Angry | 0.6030 | 0.6232 | 0.6129 | 958 |
| Disgust | 0.4722 | 0.7658 | 0.5842 | 111 |
| Fear | 0.5740 | 0.5039 | 0.5367 | 1024 |
| Happy | 0.8849 | 0.8884 | 0.8866 | 1774 |
| Sad | 0.5688 | 0.5966 | 0.5824 | 1247 |
| Surprise | 0.7826 | 0.8315 | 0.8063 | 831 |
| Neutral | 0.6781 | 0.6253 | 0.6506 | 1233 |
| **accuracy** | | | **0.6938** | 7178 |
| macro avg | 0.6519 | 0.6907 | 0.6657 | 7178 |
| weighted avg | 0.6943 | 0.6938 | 0.6928 | 7178 |

Notable findings:

- **Disgust recall 0.766** — the class-weighted loss successfully rescued the minority class (at the cost of precision 0.47)
- Weakest class: **Fear (F1 0.537)** — classically confused with Angry/Sad
- Normalized confusion matrix saved: `results\figures\convnext_tiny_confusion_matrix.png`

---

## 11. Error Analysis & Explainability (Grad-CAM)

- Full **GradCAM** class implemented (gradient-weighted class activation mapping for CNN architectures)
- Produced a 4-panel qualitative heatmap figure for the best model (no numeric outputs)
- Note: config `gradcam_target_model` lists `efficientnet_b3`, but Grad-CAM ran in the context of the selected best model

---

## 12. Synthetic Robustness Testing

On ConvNeXt-Tiny (clean accuracy = 69.38%):

| Perturbation | Accuracy | Drop |
|---|---|---|
| Gaussian noise σ=0.02 | 68.15% | −1.23% |
| Gaussian noise σ=0.05 | 64.88% | −4.50% |

Mild, graceful degradation under noise. (Brightness/contrast perturbations were configured in YAML but the executed cell only implements the Gaussian noise test.)

---

## 13. Inference Latency Benchmarking

- **ConvNeXt-Tiny on CPU: 29.89 ± 0.90 ms per image → 33.5 FPS** (batch size 1, 50 iterations after 10 warmup runs)

---

## 14. Model Export (Production Formats)

Exported to `Notebooks\exported_models\convnext_tiny\`:

| Artifact | Details |
|---|---|
| `metadata.json` | Architecture, 7 classes, input shape, ImageNet normalization, test accuracy & Macro-F1 |
| `model_torchscript.pt` | `torch.jit.trace` serialized |
| `model.onnx` | **106.23 MB**, opset 17, dynamic batch axis (`input_faces` → `emotion_logits`) |

---

## 15. Conclusions, Findings & Limitations

### Key Empirical Takeaways

1. **Macro-F1 vs. Accuracy:** optimizing for accuracy alone favors dominant classes (Happy/Neutral/Sad) and collapses recall on Disgust/Fear; cost-sensitive weighting + Macro-F1 checkpoint selection protects minority classes.
2. **CNN vs. ViT on FER2013:** EfficientNet-B3 and ConvNeXt-Tiny show superior inductive bias on low-resolution 48×48 upscaled imagery (faster convergence, lower memory). ViT-B/16 needs strict regularization (WD 0.05, dropout) and is far more expensive (85.8M params, ~38 h CPU training) for comparable accuracy.
3. **Attention (DDAMFN):** Coordinate Attention is designed to highlight localized facial muscle movements (Action Units) — but was not actually trained in this run.

### Dataset Limitations

- ~10–15% label noise from crowdsourced annotation
- 48×48 grayscale native resolution limits micro-expression detection vs. RAF-DB/AffectNet

### Production Readiness Checklist

- [x] High-speed inference (<40 ms; 29.89 ms CPU)
- [x] ONNX & TorchScript serialization
- [ ] Face detection & alignment front-end (e.g., MediaPipe / YOLOv8-Face)
- [ ] Temporal smoothing (EMA over consecutive video frames)

### Caveats

1. Entire run was **CPU-only** — GPU + AMP would substantially reduce training time and likely improve results.
2. **DDAMFN (only scratch model) was never trained** — no scratch-vs-pretrained comparison in results.
3. Only FER2013 was evaluated; external dataset testing is stubbed but disabled.

---

## Appendix — Other Models in the Project (NOT Trained by This Notebook)

The repo contains separately trained models from other pipelines (verified via their metadata files):

| Model | Training | Test Results | Notes |
|---|---|---|---|
| **EfficientNet-B0** | **From scratch** (`pretrained: false`), 224px, AdamW, cosine, 30+20 epochs | **Acc 67.94%, Macro-F1 0.649**, 4.0M params, 15.3 MB, 22.07 ms latency | `models\efficientnet_b0_fer2013.pt`, `results\metrics\efficientnet.json` |
| **DeepFace (Keras)** | Fine-tuned on FER2013, 60 epochs, LR 0.01 | **Acc 55.24%, Macro-F1 0.517**, 30.7 ms latency | `models\deepface_fer2013\`, `results\metrics\deepface.json` |
| **DDAM / DAN** | Pretrained on AffectNet (`affectnet8_epoch5_acc0.6209.pth`), then FER2013-finetuned | Finetuned checkpoint: `dan_fer2013_finetuned_best.pt` (79 MB) | `models\dan_pretrained\` |
| **Deep Attention CNN** | Custom scratch experiments (LR search, cosine scheduler, 30ep+pat5) | Checkpoints under `experiments\exp02\runs\` | `models\deep_attention\` |

---

*End of report.*


