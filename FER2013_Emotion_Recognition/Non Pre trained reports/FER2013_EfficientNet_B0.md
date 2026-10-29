# EfficientNet-B0 Training and Evaluation Report

**Project:** FER2013 facial emotion recognition  
**Notebook:** `Notebooks/FER2013_EfficientNet_B0.ipynb`  
**Report prepared:** 2026-10-05  
**Evidence used:** saved `training_history.json`, `efficientnet.json`, model metadata, and notebook outputs.

## Executive summary

The EfficientNet-B0 run completed 9 recorded epochs on CPU: four head-only epochs followed by five fine-tuning epochs. The best recorded validation accuracy was **25.35%** (fine_tune, stage epoch 2). Official-test accuracy was **24.98%**, while macro F1 was **6.52%**. The results show severe prediction bias toward `happy`; this checkpoint is not a reliable seven-class emotion classifier.

**Critical training caveat:** the run could not access the ImageNet pretrained weights and initialized EfficientNet-B0 randomly. Although the notebook labels its two stages transfer learning and fine-tuning, this recorded experiment is effectively training from scratch with a frozen, randomly initialized feature extractor in stage one. It should not be reported as a successful pretrained transfer-learning result.

## Dataset and preparation

- Dataset: FER2013 image-folder version at `D:\FYP\Datasets\FER2013`.
- Inventory: 35,887 valid 48?48 grayscale images across seven classes; zero unreadable files; 1,853 exact duplicate files detected. Duplicates were reported, not removed.
- Official split: 28,709 training images and 7,178 test images. A stratified 15% validation split produced 24,402 training and 4,307 validation samples.
- Inputs were converted to RGB and resized/cropped to 224?224. Training used random resized crop, horizontal flip, mild rotation and color jitter; validation and test preprocessing were deterministic.
- Training configuration: seed 42, batch size 32, AdamW, learning rates 0.001 (head) and 0.0001 (fine-tuning), weight decay 0.0001, dropout 0.30, last two feature blocks selected for fine-tuning, patience 3.
- Runtime reported in the notebook: CPU, Python 3.11.9, PyTorch 2.5.1, torchvision 0.20.1.

## Training history

Loss is cross-entropy; accuracy is a fraction of correctly classified images. Times are per epoch from the saved history artifact.

| Stage | Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy | Learning rate | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| transfer_learning | 1 | 1.8532 | 1.8341 | 24.26% | 25.21% | 1.0e-03 | 266.6 |
| transfer_learning | 2 | 1.8624 | 1.8556 | 24.12% | 25.12% | 1.0e-03 | 257.0 |
| transfer_learning | 3 | 1.8629 | 1.8642 | 24.11% | 25.21% | 3.0e-04 | 277.4 |
| transfer_learning | 4 | 1.8388 | 1.8265 | 24.37% | 25.05% | 3.0e-04 | 352.6 |
| fine_tune | 1 | 1.8361 | 1.8218 | 24.29% | 25.24% | 1.0e-04 | 331.2 |
| fine_tune | 2 | 1.8215 | 1.8132 | 24.61% | 25.35% | 1.0e-04 | 275.2 |
| fine_tune | 3 | 1.8124 | 1.8024 | 24.74% | 25.10% | 1.0e-04 | 274.0 |
| fine_tune | 4 | 1.8085 | 1.8050 | 24.83% | 25.24% | 3.0e-05 | 289.9 |
| fine_tune | 5 | 1.8013 | 1.8020 | 25.24% | 25.28% | 3.0e-05 | 313.1 |

Total recorded training time was approximately **43.9 minutes**. Stage one stopped after four epochs under the configured early-stopping rule. Fine-tuning ran all five configured epochs. The best validation accuracy (25.35%) occurred at fine-tuning epoch 2; training accuracy remained near 25%, indicating little useful learning.

## Official test-set results

| Metric | Result |
|---|---:|
| Accuracy | 24.98% |
| Macro precision | 27.17% |
| Macro recall | 14.60% |
| Macro F1 | 6.52% |
| Weighted F1 | 10.72% |
| Parameters | 4,016,515 |
| Model file size | 15.32 MB |
| Test inference time | 45.59 seconds total; 6.35 ms/image |

### Per-class report

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 50.00% | 0.10% | 0.21% | 958 |
| disgust | 0.00% | 0.00% | 0.00% | 111 |
| fear | 25.00% | 0.10% | 0.19% | 1,024 |
| happy | 24.91% | 99.21% | 39.82% | 1,774 |
| neutral | 17.65% | 0.24% | 0.48% | 1,233 |
| sad | 28.17% | 1.60% | 3.03% | 1,247 |
| surprise | 44.44% | 0.96% | 1.88% | 831 |

The `happy` test class accounts for 1,774 of 7,178 images (24.7%). The model recalled 99.2% of happy samples but recalled at most 1.6% of any other class. Overall accuracy (24.98%) is only slightly above the always-`happy` baseline (24.70%), and macro F1 (6.52%) confirms poor performance across classes.

## Interpretation and limitations

1. **No pretrained initialization:** the weight download failed and no local checkpoint was available. Stage one therefore trained the new classifier on top of random features; stage two only unfroze two feature blocks.
2. **Class collapse:** the near-majority-class accuracy and near-zero recall for six emotions indicate the model did not learn useful class separation.
3. **Do not use this checkpoint as a result demonstrating effective transfer learning or as a deployment model.** The metrics are valid for this saved checkpoint and split, but they do not establish the performance expected from pretrained EfficientNet-B0.
4. **Notebook output is incomplete:** the saved training-cell output ends partway through fine-tuning epoch 3 and its execution count is unset. The report?s nine-epoch history and test metrics come from the complete JSON artifacts in `results/metrics/`; the notebook output alone does not show that full run.
5. Exact duplicates remain in the official dataset and were not removed. The official test partition was kept separate for the stratified training/validation split.

## Recommended next run

- Ensure the torchvision EfficientNet-B0 ImageNet weights are cached or accessible before training; confirm the notebook reports pretrained weights loaded.
- Train again with the same fixed split and save a fresh checkpoint and history so the pretrained run is clearly distinguishable from this from-scratch run.
- Review the confusion matrix and class-wise recall; consider training-only class weighting or a sampler if imbalance remains a problem.
- Preserve the official test set for final evaluation and compare the new run against this baseline using macro F1 and per-class recall as well as accuracy.

## Artifacts

- Training history: [`results/metrics/training_history.json`](../results/metrics/training_history.json) and [`training_history.csv`](../results/metrics/training_history.csv).
- Test metrics: [`results/metrics/efficientnet.json`](../results/metrics/efficientnet.json).
- Confusion matrices: [`efficientnet_raw.png`](../results/metrics/efficientnet_raw.png) and [`efficientnet_normalized.png`](../results/metrics/efficientnet_normalized.png).
- Training curves: [`results/plots/training_curves.png`](../results/plots/training_curves.png).
- Saved checkpoint: [`models/efficientnet_b0_fer2013.pt`](../models/efficientnet_b0_fer2013.pt).
