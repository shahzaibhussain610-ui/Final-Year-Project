# FER2013 Deep Attention Network: Notebook Report

## Executive summary

`FER2013_Deep_Attention_Network.ipynb` implements a PyTorch pipeline for seven-class facial-expression classification using a custom convolutional network with channel and spatial attention (CBAM-style). It covers data discovery/loading, exploratory analysis, augmentation, training with class-weighted cross-entropy, validation-based checkpoint selection, test evaluation, error analysis, attention visualization, inference timing, checkpointing, and a single-image predictor.

The saved results report **61.28% test accuracy**, **0.5797 macro-F1**, and **0.6064 weighted F1** on 7,178 images. The best validation-accuracy checkpoint was from epoch 21; training stopped after 23 epochs out of a configured maximum of 60. The model has 1,262,861 trainable parameters and its reported CPU inference time is 61.06 ms per image.

Interpret these results with care: the FER2013 image-folder `test` split contains 7,178 images, exactly twice the commonly reported 3,589-image official test partition. The recorded run does not preserve FER2013's `PublicTest` and `PrivateTest` identities separately. Consequently, these metrics should be described as performance on the supplied test folder, **not as a confirmed PrivateTest-only result**.

## 1. Objective and task

The notebook frames facial-expression recognition as a component of an emotion-aware receptionist robot. Given a pre-cropped face image, it predicts one of seven labels:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

FER2013 images are grayscale and 48 × 48 pixels. The notebook uses the face crops directly; a face detector or camera pipeline is not trained or evaluated here.

## 2. Dataset and split

The YAML configuration points to:

```text
D:\FYP\Datasets\Facial Datasets\FER2013
```

The saved dataset manifest contains 35,887 images:

| Partition used by the notebook | Images | Description |
|---|---:|---|
| Training | 25,838 | 90% stratified subset of the source `train` folder |
| Validation | 2,871 | Remaining 10% stratified subset of the source `train` folder |
| Test | 7,178 | Source `test` folder, evaluated at the end |
| **Total** | **35,887** | |

The source training folder has 28,709 images before the notebook's train/validation split. The test folder is not subdivided by the notebook. Its 7,178-image size matches the combined count of the commonly distributed FER2013 PublicTest and PrivateTest portions, but the saved manifest records only a single `test` label. The notebook therefore cannot establish that the final test metrics are PrivateTest-only.

Class counts in the saved manifest:

| Emotion | Training | Validation | Test |
|---|---:|---:|---:|
| Angry | 3,995 | 467 | 958 |
| Disgust | 436 | 111 | 111 |
| Fear | 4,097 | 1,024 | 1,024 |
| Happy | 7,215 | 1,774 | 1,774 |
| Neutral | 4,965 | 1,233 | 1,233 |
| Sad | 4,830 | 1,247 | 1,247 |
| Surprise | 3,171 | 831 | 831 |

The marked imbalance is important: Happy has far more examples than Disgust, and per-class scores should be read together with support.

## 3. Preprocessing and augmentation

The data pipeline detects the supported source layout (image folders or FER2013 CSV), loads face images, validates/decodes them, encodes emotion labels, and converts inputs to the configured grayscale format and 48 × 48 dimensions. The training transform supports random rotation (up to 10 degrees), horizontal flipping, translation, zoom, brightness, and contrast changes. Augmentation applies to training data, not final evaluation data.

The run uses batch size 64 and zero data-loader workers. The fixed random seed is 42, including seeded data-loader shuffling. This improves repeatability, though exact determinism may still depend on runtime and hardware behavior.

## 4. Model architecture

The primary model is a custom Deep Attention Network with approximately 1.26 million parameters:

1. A grayscale convolutional stem maps the 48 × 48 input into 32 feature channels and downsamples it to 24 × 24.
2. Three residual feature blocks expand the channel widths to 64, 128, and 256 while downsampling spatial dimensions.
3. Each feature block can apply a CBAM-style attention module:
   - **Channel attention** reweights feature channels.
   - **Spatial attention** reweights spatial positions.
4. Adaptive global average pooling aggregates the final feature maps.
5. A fully connected head maps 256 features through a 128-unit hidden layer with ReLU and 0.4 dropout to seven output logits.

Attention can be disabled in the architecture to create a no-attention ablation, but the main recorded configuration has `use_attention: true` and `run_baseline: false`. Therefore, the saved primary evaluation is not itself a controlled attention-versus-no-attention test.

## 5. Training protocol

The primary run's recorded configuration includes:

| Setting | Value |
|---|---|
| Maximum epochs | 60 |
| Epochs recorded | 23 |
| Batch size | 64 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Weight decay | 0.0 |
| Loss | Cross-entropy with inverse-frequency class weights |
| Scheduler | ReduceLROnPlateau on validation loss; factor 0.5, patience 3 |
| Early stopping | Validation loss; patience 5, minimum improvement 0.0001 |
| Checkpoint selection | Highest validation accuracy |
| Best checkpoint epoch | 21 |
| Device | CPU |

The training loop records train/validation loss and accuracy by epoch. Macro metrics are emphasized in evaluation because overall accuracy can obscure weaker performance on underrepresented classes.

## 6. Primary saved evaluation

The values below come from the saved `evaluation_metrics.json` and `classification_report.json`, not from visual estimation of a plot.

| Metric | Result |
|---|---:|
| Test accuracy | 0.6128 (61.28%) |
| Macro precision | 0.5742 |
| Macro recall | 0.6065 |
| Macro F1 | 0.5797 |
| Weighted precision | 0.6170 |
| Weighted recall | 0.6128 |
| Weighted F1 | 0.6064 |
| Test loss | 1.0268 |
| Parameters | 1,262,861 |
| CPU inference time | 61.06 ms/image |

### Per-class test results

| Emotion | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Angry | 958 | 0.4808 | 0.5887 | 0.5293 |
| Disgust | 111 | 0.4398 | 0.6577 | 0.5271 |
| Fear | 1,024 | 0.4723 | 0.2832 | 0.3541 |
| Happy | 1,774 | 0.8788 | 0.8089 | 0.8424 |
| Neutral | 1,233 | 0.5120 | 0.6934 | 0.5890 |
| Sad | 1,247 | 0.5253 | 0.4170 | 0.4649 |
| Surprise | 831 | 0.7103 | 0.7966 | 0.7510 |

Happy and Surprise are recognized most effectively. Fear has the lowest recall (0.2832) and F1 (0.3541), followed by Sad. The six most frequent off-diagonal confusion pairs in the saved test predictions are:

| True emotion | Predicted emotion | Images |
|---|---|---:|
| Sad | Neutral | 322 |
| Fear | Sad | 197 |
| Fear | Angry | 189 |
| Sad | Angry | 182 |
| Fear | Neutral | 170 |
| Angry | Neutral | 149 |

These errors are consistent with the difficulty of separating visually similar expressions in low-resolution images.

## 7. Analysis and visualizations

The notebook generates dataset-level plots (class/pixel distributions, representative samples, augmentation previews), learning curves, correct/incorrect prediction galleries, high-confidence error examples, and attention visualizations. Its confusion-matrix section reports both raw counts and row-normalized percentages (per-class recall). These visualizations complement the metrics; they do not replace evaluation on a clearly isolated test partition.

## 8. Supplementary optimization records

The project also contains an `exp02` learning-rate/scheduler sweep. It records validation results for a reused 0.001 reference checkpoint, newly trained runs at learning rates 0.0005 and 0.002, and a cosine-annealing schedule at 0.0005. Among those records, the 0.0005 ReduceLROnPlateau run had the highest validation accuracy (0.6134) and validation macro-F1 (0.5751).

Those sweep records explicitly mark official test evaluation as not performed. They are validation-only optimization results and must not be combined with, or reported as, additional test-set comparisons. The experiment log also describes the sweep as running with five planned training runs; the saved comparison table contains four records. This report therefore treats it as supplementary/incomplete rather than a finished test benchmark.

## 9. Outputs and inference support

The primary outputs are stored under:

```text
D:\FYP\FER2013_Emotion_Recognition\results\deep_attention
```

They include the trained model checkpoint, training history, evaluation metrics, classification report, predictions, dataset manifest, plots, attention visualizations, and experiment records. The notebook's `EmotionPredictor` class loads a checkpoint, applies the model's preprocessing, and returns a prediction for a single image, providing a starting point for integration.

The reported inference benchmark is a model-level timing on CPU. It does not measure end-to-end camera capture, face detection, image transfer, application logic, or response generation, so it should not be interpreted as a complete real-time robot latency.

## 10. Limitations and recommendations

1. **Clarify test provenance.** The test directory contains 7,178 images, while the standard FER2013 PrivateTest split contains 3,589. Preserve Usage/split identifiers and report PublicTest for validation and PrivateTest for final testing if strict official-split claims are needed.
2. **Check for overlap.** The YAML explicitly notes that pixel-level deduplication and train/test-overlap removal are not implemented. Check duplicate or near-duplicate images across partitions before making benchmark claims.
3. **Run an attention ablation.** Train the same backbone and protocol with attention disabled, using identical splits and seeds, then compare validation-selected checkpoints on the same held-out test set.
4. **Repeat across seeds.** One primary seed does not quantify run-to-run variation. Report mean and standard deviation or confidence intervals across multiple seeds.
5. **Address difficult classes.** Fear and Sad recall/F1 are relatively weak. Review their confusion patterns and evaluate class weighting, focal loss, or targeted augmentation as separate, validation-selected experiments.
6. **Avoid overinterpreting the validation sweep.** Its records are not official-test comparisons and its experiment log indicates it was incomplete.
7. **Validate deployment end-to-end.** Measure actual camera-to-prediction latency and robustness under representative lighting, pose, occlusion, and user populations.

## Conclusion

The notebook provides a substantial end-to-end baseline: a compact residual CNN augmented with channel and spatial attention, class-weighted cross-entropy, early stopping, saved checkpoints, detailed error analysis, and single-image inference support. The saved run achieved 61.28% accuracy and 0.5797 macro-F1 on the supplied 7,178-image test folder. Its strongest class was Happy; Fear and Sad were the principal weaknesses.

The result is useful as a project baseline, but a strict benchmark claim requires preserving the official FER2013 validation/test identities, auditing overlap, and reporting repeated-seed results. The current artifacts do not demonstrate that attention improves performance over an otherwise identical no-attention model.
