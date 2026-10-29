# FER2013 DeepFace-Compatible Emotion Recognition ? Report

**Notebook:** [`FER2013_DeepFace_FER2013.ipynb`](./FER2013_DeepFace_FER2013.ipynb)  
**Configuration:** [`FER2013_DeepFace_FER2013.yaml`](./FER2013_DeepFace_FER2013.yaml)  
**Task:** seven-class facial-expression recognition on FER2013.

## 1. Executive summary

This notebook trains a Keras classifier on FER2013 and provides helpers for still-image inference and DeepFace face extraction. DeepFace is used as an integration framework; it is not the trained FER2013 classifier in this experiment. The notebook attempted to construct DeepFace's VGG-Face backbone, but the saved execution reports that it was unavailable. It therefore used an ImageNet-pretrained ResNet50 backbone, with a custom seven-class classification head.

The saved test evaluation reports **24.71% accuracy**, **5.66% macro F1**, and **approximately 10% weighted F1**. The classification report shows that the model predicts `happy` for every test image: recall is 1.00 for `happy` and 0 for the other six classes. This is a severe class-collapse failure, not a useful FER classifier. The saved training outputs also show validation accuracy fixed at **25.12%** across the displayed epochs. The scores are reported as saved in the notebook and should be treated as this run's recorded outputs, not as a general performance claim.

## 2. Objective and scope

The notebook explores a model for later integration into a camera-based receptionist robot. Its workflow separates:

1. **DeepFace:** face detection/alignment and optional pretrained emotion analysis.
2. **Custom classifier:** a model trained on FER2013 labels.
3. **Inference integration:** apply the custom classifier to an image or a face extracted by DeepFace.

The notebook prepares a conceptual real-time pipeline but does not implement webcam streaming, robot control, or a complete real-time application. The primary dataset inputs are pre-cropped FER2013 face images; face extraction is only used by the optional inference helper.

## 3. Environment and reproducibility

The saved environment output records:

| Item | Recorded value |
|---|---|
| Python | 3.11.9 |
| TensorFlow | 2.21.0 |
| DeepFace | Import succeeded |
| GPU | None detected; CPU execution |
| Random seed | 42 |

The setup checks imports for NumPy, pandas, Matplotlib, Seaborn, OpenCV, scikit-learn, TensorFlow, Keras and DeepFace. The notebook reads settings from the YAML file, seeds Python/NumPy/TensorFlow-related randomness, and writes models and results under the project directories configured there. Exact reproducibility may still depend on package versions, hardware and TensorFlow kernels.

## 4. Dataset and labels

The loader supports either FER2013 CSV files or a folder dataset with `train/<class>` and `test/<class>` directories. The saved execution used the folder layout at the configured FER2013 dataset path.

The inventory contains **35,887 images**:

| Split | Images |
|---|---:|
| Official training partition | 28,709 |
| Validation (stratified from training) | 4,307 |
| Training after validation split | 24,402 |
| Official test partition | 7,178 |

The seven labels are `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. The saved class counts across the full dataset are:

| Class | Images |
|---|---:|
| angry | 4,953 |
| disgust | 547 |
| fear | 5,121 |
| happy | 8,989 |
| neutral | 6,198 |
| sad | 6,077 |
| surprise | 4,002 |

The dataset is imbalanced: `disgust` is the smallest class and `happy` the largest. The CSV path has an explicit FER2013 integer-to-name mapping; for folder data, classes are sorted alphabetically. The official test directory is kept separate. For folder data, 15% of official training images are used as validation with stratification and seed 42.

## 5. Input preprocessing and augmentation

Images are resized to **224 ? 224**, converted to RGB and scaled to floating point in `[0, 1]`. The pipeline then normalizes channels with ImageNet statistics:

- Mean: `[0.485, 0.456, 0.406]`
- Standard deviation: `[0.229, 0.224, 0.225]`

Training augmentation applies random horizontal flip, rotation, zoom, brightness and contrast. Validation and test images are resized and normalized without random augmentation. Tensor batches have shape `(N, 224, 224, 3)`.

The face-folder loader reads with OpenCV, whose native channel order is BGR. The preprocessing helper converts loaded arrays using PIL's RGB conversion; when adapting this pipeline, verify channel ordering and preprocessing consistently between training and inference.

## 6. Backbone and classifier architecture

The intended backbone is DeepFace VGG-Face when `DeepFace.build_model('VGG-Face')` succeeds. In the saved environment this failed with a Keras layer input-state error. The notebook then used:

```text
ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    ? GlobalAveragePooling2D
    ? Dense(256, activation="relu")
    ? Dropout(0.30)
    ? Dense(7, activation="softmax")
```

The recorded model summary reports **24,114,055 total parameters**, **24,060,935 trainable parameters**, and **53,120 non-trainable parameters**. The final layer gives one probability per FER2013 class.

Although the notebook's prose describes a later fine-tuning phase, the executed training code freezes the backbone before compiling and fitting. It does not contain a subsequent unfreeze-and-fine-tune training pass. Also, the saved model summary shows nearly all parameters as trainable, which appears inconsistent with the freeze helper's intended effect. Verify the actual layer trainability and checkpoint when reproducing this experiment; the saved output alone does not establish a clean frozen-backbone training stage.

## 7. Training configuration and procedure

The YAML configuration used by the notebook is:

| Setting | Value |
|---|---:|
| Input size | 224 |
| Batch size | 32 |
| Maximum epochs | 60 |
| Optimizer | Adam |
| Initial learning rate | 0.01 |
| Dropout | 0.30 |
| Validation fraction | 0.15 |
| Early-stopping patience | 5 |
| Number of classes | 7 |
| `run_training` | true |

The model is compiled with sparse categorical cross-entropy and accuracy. Callbacks:

- Save the best checkpoint according to `val_accuracy` (`mode="max"`).
- Stop when `val_loss` fails to improve for the configured patience, restoring best weights.
- Reduce learning rate on a validation-loss plateau by a factor of 0.5, with a minimum rate of `1e-6`.

The saved epoch logs show training accuracy around 24.6?25.1% and validation accuracy of 25.12% in the displayed epochs. Validation loss stays around 1.81, while the learning rate is reduced. This indicates little useful learning. The notebook saves the Keras model and its history when training runs.

## 8. Validation results

Validation is a stratified 4,307-image subset of the official training partition. It is used by the callbacks for learning-rate reduction, early stopping and best-checkpoint selection. It is not the held-out official test set.

The stored epoch output reports **25.12% validation accuracy** for each of the displayed epochs (including the initial epoch). There is no reported validation macro F1 or per-class validation report in the notebook's saved outputs. The validation score is therefore a single accuracy measure and should be interpreted alongside the test classification report and collapsed predictions.

## 9. Held-out test results

The final saved evaluation uses the official test partition of 7,178 images. The notebook reports:

| Test metric | Saved result |
|---|---:|
| Accuracy | **0.2471 (24.71%)** |
| Macro precision | **0.0353 (3.53%)** |
| Macro recall | **0.1429 (14.29%)** |
| Macro F1 | **0.0566 (5.66%)** |
| Weighted F1 | **approximately 0.10 (10%)** |

Per-class test report:

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| angry | 0.00 | 0.00 | 0.00 | 958 |
| disgust | 0.00 | 0.00 | 0.00 | 111 |
| fear | 0.00 | 0.00 | 0.00 | 1,024 |
| happy | 0.25 | 1.00 | 0.40 | 1,774 |
| neutral | 0.00 | 0.00 | 0.00 | 1,233 |
| sad | 0.00 | 0.00 | 0.00 | 1,247 |
| surprise | 0.00 | 0.00 | 0.00 | 831 |

The overall accuracy is close to the test-set share of `happy` (1,774 / 7,178 = 24.7%). The classification report and error examples show that the model predicts `happy` for essentially every sample. Its apparent correct predictions are therefore the `happy` test examples; it fails to recognize the other expressions. The high `happy` recall does not indicate balanced performance.

## 10. Error analysis and interpretation

The confusion matrix and saved examples show the dominant error pattern: actual non-`happy` emotions are classified as `happy`. The example gallery includes angry faces predicted as happy with confidence around 0.266. Correct happy examples have a similar confidence, indicating weak separation rather than strong, calibrated certainty.

Likely concerns visible in the experiment include:

- Severe class collapse despite using an ImageNet-pretrained fallback.
- A high learning rate (`0.01`) for the transfer-learning setup.
- Large imbalance, with no class-weighted loss or balanced sampler in the shown training code.
- CPU-only execution and long epoch times.
- A mismatch between intended freeze/fine-tune narrative and the observed parameter trainability/training code.
- Potential inconsistency between preprocessing/channel order across data loading and inference paths.

These are observations and plausible contributors, not proof of a single root cause. The model should not be used as a working emotion classifier based on these saved results.

## 11. DeepFace inference and optional comparison

`detect_face_and_predict(image_path, model)` attempts to extract and align a face using DeepFace's OpenCV detector, then preprocesses the first extracted face and passes it to the custom Keras classifier. If extraction raises an exception, it falls back to the original image. It returns the predicted emotion, confidence and all seven class probabilities.

`compare_with_deepface_pretrained(image_path)` can call DeepFace's built-in emotion analysis for one image and print the custom classifier output for comparison. This is a qualitative single-image comparison, not a benchmark evaluation over the full test set. The saved notebook does not contain aggregate DeepFace baseline accuracy or F1 values.

`predict_single_image(image_path)` loads the saved Keras checkpoint and predicts a single input image. `real_time_emotion_loop()` only prints a conceptual pipeline; it does not open a camera or perform live inference.

## 12. Saved artifacts

The YAML config sets these output directories relative to the notebook folder:

```text
../models/deepface_fer2013/
../results/deepface_fer2013/
```

The notebook writes or expects:

| Artifact | Purpose |
|---|---|
| `deepface_fer2013_model.keras` | Trained final model. |
| `deepface_fer2013_model_best.keras` | Best checkpoint selected by validation accuracy. |
| `metadata.json` | Class names, config and model/checkpoint paths. |
| `training_history.json` | Per-epoch training and validation loss/accuracy. |
| Confusion matrix plot | Test-set counts by actual and predicted class. |

The notebook's saved output confirms metadata was written at `FER2013_Emotion_Recognition/models/deepface_fer2013/metadata.json`. Check the configured results directory for history and other files from the same run.

## 13. Limitations and recommendations

1. The observed classifier predicts only `happy`, so its test performance is not adequate for deployment.
2. VGG-Face did not load; the actual experiment uses ImageNet ResNet50. Name the backbone accurately in any results table.
3. The notebook does not execute the fine-tuning phase described in prose, and saved trainable parameter counts warrant checking.
4. Class imbalance is not corrected in the shown training loop.
5. CPU-only execution is slow; the saved run's validation scores remain flat.
6. The notebook's single-image DeepFace comparison is not an aggregate baseline study.
7. Face detector fallback may classify a full image if face extraction fails; callers should validate detections explicitly.
8. FER2013 labels are benchmark annotations, not definitive measurements of a person's internal emotional state. Generalization, demographic fairness, calibration and real-world robustness are not established.

Recommended follow-up: verify backbone and trainable layers; ensure consistent RGB/BGR conversion; lower the learning rate; add class-weighted loss or balanced sampling; track per-class validation metrics; confirm labels/order; train and evaluate from a clean run; and report test metrics only from artifacts tied to the matching checkpoint and history. Keep the official test split reserved for the final evaluation.

## 14. Conclusion

The notebook demonstrates a complete prototype path from FER2013 loading and validation splitting to a Keras classifier, checkpointing, test evaluation and DeepFace-assisted still-image inference. In the saved run, however, the fallback ResNet50-based classifier collapses to the majority `happy` prediction, achieving 24.71% test accuracy and 5.66% macro F1, with 25.12% validation accuracy in the displayed training epochs. These results establish that the workflow executes, but they do not establish a useful or deployable emotion-recognition model.
