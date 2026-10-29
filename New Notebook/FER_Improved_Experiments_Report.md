# FER Improved Experiments Notebook Report

This report summarizes the architecture families, training logic, and measured accuracy from [New Notebook/FER_Improved_Experiments.ipynb](New%20Notebook/FER_Improved_Experiments.ipynb), using the stored outputs in [New Notebook/fer_outputs/selection.json](New%20Notebook/fer_outputs/selection.json), [New Notebook/fer_outputs/run_manifest.json](New%20Notebook/fer_outputs/run_manifest.json), and [New Notebook/fer_outputs/tables/test_results_all_candidates.csv](New%20Notebook/fer_outputs/tables/test_results_all_candidates.csv).

## 1. Purpose of the notebook

The notebook is a FER2013-focused emotion-recognition benchmark built around a single shared experimental workflow:

- dataset loading and split validation from FER2013 CSV or image-folder input,
- recipe tuning for training behavior (augmentation, loss, EMA, label smoothing, sampling),
- architecture search under a consistent recipe,
- final validation-based selection,
- held-out test evaluation, CPU benchmarking, and export decisions.

The configuration in [New Notebook/config.yaml](New%20Notebook/config.yaml) sets a quick profile where the candidate model pool is a mix of CNNs, a ViT, a hybrid CNN-transformer, and a custom attention CNN.

## 2. Model families and architecture structure

### A. CNN backbones (transfer learning)
These models are pretrained and fine-tuned on FER2013 using the same final recipe, with different family-specific learning rates and input sizes.

1. ResNet-50
   - Family: standard convolutional CNN
   - Variant: torchvision ResNet-50 baseline (`tv_resnet50`) and timm ResNet-50 (`resnet50`)
   - Typical structure: convolutional stem, residual bottleneck blocks, global average pooling, final linear classifier
   - Parameter count: about 23.5M
   - Role: strong baseline for comparing with newer architectures

2. ConvNeXt-Tiny
   - Family: modern CNN with depthwise/transformer-inspired design
   - Structure: patchify-style stem, depthwise separable blocks, inverted residual-like stages, layer normalization and improved block design
   - Parameter count: about 27.8M
   - Strength: strong accuracy/speed trade-off in modern vision benchmarks

3. EfficientNet-B0
   - Family: efficient mobile CNN with compound scaling
   - Structure: mobile inverted bottleneck blocks, squeeze-excitation, optimized width/depth scaling
   - Parameter count: about 4.0M
   - Strength: compact and efficient, with good generalization in image classification tasks

4. MobileNetV3-Large 100
   - Family: lightweight CNN designed for speed and deployment
   - Structure: depthwise separable conv blocks, efficient bottlenecks, squeeze-excitation, lightweight head
   - Parameter count: about 4.2M
   - Role: strongest deployment-oriented candidate in this search

### B. Transformer-based model

5. ViT-Small / Patch16 / 224
   - Family: pure vision transformer
   - Structure: image patches embedded into tokens, positional embeddings, multi-head self-attention blocks, MLP feed-forward layers, final classification head
   - Parameter count: about 21.6M
   - Role: strongest single model in the search, designed to capture long-range spatial dependencies between facial regions

### C. Hybrid CNN + Transformer

6. Hybrid CNN-Transformer
   - Family: hybrid attention model
   - Structure: CNN feature extractor (ResNet-18-style stages), then transformer encoder on the resulting feature map or patch tokens
   - Parameter count: about 3.9M
   - Role: tries to combine local texture extraction with global attention

### D. Custom attention CNN (scratch-trained)

7. Custom attention CNN (`custom_attn_cnn`)
   - Family: lightweight custom CNN built from scratch
   - Structure: residual blocks, attention modules (CBAM-like or channel-spatial attention), 48px grayscale input, simple classification head
   - Parameter count: about 1.26M
   - Role: low-parameter, from-scratch experimental baseline intended to analyze attention usefulness and compact-network behavior

The notebook also evaluates an attention ablation variant (`custom_cnn_noattn`) and compares training regimes such as scratch, linear probe, and full fine-tuning. The notebook explicitly states that the architecture differences are compared under one common recipe so the report measures model capacity rather than recipe drift.

## 3. Training philosophy of the notebook

The notebook is not just a single-model training script. It follows a structured training pipeline:

- Stage A: baseline reproduction / report-matching run
- Stage B: ablation of recipe choices (smoothing, augmentations, EMA, sampling strategy, input resolution)
- Stage C: training regime comparison (scratch vs linear probe vs full fine-tuning)
- Stage D: architecture search under one shared recipe
- final selection of the best validation model and final test evaluation

The validation metric used for selection is macro-F1, which is especially useful in FER because class imbalance and underrepresented expressions matter more than raw accuracy alone.

## 4. Accuracy results from the saved experiment outputs

The project output files show the actual measured model performance from the executed experiment pipeline. The strongest result is the ensemble candidate, while the strongest single model is ViT-S/16.

### Best overall candidate

| Candidate | Type | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| ensemble(vit_small_patch16_224, mobilenetv3_large_100) | ensemble | 0.6941 | 0.7072 | 0.6885 |

Interpretation:
- This is the notebook’s selected winner.
- It outperforms every single model on validation and holds the best test accuracy in the saved results.
- Because the selection used validation macro-F1, the ensemble is selected for its stronger class-balanced recognition rather than just overall frequency-weighted accuracy.

### Best single architecture

| Model | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| vit_small_patch16_224 | 0.6704 | 0.6896 | 0.6756 |
| mobilenetv3_large_100 | 0.6459 | 0.6718 | 0.6499 |
| convnext_tiny | 0.6409 | 0.6643 | 0.6463 |
| efficientnet_b0 | 0.6390 | 0.6662 | 0.6480 |
| resnet50 | 0.6336 | 0.6606 | 0.6333 |
| hybrid_cnn_transformer | 0.6177 | 0.6631 | 0.6399 |
| custom_attn_cnn | 0.5664 | 0.6325 | 0.5947 |
| tv_resnet50 (baseline) | 0.6490 | 0.6868 | 0.6635 |

### Accuracy interpretation

- The best single model is ViT-S/16, with test accuracy near 68.96% and test macro-F1 near 67.56%.
- The benchmark baseline `tv_resnet50` remains strong at 68.68% test accuracy and 66.35% macro-F1, which shows the improvement is not huge but still measurable.
- MobileNetV3 and EfficientNet-B0 are close behind, suggesting lightweight models can be competitive.
- The custom attention CNN is lower-performing, but it remains useful as a compact and interpretable scratch-trained model.
- The ensemble’s lead over the best single model is modest but meaningful: around +1.8 percentage points in test accuracy and about +1.3 points in test macro-F1.

## 5. Main conclusions

1. The notebook is a strong architecture-ablation study, not only a single-model classifier.
2. The strongest architecture family is the transformer-based ViT, followed by efficient/mobile CNNs and then the custom attention CNN.
3. The final selected model is the ensemble of ViT-S/16 and MobileNetV3, which gives the highest validation and test metrics in the saved outputs.
4. The report’s design is statistically careful: it compares models under a shared recipe and uses macro-F1 as the primary selection measure.
5. The chosen winner is deployment-relevant because the ensemble contains a highly accurate transformer and a compact CNN; however, the single ViT remains the best standalone model if simplicity matters more than slight performance gain.

## 6. Recommendation

If the aim is a final usable FER pipeline, the notebook strongly suggests one of these two deployment paths:

- Best accuracy: ensemble(ViT-S/16 + MobileNetV3)
- Best single-model trade-off: ViT-S/16
- Best small/efficient model: MobileNetV3-Large or EfficientNet-B0

The saved results are recorded in [New Notebook/fer_outputs/tables/test_results_all_candidates.csv](New%20Notebook/fer_outputs/tables/test_results_all_candidates.csv) and the selected winner is locked in [New Notebook/fer_outputs/selection.json](New%20Notebook/fer_outputs/selection.json).
