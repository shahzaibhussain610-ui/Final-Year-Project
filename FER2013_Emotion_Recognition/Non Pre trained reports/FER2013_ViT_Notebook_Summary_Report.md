# FER2013 ViT Notebook Summary Report

## Project context

This report summarizes the content of the FER2013 Vision Transformer notebook (`Notebooks/FER2013_ViT.ipynb`). The notebook is designed to build a complete FER2013 emotion-classification pipeline using a ViT-B/16 model from `torchvision`, with a focus on reproducibility, dataset validation, and research-style model evaluation.

The notebook follows a standard experimental workflow:

- dataset inspection and validation
- image-level feature analysis
- official split preservation
- stratified validation split creation
- model construction and training
- evaluation using classification metrics and confusion matrices
- artifact saving for model and outputs

## Objective of the notebook

The notebook aims to classify seven FER2013 emotions:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The model uses a Vision Transformer backbone (`vit_b_16`) trained on FER2013 images. The pipeline is set up so that the official test set remains untouched, while the training data is split into train/validation subsets for model selection and monitoring.

## Data and configuration overview

The notebook sets the dataset path to:

- `D:\FYP\Datasets\Facial Datasets\FER2013`

The configuration object defines a reproducible experimental setup with:

- random seed: 42
- image size: 224x224
- validation fraction: 15%
- batch size: 8 on CPU, 32 on GPU
- number of epochs: 20
- learning rate: 1e-4
- weight decay: 1e-4
- dropout: 0.1
- patience: 4
- model name: `vit_b_16_fer2013`

The environment is also configured to automatically use CUDA if available; otherwise it runs on CPU.

## Dataset inspection process

A custom `inspect_dataset()` function scans both the official `train` and `test` folders and records:

- valid image files
- invalid/corrupt images
- exact duplicate images
- class labels
- image dimensions
- image color mode

The function reads each image with PIL, verifies that it can be opened successfully, and computes a SHA-256 hash to detect duplicates. It stores the result in a `Inventory` dataclass that includes a pandas DataFrame, duplicate list, invalid list, class list, and counters for dimensions and modes.

## Key dataset findings from the notebook outputs

The notebook reports the following values from the FER2013 dataset inventory:

- total valid image samples: 35,887
- classes: `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`
- invalid images: 0
- duplicate images: 1,853
- all images are grayscale: `{'L': 35887}`
- all images are 48x48 pixels: `{(48, 48): 35887}`

Training data class distribution (official training set) is highly imbalanced:

- angry: 3,995
- disgust: 436
- fear: 4,097
- happy: 7,215
- neutral: 4,965
- sad: 4,830
- surprise: 3,171

This imbalance is a major characteristic of the FER2013 dataset and is likely to affect model learning, especially for underrepresented classes like `disgust`.

## Model and training design

The notebook uses a Vision Transformer architecture (`vit_b_16`) from the `torchvision` model zoo. It is configured to:

- resize and normalize inputs for ViT-compatible preprocessing
- preserve the official test split
- generate a validation split from the training data using a reproducible random process
- train the model with a standard cross-entropy objective
- evaluate on validation and test sets using accuracy and classification metrics
- save outputs to structured result directories

The project also includes a `RUN_TRAINING` flag, which allows the notebook to be explored without immediately launching a long training run. This is useful for experimentation, debugging, and reproducibility.

## Strengths of the notebook

1. Reproducible setup
   - Fixed seed values
   - explicit dataset paths
   - saved configuration metadata
   - deterministic train/validation split logic

2. Strong data auditing
   - duplicate checking
   - invalid image detection
   - class distribution reporting
   - image mode and dimension validation

3. Research-oriented structure
   - separate stages for dataset inspection, training, validation, and evaluation
   - use of metrics and confusion matrices
   - artifact generation for downstream reporting and comparison

4. Proper dataset handling discipline
   - official test set kept intact
   - validation split derived only from training data

## Challenges and risks

1. Dataset imbalance
   - `happy` is significantly overrepresented
   - `disgust` is much less frequent than the rest
   - this may lead to biased learning and low recall for minority classes

2. Duplicate images remain in the dataset
   - exact duplicates are identified but not removed
   - duplicates can inflate confidence in the model or distort performance estimates if not carefully managed

3. Training from scratch
   - the notebook description explicitly states the model is trained without pretrained weights
   - this can reduce performance, especially for a data-hungry architecture like ViT

4. Real-world FER2013 difficulty
   - FER2013 is a challenging and noisy dataset with grayscale, low-resolution, and often ambiguous facial expressions

## Expected outputs and artifacts

The notebook is designed to produce saved outputs under the project’s result folders, including:

- dataset-level reports
- class distribution plots
- confusion matrices
- training curves or loss/accuracy history
- evaluation metrics
- model checkpoints

The notebook creates outputs in:

- `results/vit/`
- `models/vit/`

These artifacts support later comparison with other FER2013 pipelines, such as EfficientNet and DeepFace baselines.

## Conclusion

The FER2013 ViT notebook is a complete experimental pipeline for facial emotion recognition using a transformer-based architecture. Its main contribution is not just the model training itself, but the careful dataset auditing and reproducible experimentation setup that makes the results interpretable and research-ready.

The notebook clearly documents the FER2013 inventory, identifies the dataset’s key issues (class imbalance and duplicates), preserves the official test partition, and structures the work in a way that supports fair comparison with other models.

Overall, this notebook is a strong foundation for a ViT-based FER2013 classifier, but its performance will likely be sensitive to the dataset’s imbalance and the fact that the model is trained from scratch rather than initialized with pretrained ImageNet weights.
