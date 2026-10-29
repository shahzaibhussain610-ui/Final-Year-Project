# FER2013 DeepFace-Compatible Training Notebook Summary Report

## Project context

This report summarizes the content of the DeepFace-compatible FER2013 notebook (`Notebooks/FER2013_DeepFace_FER2013.ipynb`). The notebook is designed around an emotion-recognition pipeline for the FER2013 dataset and is intended to support a future live-camera emotion-aware application, such as a receptionist robot.

The notebook explicitly distinguishes between three roles:

- DeepFace for face detection, preprocessing, and alignment
- a custom FER2013-trained classifier for emotion recognition
- a DeepFace-compatible inference workflow for deployment-style use

This is important because DeepFace is a face-analysis framework, not a direct FER2013 fine-tuning solution in the same way that a standard Keras model pipeline is.

## Objective of the notebook

The notebook trains a FER2013 emotion-classification model capable of predicting seven expressions:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

The overall aim is to create a reusable model that can later be integrated into a real-time or camera-based inference system while still keeping the dataset pipeline, evaluation, and training logic organized and reproducible.

## Environment and setup

The notebook first verifies that required libraries are installed and available in the Python environment. The reported environment includes:

- Python: 3.11.9
- TensorFlow: 2.21.0
- Keras: available
- DeepFace: available
- OpenCV: available
- NumPy, pandas, seaborn, scikit-learn: available

It also checks for GPU availability, and in this run the output reported:

- GPU available: []

This means the workload ran on CPU, which is important because training time and training quality can be affected by hardware constraints.

## Configuration summary

The notebook defines a single `Config` dataclass for repeatable experimentation. The key settings are:

- dataset path: `D:\FYP\Datasets\Facial Datasets\FER2013`
- seed: 42
- image size: 224x224
- batch size: 32
- epochs: 20
- learning rate: 1e-4
- weight decay: 1e-4
- dropout: 0.3
- patience: 5
- validation fraction: 0.15
- num classes: 7
- model name: `deepface_fer2013_model`

The notebook uses standard Keras callbacks for model selection and regularization:

- `ModelCheckpoint` to save the best validation model
- `EarlyStopping` to stop training when validation loss stops improving
- `ReduceLROnPlateau` to lower learning rate when progress stalls

## Dataset handling and preprocessing

The notebook is explicit that FER2013 may appear in one of two common formats:

- a CSV annotation file
- a class-directory image dataset

It includes logic to handle both cases and reports the actual data source found. The dataset is then converted into a consistent format for model training.

The notebook also sets up output directories for:

- results: `results/deepface_fer2013`
- model checkpoints: `models/deepface_fer2013`

This indicates the project expects structured saving of training history, best weights, and finalized model artifacts.

## Model architecture and training approach

The notebook builds a Keras-based emotion model using standard deep-learning image backbones. The source code includes imports for:

- `ResNet50`
- `VGG16`
- `keras.layers`
- `keras.models`

This suggests a transfer-learning style or feature-extraction approach built on well-known convolutional backbones, with a custom classification head for seven emotion classes. The model is designed to process 224x224 RGB inputs and output a 7-class emotion prediction.

Training is performed with:

- training/validation split from the dataset
- `categorical_crossentropy`-style classification objective
- validation monitoring to select the best checkpoint
- early stopping to prevent overfitting or wasted epochs

## Training behavior observed in the notebook output

The notebook output shows a full 20-epoch run on CPU. The recorded training progression was:

- Epoch 1: training accuracy 0.2283, validation accuracy 0.2526
- Epoch 2: training accuracy 0.2587, validation accuracy 0.2853
- Epoch 3: training accuracy 0.2689, validation accuracy 0.2953
- Epoch 4: training accuracy 0.2774, validation accuracy 0.3146
- Epoch 5: training accuracy 0.2853, validation accuracy 0.3141
- Epoch 6: training accuracy 0.2885, validation accuracy 0.3162
- Epoch 7: training accuracy 0.2919, validation accuracy 0.3332
- Epoch 8: training accuracy 0.2967, validation accuracy 0.3350
- Epoch 9: training accuracy 0.2992, validation accuracy 0.3316
- Epoch 10: training accuracy 0.3033, validation accuracy 0.3278
- Epoch 11: training accuracy 0.3091, validation accuracy 0.3501
- Epoch 12: training accuracy 0.3100, validation accuracy 0.3548
- Epoch 13: training accuracy 0.3101, validation accuracy 0.3513
- Epoch 14: training accuracy 0.3112, validation accuracy 0.3520
- Epoch 15: training accuracy 0.3151, validation accuracy 0.3536
- Epoch 16: training accuracy 0.3133, validation accuracy 0.3545
- Epoch 17: training accuracy 0.3153, validation accuracy 0.3636
- Epoch 18: training accuracy 0.3181, validation accuracy 0.3606
- Epoch 19: training accuracy 0.3164, validation accuracy 0.3585
- Epoch 20: training accuracy 0.3184, validation accuracy 0.3580

The best validation accuracy observed was approximately:

- 0.3636 (Epoch 17)

This is a meaningful improvement over the early epochs, but still modest overall. The model improves steadily during the middle training phase and then plateaus, which is consistent with a difficult dataset and a constrained training setup.

## Key interpretation

This notebook demonstrates a workable DeepFace-compatible FER2013 training pipeline, but the results indicate that the model is still limited in performance.

The most important observations are:

1. Training is happening on CPU only
   - this significantly increases runtime and may limit optimization quality

2. Validation accuracy improves gradually but remains moderate
   - best validation accuracy is around 36%
   - this indicates the model does not yet reach strong emotion-recognition performance

3. The dataset is difficult and imbalanced
   - FER2013 is known to have noisy labels and class imbalance
   - some classes are much less frequent than others, which can reduce recall for minority emotions

4. The model is not yet a production-ready emotion classifier
   - it is a strong experimental baseline and a solid foundation for further tuning

## Strengths of the notebook

1. Clear research framing
   - the notebook explains the role of DeepFace separately from the custom FER model
   - this avoids confusion between preprocessing and classification

2. Reproducible experimental setup
   - configurable seed, batch size, learning rate, and epochs
   - standard callbacks for checkpointing and early stopping

3. Realistic deployment-minded design
   - the project is oriented toward future live-camera integration instead of just offline evaluation

4. Good artifact handling
   - model and training artifacts are saved in structured directories
   - this supports later comparison, deployment, and reporting

## Challenges and limitations

1. CPU-only execution
   - this slows training dramatically and limits how aggressively the model can be tuned

2. Moderate validation performance
   - around 36% validation accuracy suggests a nontrivial classification challenge

3. Potential class imbalance
   - the FER2013 dataset is not balanced across emotion categories

4. No guarantee of strong real-time deployment yet
   - the notebook is useful as a foundation, but further improvements are needed before it is a robust system for live emotion recognition

## Outputs and artifacts

The notebook saves its training artifacts to the project structure:

- `models/deepface_fer2013/deepface_fer2013_model.keras`
- `models/deepface_fer2013/deepface_fer2013_model_best.keras`
- `results/deepface_fer2013/training_history.json`

The recorded training history shows an upward trend in validation accuracy, with the best model appearing near the end of the training process.

## Conclusion

The FER2013 DeepFace-compatible notebook is a comprehensive and well-structured experiment for emotion recognition using a custom Keras model in a DeepFace-aware workflow. It is useful for both research and practical system integration, especially because it clearly separates preprocessing, model training, and deployment-oriented inference.

The training output demonstrates that the model learns meaningful patterns over time, but it remains a modest-performing baseline rather than a high-accuracy emotion-recognition system. The results are reasonable for an exploratory FER2013 pipeline, especially under CPU-only training and with a challenging, imbalanced dataset.

Overall, this notebook is a solid foundation for future optimization, including stronger backbone tuning, better handling of class imbalance, and potentially GPU-based training for improved performance.
