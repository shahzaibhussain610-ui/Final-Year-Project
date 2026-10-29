# Multimodal Emotion Recognition & FER 95%+ Deep Learning System
### Final Year Project (FYP) — Computer Science & Artificial Intelligence

An advanced, end-to-end framework combining **Multimodal Emotion Recognition (Facial + Audio/Voice)** and a **High-Performance 95%+ Facial Emotion Recognition (FER)** pipeline designed for Human-Robot Interaction (HRI) and real-time affective computing.

---

## 📌 Project Overview

This repository contains the complete research, deep learning architectures, experimental pipelines, reports, and interactive Jupyter notebooks for:
1. **Multimodal Emotion Recognition**: Combining visual facial features (ViT, ResNet-50, EfficientNet) and audio spectrogram features (CREMA-D, RAVDESS).
2. **FER 95%+ Master Benchmark Pipeline**: Two-stage curriculum learning regime (Stage 1 broad pretraining on AffectNet+ & FER2013, Stage 2 target finetuning on CK+ & RAVDESS) achieving state-of-the-art benchmark accuracy.
3. **Deep Attention Networks (DAN)**: Multi-head cross-attention for subtle facial action units (AUs).
4. **HRI Deployment Ready**: Inference pipelines with CLAHE illumination correction, MTCNN/YuNet face detection, and latency-optimized inference.

---

## 📂 Repository Structure

```
FYP/
├── Emotion Detection through Voice and Face/ # Multimodal audio + visual experiments
│   ├── config.yaml
│   └── facial_emotion_experiments.ipynb
├── FER 95 +/                                 # 95%+ Accuracy target pipeline & 2D CNN
│   ├── FER_95Plus_HRI_Pipeline.ipynb
│   ├── Simple_2DCNN_FER.ipynb
│   └── config.yaml
├── FER2013_Emotion_Recognition/             # Baseline & transfer learning models
│   ├── Notebooks/
│   ├── Notebooks Pre-Trained Models/
│   └── requirements.txt
├── New Notebook/                            # Research ablation experiments & recipes
│   ├── Facial Emotion Research/
│   ├── fer_outputs/
│   ├── fer_research/
│   └── FER_Improved_Experiments.ipynb
├── New Notebook2/                           # GPU-accelerated master benchmark pipeline
│   ├── FER_95Plus_Master_Pipeline.ipynb
│   ├── NewNotebook2.ipynb
│   └── config_fer_95plus.yaml
├── Voice and Facial recognition/            # Voice + Face pipeline & multimodal models
│   ├── config.yaml
│   └── notebook.ipynb
├── Reports/                                 # 15+ Comprehensive Technical Research Reports
│   ├── FER_95Plus_Master_Pipeline_Report.md
│   ├── Emotion_Datasets_Comprehensive_Report.md
│   ├── FER_GPU_90Plus_Benchmark_Report.md
│   └── ...
├── requirements.txt                         # Global Python dependencies
└── .gitignore                               # Standard ML ignore rules (datasets/weights)
```

---

## 🔬 Benchmark Datasets Supported

- **Facial Vision**:
  - **AffectNet+**: Wild in-the-wild facial expressions with challenging poses and lighting.
  - **FER2013**: Real-world emotion benchmark with 35,887 images across 7 classes.
  - **CK+ (Extended Cohn-Kanade)**: Lab-controlled apex emotion expressions.
- **Vocal & Audiovisual**:
  - **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song.
  - **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset.

> *Note: Raw datasets and large trained binary weight checkpoints (`.pt`, `.pth`, `.keras`, `.onnx` > 100MB) are excluded from Git version control per GitHub storage constraints. Place datasets into the `Datasets/` directory locally.*

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/shahzaibhussain610-ui/Final-Year-Project.git
cd Final-Year-Project
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Running the Pipelines
Launch Jupyter Lab or Notebook:
```bash
jupyter lab
```
Open any of the core notebooks:
- `New Notebook2/FER_95Plus_Master_Pipeline.ipynb` — SOTA 95%+ FER Master Pipeline.
- `FER 95 +/FER_95Plus_HRI_Pipeline.ipynb` — Real-time HRI Emotion Recognition Pipeline.
- `Voice and Facial recognition/notebook.ipynb` — Multimodal Audio + Visual Emotion Fusion.

---

## 📊 Comprehensive Technical Reports

Detailed experiment logs, confusion matrices, ablation tables, and loss curves are documented in the [`Reports/`](Reports/) directory.

---

## 👤 Author

- **Shahzaib Hussain**
- GitHub: [@shahzaibhussain610-ui](https://github.com/shahzaibhussain610-ui)

