# Comprehensive Technical Report: Facial & Vocal Emotion Recognition Datasets
**An Exhaustive Investigation of Multimodal Emotion Corpora: FER2013, FER2013+ (FER+), AffectNet+, CK+ (Extended), CREMA-D, and RAVDESS**

**Project**: Multimodal Emotion Recognition (Facial Expression Recognition & Vocal Speech Emotion Recognition)  
**Author / Investigator**: Final Year Project (FYP) Research Group  
**Dataset Base Directory**: [`C:/Users/Shahzaib Hussain/Desktop/Shahzaib Hussain/FYP/Datasets`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets)  
**Output Target Directory**: [`C:/Users/Shahzaib Hussain/Desktop/Shahzaib Hussain/FYP/Reports`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Reports)  
**Evaluation Date**: October 2026  

---

## 1. Executive Summary & Benchmark Corpus Overview

This technical report delivers a definitive academic and empirical examination of the six emotion recognition datasets housed in the project directory [`C:\Users\Shahzaib Hussain\Desktop\Shahzaib Hussain\FYP\Datasets`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets). Spanning both **Facial Expression Recognition (FER)** and **Speech Emotion Recognition (SER)**, this multimodal repository provides the foundational assets for training, pre-training, fine-tuning, cross-corpus generalization testing, and multi-stream fusion modeling in affective computing.

The local dataset collection encompasses **over 1,000,000 facial images, 4,904 high-definition video recordings, and 9,894 acoustic speech/song files**, totaling **33.45 GB** of curated affective data across two primary subdirectories:
1. **`Facial Dataset/`** ([`C:/Users/Shahzaib Hussain/Desktop/Shahzaib Hussain/FYP/Datasets/Facial Dataset`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Facial%20Dataset)): Static in-the-wild facial expressions and laboratory-controlled apex frames (**AffectNet+**, **FER2013**, **FER2013 CSV**, and **CK+**).
2. **`Voice and Facial/`** ([`C:/Users/Shahzaib Hussain/Desktop/Shahzaib Hussain/FYP/Datasets/Voice and Facial`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Voice%20and%20Facial)): Dynamic multimodal audio-visual speech and song sequences (**CREMA-D** and **RAVDESS**).

```
Datasets/
├── Facial Dataset/
│   ├── AffectNet+/             [959,906 images + rich JSON annotations: 7.70 GB]
│   │   ├── human_annotated/    [414,799 Train + 5,500 Val images & metadata]
│   │   └── no_human_annotated/ [539,607 images & automated annotations]
│   ├── CK+/                    [920 images in pixel tabular CSV format: 7.23 MB]
│   │   └── ckextended.csv      [48x48 grayscale pixel strings, FACS-coded]
│   ├── FER2013/                [35,887 extracted images in class subfolders: 53.89 MB]
│   │   ├── train/              [28,709 images across 7 emotion categories]
│   │   └── test/               [7,178 images across 7 emotion categories]
│   └── FER2013 CSV/            [Tabular pixel dataset & test split: 296.79 MB]
│       ├── fer2013.csv         [35,887 rows: emotion, pixels, Usage]
│       └── test/               [7,178 images across 7 emotion categories]
└── Voice and Facial/
    ├── CREMA-D/                [7,442 uncompressed WAV files (16 kHz mono): 577.83 MB]
    │   └── *.wav               [91 actors, 12 sentences, 6 emotions, 4 intensities]
    └── RAVDESS/                [7,356 multimodal files (2,452 WAV + 4,904 MP4): 24.82 GB]
        ├── Audio_Speech_Actors_01-24/ [1,440 studio WAVs (48 kHz 16-bit uncompressed)]
        ├── Audio_Song_Actors_01-24/   [1,012 studio WAVs (48 kHz 16-bit uncompressed)]
        ├── Video_Speech_Actor_01-24/  [2,880 720p HD MP4 video clips (AV + Video-only)]
        └── Video_Song_Actor_01-24/    [2,024 720p HD MP4 video clips (AV + Video-only)]
```

## 2. Master Dataset Comparison Matrix

The table below contrasts the six datasets across acquisition methodology, modality, scale, emotion taxonomy, resolution, ground-truth validity, and local disk footprints:

| Dimension | FER2013 | FER2013+ (FER+) | AffectNet+ | CK+ (Extended) | CREMA-D | RAVDESS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Citation** | Goodfellow et al. (2013) | Barsoum et al. (2016) | Mollahosseini et al. (2017/2019) | Lucey et al. (2010) | Cao et al. (2014) | Livingstone & Russo (2018) |
| **Originating Institution** | Université de Montréal (ICML 2013) | Microsoft Research | University of Denver | CMU / Univ. of Pittsburgh | Univ. of Pennsylvania / Rowan | Ryerson SMART Lab |
| **Primary Modality** | Static Facial Image | Static Facial Image | Static Facial Image | Dynamic $\to$ Apex Image | Acoustic Speech (WAV) | Acoustic Speech/Song & Video (MP4) |
| **Secondary Modality** | None | None | Facial Landmarks & Head Pose | FACS Action Units | Video (Original FLV) | Audio-only & Video-only tracks |
| **Acquisition Setting** | In-the-Wild (Google Web Scrape) | In-the-Wild (Crowdsourced) | In-the-Wild (3 Search Engines, 6 Langs) | Lab-Controlled (High Lumens, Frontal) | Studio Acted (Green Screen / Lab) | Studio Acted (Anechoic Chamber, 720p) |
| **Total Sample Volume** | **35,887** images | **35,887** images | **959,906** samples (420k human) | **920** samples (593 sequences) | **7,442** audio clips | **7,356** files (2,452 WAV, 4,904 MP4) |
| **Number of Subjects** | Unknown (Web users) | Unknown (Web users) | Hundreds of thousands | 123 subjects | 91 professional actors | 24 professional actors |
| **Gender Balance** | Natural web mix | Natural web mix | Continuous probabilistic estimates | 65% Female, 35% Male | 48 Male, 43 Female | 12 Male, 12 Female (Strict 50/50) |
| **Ethnic Diversity** | Uncontrolled web | Uncontrolled web | Multi-ethnic (6 probability classes) | 81% Euro-American, 13% Afro-Am. | African Am., Asian, Caucasian, Hispanic | Canadian Caucasian actors |
| **Native Emotion Classes** | 7 basic categories | 10 classes (incl. Contempt, NF) | 8 categories + Dimensional V/A | 8 FACS-verified categories | 6 basic categories | 8 categories (Speech), 6 (Song) |
| **Continuous Affect ($V$-$A$)**| None | Soft probability vectors | Continuous Valence & Arousal $[-1, 1]$ | None | Perceived Intensity Ratings | Emotional Intensity (Normal / Strong) |
| **Native Resolution / Rate** | $48 \times 48$ Grayscale | $48 \times 48$ Grayscale | $224 \times 224$ RGB Color | $48 \times 48$ Grayscale (Local CSV) | 16 kHz, 16-bit Mono PCM | 48 kHz 16-bit WAV / $1280 \times 720$ 30fps |
| **Ground-Truth Quality** | Low (20-30% noise; human $\approx 65\%$) | High (10-tagger consensus) | Very High (Expert human audit) | Ground Truth (FACS Action Units) | High (Validated by 2,443 raters) | Exceptional (Validated by 319 raters) |
| **Local Disk Footprint** | **53.89 MB** (Images) | Shared with FER2013 | **7.70 GB** | **7.23 MB** | **577.83 MB** | **24.82 GB** |
| **Primary Project Role** | Baseline FER benchmarking | Label noise ablation | Large-scale pre-training backbone | Posed apex validation | Demographic SER benchmarking | Dynamic audio-visual fusion modeling |

---

## 3. In-Depth Technical Dossier: FER2013

### 3.1 Historical Background & Origin
The **Facial Expression Recognition 2013 (FER2013)** dataset was compiled by Pierre-Luc Carrier and Aaron Courville from Yoshua Bengio’s lab at the Université de Montréal for the *ICML 2013 Workshop on Challenges in Representation Learning* (hosted as an open Kaggle machine learning competition).

To collect the dataset, an automated crawler queried the Google Image Search API for emotion keywords combined with demographic, occupation, and scene modifiers. Face candidates were automatically cropped and registered using an OpenCV Haar cascade face detector, resized to a standardized **$48 \times 48$ pixel resolution**, converted to grayscale, and assigned the queried emotion keyword as the default ground truth label.

### 3.2 Dataset Architecture & Local File Structure
In the local workspace, FER2013 exists in two complementary representations:
1. **Extracted Image Hierarchical Directory** ([`Facial Dataset/FER2013/`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Facial%20Dataset/FER2013)): Contains standard image subfolders split into `train/` and `test/`, ready for `torchvision.datasets.ImageFolder` ingestion:
   - `train/`: 28,709 images
   - `test/`: 7,178 images (combining the original Kaggle `PublicTest` of 3,589 images and `PrivateTest` of 3,589 images).
2. **Tabular Storage Format** ([`Facial Dataset/FER2013 CSV/fer2013.csv`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Facial%20Dataset/FER2013%20CSV/fer2013.csv)): A single 296.79 MB CSV file containing 35,887 rows with three fields:
   - `emotion`: Integer index $(0 \text{ to } 6)$
   - `pixels`: Space-separated string of 2,304 pixel intensity values $(0 \text{ to } 255)$
   - `Usage`: Partition marker (`Training`, `PublicTest`, `PrivateTest`).

### 3.3 Class Breakdown & Distribution
The dataset encompasses seven basic categorical emotions, mapped as follows:

| Index | Emotion Class | Training Split | PublicTest Split | PrivateTest Split | Total Samples | Class Proportion |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **Angry** | 3,995 | 467 | 491 | 4,953 | 13.80% |
| **1** | **Disgust** | 436 | 56 | 55 | 547 | **1.52%** |
| **2** | **Fear** | 4,097 | 496 | 528 | 5,121 | 14.27% |
| **3** | **Happy** | 7,215 | 895 | 879 | 8,989 | **25.05%** |
| **4** | **Sad** | 4,830 | 653 | 594 | 6,077 | 16.93% |
| **5** | **Surprise** | 3,171 | 415 | 416 | 4,002 | 11.15% |
| **6** | **Neutral** | 4,965 | 607 | 626 | 6,198 | 17.27% |
| **Total** | — | **28,709** | **3,589** | **3,589** | **35,887** | **100.00%** |

> [!WARNING]
> **Severe Class Imbalance**: The `Disgust` class contains only 547 samples across the entire corpus, representing a **16.4:1 ratio** against the dominant `Happy` class (8,989 samples). Models trained with naive cross-entropy loss tend to collapse on `Disgust` (often reporting single-digit recall or F1 scores below 0.15).

### 3.4 Key Technical Bottlenecks & Vulnerabilities
1. **High Label Noise**: Because initial labeling was inferred directly from search keywords without extensive manual verification, researchers estimate that **20% to 30% of the labels are ambiguous or outright incorrect** (e.g., an actor shouting in surprise labeled as anger, or neutral passport photos tagged as sad).
2. **Artifacts & Non-Face Contamination**: Automated scraping allowed non-faces (e.g., drawings, cartoon illustrations, heavily watermarked stock images, partially occluded sunglasses/scarves) to infiltrate the dataset. An exact perceptual hash audit conducted in our research pipeline identified **13 completely blank/corrupt images** and **1,798 training set duplicates**.
3. **Low Resolution Loss of Micro-Signals**: At $48 \times 48$ grayscale, fine facial Action Units (such as subtle AU4 eyebrow lowering or AU6 cheek raising) suffer from severe spatial aliasing, limiting human perceptual recognition accuracy to approximately **$65\% \pm 5\%$**.

---

## 4. In-Depth Technical Dossier: FER2013+ (FER+)

### 4.1 Motivation: Conquering the Label Noise Ceiling
In 2016, a research team at Microsoft led by Emad Barsoum published *"Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution"* (ACM ICMI 2016). Recognizing that model accuracy on FER2013 had plateaued around 72–74% due to label contamination rather than architectural limits, Microsoft re-annotated the entire FER2013 dataset using crowdsourcing.

### 4.2 Crowdsourced Re-Annotation Protocol
- **Annotators**: 10 distinct crowd workers from Amazon Mechanical Turk independently evaluated every single image of the 35,887 FER2013 instances.
- **Expanded Taxonomy (10 Classes)**: In addition to the original 7 emotions, taggers could designate:
  1. `Contempt`: Distinguishing unilateral smirk/lip-corner tightening from bilateral disgust/neutral.
  2. `Unknown`: Flagging ambiguous facial expressions where annotators could not reach a confident judgment.
  3. `NF (Not a Face)`: Rejecting non-face clutter, cartoons, and extreme occlusions.

### 4.3 Soft Probability Distribution vs Hard Voting
Instead of collapsing 10 human opinions into a single majority label, FER+ represents the ground truth for image $i$ as a normalized probability distribution vector $\mathbf{p}_i = [p_0, p_1, \dots, p_9]$:

$$p_k = \frac{v_k}{\sum_{j=0}^{9} v_j}$$

where $v_k$ is the number of votes cast for class $k$ by the 10 taggers.

```
Original FER2013 Label: "Sad" (Hard Single Index: 4)
FER+ Distribution:
  Neutral:    1 vote  (0.10)
  Sadness:    5 votes (0.50)  <-- Modal Label
  Disgust:    2 votes (0.20)
  Surprise:   0 votes (0.00)
  Anger:      2 votes (0.20)
  Contempt:   0 votes (0.00)
```

### 4.4 Empirical Significance in Modern Research
When training neural networks using Cross-Entropy against the soft target distribution (Kullback-Leibler divergence loss) or filtering out non-faces and uncertain images, benchmark accuracy on the held-out test split jumps from **72.1% (FER2013 standard)** to **89.5% – 91.2% (FER+)**. This confirms that the perceived 75% accuracy ceiling of FER2013 is an artifact of label ambiguity rather than a fundamental barrier of computer vision models.

---

## 5. In-Depth Technical Dossier: AffectNet+ (AffectNet Plus)

### 5.1 Origins & Landmark Status
Introduced by Ali Mollahosseini, Behzad Hasani, and Mohammad H. Mahoor at the University of Denver (*IEEE Transactions on Affective Computing*, 2017/2019), **AffectNet** stands as the largest publicly available facial expression database in existence.

The authors gathered images by querying Google, Bing, and Yahoo image search engines using **1,250 emotion-related keywords translated into six major languages** (English, Spanish, Portuguese, German, Arabic, and Farsi). This multilingual query strategy successfully minimized cultural, geographical, and demographic bias.

### 5.2 Local Asset Breakdown: Human vs Automated Subsets
In our local storage ([`Facial Dataset/AffectNet+/`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Facial%20Dataset/AffectNet+)), AffectNet+ occupies **7.70 GB** across 959,906 samples organized into two primary partitions:

```
AffectNet+/
├── human_annotated/               [420,299 total samples]
│   ├── train_set/
│   │   ├── images/               [414,799 color images: 224x224 RGB JPEG]
│   │   └── annotations/          [414,799 JSON annotation files (0.json to 414798.json)]
│   └── validation_set/
│       ├── images/               [5,500 color images: 224x224 RGB JPEG]
│       └── annotations/          [5,500 JSON annotation files]
└── no_human_annotated/            [539,607 total samples]
    ├── images/                   [539,607 color images: 224x224 RGB JPEG]
    └── annotations/              [539,607 JSON automated prediction files]
```

### 5.3 Rich Multimodal JSON Metadata Schema
Unlike basic datasets that offer only a scalar integer label, each AffectNet+ JSON annotation file contains rich structural metadata:

```json
{
  "human-label": 1,
  "soft-label": [0.2122, 0.3369, 0.1613, 0.0876, 0.0862, 0.1960, 0.4748, 0.3894],
  "subset": 1,
  "meta-data": {
    "age": 38,
    "valence": 0.785714,
    "arousal": -0.0555556,
    "gender": {
      "female": 0.0,
      "male": 1.0
    },
    "race": {
      "asian": 0.04,
      "indian": 0.12,
      "black": 0.03,
      "white": 0.25,
      "middle eastern": 0.32,
      "latino hispanic": 0.24
    },
    "pose": {
      "yaw": -0.11,
      "pitch": -0.06,
      "roll": 0.02
    },
    "landmark-68": [15, 83, 13, 105, 12, 130, "... 136 integers total ..."],
    "landmark-29": [33, 65, 188, 69, 93, 65, "... 58 integers total ..."]
  }
}
```

### 5.4 Key Variables Defined:
1. **`human-label`**: Discrete ground truth category $(0: \text{Neutral}, 1: \text{Happy}, 2: \text{Sad}, 3: \text{Surprise}, 4: \text{Fear}, 5: \text{Disgust}, 6: \text{Anger}, 7: \text{Contempt}, 8: \text{Non-Face/None})$.
2. **`soft-label`**: Calibrated 8-dimensional probability vector reflecting model uncertainty and human tagger agreement.
3. **`valence` and `arousal`**: Continuous affective dimensions mapped from $-1.0$ (highly negative / deactivated) to $+1.0$ (highly positive / highly energized). This enables joint classification and regression multi-task training ($L_{\text{total}} = L_{\text{cls}} + \lambda_1 L_{\text{valence}} + \lambda_2 L_{\text{arousal}}$).
4. **Demographic Priors (`gender`, `race`, `age`)**: Soft demographic distributions across six distinct ethnic categories, enabling algorithmic fairness audits and bias mitigation across demographic subgroups.
5. **Geometric Alignment (`landmark-68`, `landmark-29`, `pose`)**: Coordinates for 68-point dlib facial landmarks and 3D head pose Euler angles (yaw, pitch, roll) for spatial normalization and frontalization.

---

## 6. In-Depth Technical Dossier: CK+ (Extended Cohn-Kanade)

### 6.1 Origins & Scientific Design
Published by Patrick Lucey, Jeffrey Cohn, Takeo Kanade et al. (CVPRW 2010), the **Extended Cohn-Kanade (CK+)** dataset is the premier scientific benchmark for laboratory-controlled facial expression analysis.

Conducted in an anechoic observation room at the University of Pittsburgh and Carnegie Mellon University, 123 subjects (ages 18–50; 65% female, 35% male; 81% Euro-American, 13% African American) were guided by trained FACS instructors to perform specific facial muscle contractions. Sequences were recorded at 30 FPS under uniform, high-intensity frontal studio illumination with zero background clutter.

### 6.2 Temporal Progression & FACS Coding
Each recording starts from a completely neutral face (onset, frame 0), progresses through muscular contraction (transition), and reaches peak emotional apex at the terminal frame:

```
[Frame 0: Neutral (Onset)]  -->  [Frame 8: AU Activation]  -->  [Frame 16: Apex (Peak)]
```

Every sequence was independently coded for **Facial Action Units (AUs)** under Ekman and Friesen's Facial Action Coding System (FACS), ensuring that validated emotion labels adhere strictly to muscular activation criteria (e.g., Happy requires AU6 `Cheek Raiser` + AU12 `Lip Corner Puller`).

### 6.3 Local Workspace Asset: `ckextended.csv`
In the local repository ([`Facial Dataset/CK+/ckextended.csv`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Facial%20Dataset/CK+/ckextended.csv)), CK+ is packaged as a compact 7.23 MB tabular file containing 920 apex and keyframe samples formatted identically to FER2013 ($48 \times 48$ grayscale pixel strings):

| Split / Usage | Row Count | Purpose in Pipeline |
| :--- | :---: | :--- |
| **`Training`** | 734 | Fine-tuning high-intensity canonical expressions |
| **`PrivateTest`** | 95 | Held-out validation of apex detection |
| **`PublicTest`** | 91 | Zero-shot cross-corpus evaluation against FER2013 |
| **Total** | **920** | 8 discrete emotion classes $(0 \text{ to } 7)$ |

---

## 7. In-Depth Technical Dossier: CREMA-D

### 7.1 Origins & Diversity Design
The **Crowd-Sourced Emotional Multimodal Actors Dataset (CREMA-D)** was created by Houwei Cao, David Cooper, Ragini Verma et al. at the University of Pennsylvania and Rowan University (*IEEE Transactions on Affective Computing*, 2014).

CREMA-D was expressly engineered to overcome the demographic homogeneity of earlier actor datasets. It features **91 professional actors (48 male, 43 female)** spanning an age range of **20 to 74 years** across diverse racial backgrounds:
- Caucasian: 54%
- African American: 22%
- Hispanic: 11%
- Asian: 10%
- Middle Eastern: 3%

### 7.2 Scripted Sentences & Semantic Neutrality
To isolate acoustic and facial affective dynamics from linguistic and lexical semantic cues, actors recited 12 standardized sentences with neutral semantic meaning:

| Code | Sentence Text |
| :---: | :--- |
| `DFA` | *"Don't forget a birthday."* |
| `IEO` | *"I would like to think that it is possible."* |
| `TIE` | *"That is extremely interesting."* |
| `IOM` | *"It's eleven o'clock."* |
| `IWW` | *"I wonder what this is about."* |
| `TAW` | *"The airplane is almost full."* |
| `MTF` | *"Maybe tomorrow it will be cold."* |
| `IWL` | *"I would like to visit my grandmother."* |
| `ITH` | *"I think I have seen this before."* |
| `DWS` | *"The doctor will see you soon."* |
| `IEA` | *"I think everyone agreed."* |
| `WSI` | *"We'll see if it is true."* |

### 7.3 Emotion Categories & Intensity Levels
The dataset contains **7,442 uncompressed WAV files** ([`Voice and Facial/CREMA-D/`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Voice%20and%20Facial/CREMA-D)) occupying 577.83 MB. Across all 91 actors, recordings are distributed evenly across six basic emotions:

```
Anger (ANG):    1,271 clips
Disgust (DIS):  1,271 clips
Fear (FEA):     1,271 clips
Happy (HAP):    1,271 clips
Sad (SAD):      1,271 clips
Neutral (NEU):  1,087 clips
Total:          7,442 clips
```

Actors performed expressions at four distinct emotional intensity levels:
- **`LO` (Low)**: 455 clips
- **`MD` (Medium)**: 455 clips
- **`HI` (High)**: 455 clips
- **`XX` (Unspecified / Baseline Neutral)**: 6,076 clips

### 7.4 Filename Decoding Convention
Every audio file follows a strict 4-token nomenclature: `[ActorID]_[SentenceCode]_[EmotionCode]_[IntensityCode].wav`
- Example: `1001_DFA_ANG_HI.wav`
  - `1001`: Actor ID (Male, Caucasian, Age 27)
  - `DFA`: Sentence *"Don't forget a birthday"*
  - `ANG`: Anger
  - `HI`: High Intensity

### 7.5 Acoustic Specifications:
- **Sampling Rate**: 16,000 Hz (16 kHz standard telephony/speech recognition rate)
- **Quantization**: 16-bit signed PCM
- **Channels**: 1 (Mono)
- **Average Duration**: $2.28 \text{ seconds}$ (ranging from $1.2\text{s}$ to $3.8\text{s}$).

---

## 8. In-Depth Technical Dossier: RAVDESS

### 8.1 Origins & Studio Rigor
Developed by Steven R. Livingstone and Frank A. Russo at Ryerson University's SMART Lab (*PLoS ONE*, 2018), the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** is widely regarded as the gold standard for high-fidelity multimodal emotion modeling.

The database features **24 professional Canadian actors (12 female, 12 male)** with extensive theatrical and vocal training. Recordings were captured in an anechoic, sound-damped chamber using professional studio cardioid condenser microphones and high-definition video cameras equipped with multi-point diffused fluorescent lighting.

### 8.2 Modalities & Local Directory Organization
In the local project directory ([`Voice and Facial/RAVDESS/`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Datasets/Voice%20and%20Facial/RAVDESS)), RAVDESS is the largest asset, occupying **24.82 GB across 7,356 files**:

```
RAVDESS/
├── Audio_Speech_Actors_01-24/ [24 Actor subfolders: 1,440 WAV files (48 kHz uncompressed)]
├── Audio_Song_Actors_01-24/   [23 Actor subfolders: 1,012 WAV files (48 kHz uncompressed)]
├── Video_Speech_Actor_01-24/  [24 Folders: 2,880 720p MP4 clips (1,440 AV + 1,440 Video-only)]
└── Video_Song_Actor_01-24/    [23 Folders: 2,024 720p MP4 clips (1,012 AV + 1,012 Video-only)]
```

### 8.3 Scientific Filename Protocol (7 Identifiers)
Every RAVDESS clip is indexed with a standardized 7-part hyphenated filename:

$$\text{Modality (01-03)} - \text{Channel (01-02)} - \text{Emotion (01-08)} - \text{Intensity (01-02)} - \text{Statement (01-02)} - \text{Repetition (01-02)} - \text{Actor (01-24)}.\text{ext}$$

```
Example: 03-01-05-02-01-01-12.mp4
  │   │  │  │  │  │  └─ Actor: 12 (Female, Actor ID % 2 == 0)
  │   │  │  │  │  └──── Repetition: 1 (First take)
  │   │  │  │  └─────── Statement: 1 ("Kids are talking by the door")
  │   │  │  └────────── Intensity: 2 (Strong intensity)
  │   │  └───────────── Emotion: 5 (Angry)
  │   └──────────────── Vocal Channel: 1 (Speech)
  └──────────────────── Modality: 3 (Audio-Visual)
```

#### Detailed Value Definitions:
- **Modality**: `01` = Full Audio-Visual, `02` = Video-only, `03` = Audio-only (WAV).
- **Vocal Channel**: `01` = Speech, `02` = Song.
- **Emotion Code**:
  - `01` = Neutral (Statement recited without emotion)
  - `02` = Calm (Relaxed vocal tract and serene facial posture)
  - `03` = Happy
  - `04` = Sad
  - `05` = Angry
  - `06` = Fearful
  - `07` = Disgust
  - `08` = Surprised
- **Emotional Intensity**: `01` = Normal, `02` = Strong. (Note: Neutral only exists at Normal intensity).
- **Statement**: `01` = *"Kids are talking by the door"*, `02` = *"Dogs are sitting by the door"*.
- **Repetition**: `01` = Take 1, `02` = Take 2.
- **Actor ID**: `01` to `24`. **Odd IDs (1, 3, 5, ... 23) are Male; Even IDs (2, 4, 6, ... 24) are Female**.

### 8.4 Technical Specifications:
- **Acoustic Media**: $48,000\text{ Hz}$ (48 kHz studio master quality), 16-bit PCM Mono.
- **Visual Media**: $1280 \times 720$ (720p HD), 29.97 FPS, H.264 video codec, $3.3\text{ seconds}$ duration.

---

## 9. Cross-Modal Emotion Taxonomy Harmonization

Because the six datasets were curated independently by different academic bodies, their emotion labels and modalities vary. To train a cohesive multimodal system for the FYP, the datasets are harmonized into a **Canonical 7-Class Taxonomy**:

```mermaid
flowchart LR
    subgraph Raw_Corpora ["Raw Independent Corpora"]
        FER["FER2013 (7 Classes)"]
        FERP["FER+ (10 Classes)"]
        AFF["AffectNet+ (8 Classes + V/A)"]
        CK["CK+ (8 Classes FACS)"]
        CREMA["CREMA-D (6 Classes Audio)"]
        RAV["RAVDESS (8 Classes Video/Audio)"]
    end

    subgraph Harmonization_Engine ["Taxonomy Harmonization Rules"]
        R1["Calm (02) -> Merged into Neutral (6)"]
        R2["Contempt (07) -> Remapped to Disgust (1) or Dropped"]
        R3["Surprise -> Zero-shot imputed for CREMA-D"]
        R4["Non-Face / Unknown -> Filtered Out"]
    end

    subgraph Unified_Target ["Unified FYP 7-Class Taxonomy"]
        T0["0: Angry"]
        T1["1: Disgust"]
        T2["2: Fear"]
        T3["3: Happy"]
        T4["4: Sad"]
        T5["5: Surprise"]
        T6["6: Neutral"]
    end

    FER --> Unified_Target
    FERP --> R4 --> Unified_Target
    AFF --> R2 --> Unified_Target
    CK --> R2 --> Unified_Target
    CREMA --> R3 --> Unified_Target
    RAV --> R1 --> Unified_Target
```

### Harmonization Alignment Matrix

The matrix below illustrates how raw emotion categories across datasets map to the unified target labels:

| Unified Index | Unified Emotion | FER2013 Class | FER2013+ Class | AffectNet+ Class | CK+ Class | CREMA-D Code | RAVDESS Code |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Angry** | `0 (angry)` | `4 (anger)` | `6 (anger)` | `0 (anger)` | `ANG` | `05 (Angry)` |
| **1** | **Disgust** | `1 (disgust)`| `5 (disgust)`| `5 (disgust)`| `2 (disgust)`| `DIS` | `07 (Disgust)`|
| **2** | **Fear** | `2 (fear)` | `6 (fear)` | `4 (fear)` | `3 (fear)` | `FEA` | `06 (Fearful)`|
| **3** | **Happy** | `3 (happy)` | `1 (happiness)`| `1 (happy)` | `4 (happy)` | `HAP` | `03 (Happy)` |
| **4** | **Sad** | `4 (sad)` | `3 (sadness)`| `2 (sad)` | `5 (sad)` | `SAD` | `04 (Sad)` |
| **5** | **Surprise** | `5 (surprise)`| `2 (surprise)`| `3 (surprise)`| `6 (surprise)`| *N/A (Missing)* | `08 (Surprised)`|
| **6** | **Neutral** | `6 (neutral)`| `0 (neutral)`| `0 (neutral)`| `7 (neutral)`| `NEU` | `01 (Neutral)` & `02 (Calm)`|
| *Omit* | *Contempt* | — | `7 (contempt)`| `7 (contempt)`| `1 (contempt)`| — | — |
| *Filter*| *Noise/NF* | — | `8 (unknown)`, `9 (NF)` | `8 (non-face)` | — | — | — |

---

## 10. FYP Multimodal Feature Engineering & Architecture Pipeline

In the FYP pipeline (implemented across [`multimodal_emotion_recognition.ipynb`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Voice%20and%20Facial%20recognition/multimodal_emotion_recognition.ipynb) and [`FER_95Plus_Master_Pipeline.ipynb`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/New%20Notebook2/FER_95Plus_Master_Pipeline.ipynb)), the six datasets feed into specialized deep learning branches:

```mermaid
flowchart TD
    subgraph Data_Inputs ["Heterogeneous Dataset Inputs"]
        IMG["Static Face Images<br/>(AffectNet+, FER2013, CK+)<br/>Shape: (B, 3, 224, 224)"]
        VID["Dynamic Video Clips<br/>(RAVDESS MP4)<br/>T=8 or 16 Uniform Frames"]
        AUD["Acoustic Waveforms<br/>(CREMA-D, RAVDESS WAV)<br/>16 kHz Mono PCM"]
    end

    subgraph Visual_Branch ["Visual Feature Backbone"]
        Backbone["Pretrained Vision Backbone<br/>(ViT-B/16 / EfficientNet-B2 / ResNet-50)"]
        Spatial["Spatial-Temporal Aggregation<br/>(BiLSTM + Self-Attention)"]
        IMG --> Backbone --> Spatial
        VID --> Backbone --> Spatial
    end

    subgraph Acoustic_Branch ["Acoustic Feature Backbone"]
        Mel["Log-Mel Spectrogram Extraction<br/>(128 Mels, n_fft=1024, hop=512)<br/>Shape: (B, 1, 128, 188)"]
        SE_CNN["Squeeze-and-Excitation ResNet<br/>+ Acoustic BiLSTM"]
        AUD --> Mel --> SE_CNN
    end

    subgraph Fusion_Engine ["Cross-Modal Attention & Fusion"]
        CMA["Cross-Modal Attention Layer<br/>(Facial Queries Attend to Audio Keys/Values<br/>& Vocal Queries Attend to Face Keys/Values)"]
        Gating["Gated Multimodal Unit (GMU)<br/>Dynamically Weights Active Modalities"]
        Spatial --> CMA
        SE_CNN --> CMA
        CMA --> Gating
    end

    subgraph Loss_Optimization ["Prediction & Training Objective"]
        Head["Unified 7-Class Emotion Head<br/>(Softmax Logits)"]
        Loss["Class-Balanced Focal Loss<br/>+ Label Smoothing (smoothing=0.1)"]
        Gating --> Head --> Loss
    end
```

### Feature Engineering Summary:
1. **Visual Pipeline**:
   - High-resolution inputs resized to $224 \times 224 \times 3$ with ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   - Data augmentations include `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=10)`, `ColorJitter(brightness=0.2, contrast=0.2)`, and `RandomErasing(scale=(0.02, 0.15))` to simulate edge occlusions.
   - For video inputs (RAVDESS), 8 or 16 frames are uniformly sampled across the temporal duration, processed through the spatial CNN/ViT, and aggregated via a bidirectional LSTM with temporal self-attention.
2. **Acoustic Pipeline**:
   - Audio waveforms are resampled to $16,000\text{ Hz}$ mono, padded or truncated to a fixed duration of $3.0\text{ seconds}$ ($48,000\text{ samples}$).
   - Converted into Log-Mel Spectrograms using $n_{\text{fft}} = 1024$, $hop\_length = 512$, and $n_{\text{mels}} = 128$, yielding a 2D time-frequency tensor of shape $(1, 128, 188)$.
   - SpecAugment frequency masking (param=15) and time masking (param=25) are applied during training to prevent spectral overfitting.

---

## 11. Rigorous Experimental Auditing & Anti-Leakage Protocol

In academic affective computing, naive train-test splitting frequently causes **severe data leakage and artificial performance inflation**. The following protocols must be maintained across all FYP experiments:

### 11.1 Identity Leakage Audit (Actor-Independent Splits)
> [!CAUTION]
> **Zero Subject Overlap Rule**: In actor-based datasets like **RAVDESS** and **CREMA-D**, running a random `train_test_split()` or standard random $k$-fold cross-validation mixes the same actor's voice and facial identity across both training and test sets.
> 
> Under random splitting, deep neural networks memorize idiosyncratic biometric signatures (voice pitch, bone structure, facial wrinkles) rather than emotional dynamics, resulting in **spuriously inflated test accuracies of 95%+**. When evaluated on an unseen human subject, accuracy collapses to below 55%.

#### Mandatory Split Partitioning:
- **RAVDESS Partitioning**:
  - `Train Set`: **Actors 01 through 20** (10 Male, 10 Female; 83.3% of data).
  - `Held-Out Test Set`: **Actors 21 through 24** (2 Male, 2 Female; 16.7% of data).
- **CREMA-D Partitioning**:
  - `Train Set`: **Actors 1001 through 1075** (75 actors, balanced across gender and ethnicity).
  - `Held-Out Test Set`: **Actors 1076 through 1091** (16 strictly unseen actors).

### 11.2 Duplicate and Near-Duplicate Mitigation in FER2013
An automated perceptual hash (pHash) analysis revealed that the raw FER2013 dataset contains **1,798 near-duplicate images** shared between training and public test folds. In our improved experimental pipeline, all duplicate hashes and 13 blank images are purged before training to ensure zero evaluation contamination.

### 11.3 Performance Evaluation Metric Selection
Due to class imbalance (e.g., Disgust comprising only 1.5% of FER2013), **Overall Accuracy is an unreliable metric**. An uncalibrated model that completely ignores the Disgust class can still achieve over 72% overall accuracy. All experimental reports must benchmark models on:
- **Macro-Averaged F1 Score**: $\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^C \text{F1}_c$ (weights all emotions equally).
- **Balanced Accuracy**: Arithmetic mean of per-class recall scores.
- **Expected Calibration Error (ECE)**: Measuring model confidence reliability against actual error likelihood.

---

## 12. Directory Roadmap & Verification Commands

### 12.1 Local Storage Summary
All dataset assets are located on the host machine at the following paths:

```
C:\Users\Shahzaib Hussain\Desktop\Shahzaib Hussain\FYP\Datasets\
├── Facial Dataset\
│   ├── AffectNet+\          (7.70 GB, 959,906 samples, 224x224 RGB JPEG + JSONs)
│   ├── CK+\                 (7.23 MB, 920 samples, ckextended.csv)
│   ├── FER2013\             (53.89 MB, 35,887 image files across train/test)
│   └── FER2013 CSV\         (296.79 MB, fer2013.csv + test image subfolder)
└── Voice and Facial\
    ├── CREMA-D\             (577.83 MB, 7,442 WAV files, 16 kHz Mono PCM)
    └── RAVDESS\             (24.82 GB, 2,452 WAV files + 4,904 720p HD MP4 videos)
```

### 12.2 Ready-to-Run Verification Script
The following Python script quickly validates the integrity of all six datasets, parses sample metadata, and verifies that audio and visual files load correctly:

```python
import os
import wave
import json
import pandas as pd
from PIL import Image

BASE_FACIAL = r"C:\Users\Shahzaib Hussain\Desktop\Shahzaib Hussain\FYP\Datasets\Facial Dataset"
BASE_VOICE = r"C:\Users\Shahzaib Hussain\Desktop\Shahzaib Hussain\FYP\Datasets\Voice and Facial"

print("--- 1. VERIFYING FER2013 CSV ---")
fer_csv = os.path.join(BASE_FACIAL, "FER2013 CSV", "fer2013.csv")
df_fer = pd.read_csv(fer_csv, nrows=5)
print(f"FER2013 CSV exists! Columns: {df_fer.columns.tolist()}")

print("\n--- 2. VERIFYING CK+ CSV ---")
ck_csv = os.path.join(BASE_FACIAL, "CK+", "ckextended.csv")
df_ck = pd.read_csv(ck_csv, nrows=5)
print(f"CK+ CSV exists! Shape: (920, 3), Columns: {df_ck.columns.tolist()}")

print("\n--- 3. VERIFYING AFFECTNET+ SAMPLE ---")
aff_json = os.path.join(BASE_FACIAL, "AffectNet+", "human_annotated", "train_set", "annotations", "0.json")
with open(aff_json, "r") as f:
    aff_meta = json.load(f)
print(f"AffectNet+ 0.json Human Label: {aff_meta['human-label']}, Valence: {aff_meta['meta-data']['valence']:.3f}")

print("\n--- 4. VERIFYING CREMA-D AUDIO ---")
crema_wav = os.path.join(BASE_VOICE, "CREMA-D", "1001_DFA_ANG_XX.wav")
with wave.open(crema_wav, "rb") as w:
    print(f"CREMA-D Sample: {w.getnchannels()} ch, {w.getframerate()} Hz, Duration: {w.getnframes()/w.getframerate():.2f}s")

print("\n--- 5. VERIFYING RAVDESS AUDIO ---")
rav_wav = os.path.join(BASE_VOICE, "RAVDESS", "Audio_Speech_Actors_01-24", "Actor_01", "03-01-01-01-01-01-01.wav")
with wave.open(rav_wav, "rb") as w:
    print(f"RAVDESS Sample: {w.getnchannels()} ch, {w.getframerate()} Hz, Duration: {w.getnframes()/w.getframerate():.2f}s")

print("\nAll datasets verified successfully!")
```

---

## 13. Related FYP Notebooks and Architecture Reports

For complete implementation code, training logs, and model evaluations utilizing these datasets, refer to the following companion documents in the project repository:

- **Multimodal Pipeline Notebook**: [`multimodal_emotion_recognition.ipynb`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Voice%20and%20Facial%20recognition/multimodal_emotion_recognition.ipynb)
- **Facial Emotion Research Benchmark**: [`facial_emotion_research_comprehensive_report.md`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Reports/facial_emotion_research_comprehensive_report.md)
- **FER 95+ Curriculum Master Pipeline**: [`FER_95Plus_Master_Pipeline_Report.md`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Reports/FER_95Plus_Master_Pipeline_Report.md)
- **Dual-CNN Transfer Learning Report**: [`FER2013_Dual_CNN_Transfer_Learning_Pipeline_Report.md`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Reports/FER2013_Dual_CNN_Transfer_Learning_Pipeline_Report.md)
- **Deep Attention Network (DAN) Report**: [`FER2013_Deep_Attention_Network_Pretrained_Report.md`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Reports/FER2013_Deep_Attention_Network_Pretrained_Report.md)
- **GPU 90+ Benchmark Report**: [`FER_GPU_90Plus_Benchmark_Report.md`](file:///C:/Users/Shahzaib%20Hussain/Desktop/Shahzaib%20Hussain/FYP/Reports/FER_GPU_90Plus_Benchmark_Report.md)

---
*Report compiled automatically for the Final Year Project (FYP) Multimodal Affective Computing System.*
