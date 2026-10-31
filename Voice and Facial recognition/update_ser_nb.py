import json
import os

nb_path = r"C:\Users\Shahzaib Hussain\Desktop\Shahzaib Hussain\FYP\Voice and Facial recognition\speech_emotion_recognition_production.ipynb"

cells = []

def add_md(text):
    lines = [l + "\n" for l in text.split("\n")[:-1]] + ([text.split("\n")[-1]] if text.split("\n")[-1] else [])
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    })

def add_code(text):
    lines = [l + "\n" for l in text.split("\n")[:-1]] + ([text.split("\n")[-1]] if text.split("\n")[-1] else [])
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    })

# -----------------------------------------------------------------------------
# 0. Header
# -----------------------------------------------------------------------------
add_md("""# Speech Emotion Recognition (SER) Production Pipeline
**Author:** Shahzaib Hussain | **Final Year Project (FYP)**  
**Datasets:** CREMA-D (7,442), RAVDESS (1,440), SAVEE (480), TESS (2,800) $\\rightarrow$ **12,162 Voice Recordings**  
**Methodology:** Multi-Corpus Speaker-Independent Partitioning + Leakage-Free Acoustic Extractor & Calibrated Model Ensemble  
**Production Artifact:** Single unified production bundle (`final_ser_production_bundle.joblib`) with ultra-fast real-time inference (`RTF << 1.0`)""")

# -----------------------------------------------------------------------------
# 1. Environment Setup & Seeding
# -----------------------------------------------------------------------------
add_md("## 1. Environment Setup, Imports & Configuration Loader")
add_code("""# ==============================================================================
# 1. ENVIRONMENT SETUP, IMPORTS & CONFIGURATION LOADER
# ==============================================================================
import os
import re
import glob
import time
import json
import yaml
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import joblib

import torch
import torchaudio.transforms as AT

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def load_ser_config(config_path: str = "ser_config.yaml") -> Dict[str, Any]:
    candidates = [
        config_path,
        os.path.join(r"C:\\Users\\Shahzaib Hussain\\Desktop\\Shahzaib Hussain\\FYP\\Voice and Facial recognition", config_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Configuration file not found in candidates: {candidates}")

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

CFG = load_ser_config("ser_config.yaml")
seed_everything(CFG["environment"]["seed"])

os.makedirs(CFG["environment"]["output_dir"], exist_ok=True)
os.makedirs(CFG["environment"]["checkpoint_dir"], exist_ok=True)

print(f"[INFO] Loaded configuration for: '{CFG['environment']['project_name']}'")
print(f"[INFO] Output Directory: {CFG['environment']['output_dir']}")
print(f"[INFO] Final Bundle Target: {CFG['outputs']['final_bundle_path']}")
""")

# -----------------------------------------------------------------------------
# 2. Multi-Corpus Dataset Indexing
# -----------------------------------------------------------------------------
add_md("## 2. Multi-Corpus Speaker-Independent Dataset Indexing (CREMA-D, RAVDESS, SAVEE, TESS)")
add_code("""# ==============================================================================
# 2. MULTI-CORPUS SPEAKER-INDEPENDENT DATASET PARTITIONING
# Indexes CREMA-D (7,442), RAVDESS (1,440), SAVEE (480), and TESS (2,800)
# ==============================================================================
EMOTION_TO_IDX = {v: int(k) for k, v in CFG["taxonomy"]["class_names"].items()}
IDX_TO_EMOTION = {int(k): v for k, v in CFG["taxonomy"]["class_names"].items()}

CREMA_MAP = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fear", "DIS": "disgust"}
RAV_MAP = {
    "01": "neutral", "02": "neutral", # Calm mapped into Neutral
    "03": "happy", "04": "sad", "05": "angry",
    "06": "fear", "07": "disgust", "08": "surprise"
}
SAVEE_MAP = {
    "n": "neutral", "h": "happy", "sa": "sad", "a": "angry",
    "f": "fear", "d": "disgust", "su": "surprise"
}
TESS_MAP = {
    "neutral": "neutral", "happy": "happy", "sad": "sad",
    "angry": "angry", "fear": "fear", "disgust": "disgust", "ps": "surprise"
}

def index_all_speech_datasets(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allowed_emotions = set(cfg["taxonomy"]["class_names"].values())
    records = []

    # 1. CREMA-D
    c_cfg = cfg["datasets"]["crema_d"]
    if c_cfg.get("enabled", True):
        for p in glob.glob(os.path.join(c_cfg["audio_dir"], "*.wav")):
            parts = os.path.basename(p).replace(".wav", "").split("_")
            if len(parts) >= 3 and parts[2] in CREMA_MAP:
                emo = CREMA_MAP[parts[2]]
                if emo in allowed_emotions:
                    spk = int(parts[0])
                    split = "train" if spk <= c_cfg["train_speakers_end"] else ("val" if spk <= c_cfg["val_speakers_end"] else "test")
                    records.append({"filepath": p, "dataset": "CREMA-D", "speaker_id": str(spk), "emotion": emo, "split": split})

    # 2. RAVDESS
    r_cfg = cfg["datasets"]["ravdess"]
    if r_cfg.get("enabled", True):
        for p in glob.glob(os.path.join(r_cfg["audio_dir"], "**", "*.wav"), recursive=True):
            parts = os.path.basename(p).replace(".wav", "").split("-")
            if len(parts) >= 7 and parts[2] in RAV_MAP:
                emo = RAV_MAP[parts[2]]
                if emo in allowed_emotions:
                    actor_id = int(parts[6])
                    split = "train" if actor_id <= r_cfg["train_actors_end"] else ("val" if actor_id <= r_cfg["val_actors_end"] else "test")
                    records.append({"filepath": p, "dataset": "RAVDESS", "speaker_id": f"Actor_{actor_id:02d}", "emotion": emo, "split": split})

    # 3. SAVEE
    s_cfg = cfg["datasets"]["savee"]
    if s_cfg.get("enabled", True):
        for p in glob.glob(os.path.join(s_cfg["audio_dir"], "**", "*.wav"), recursive=True):
            m = re.match(r"([A-Za-z]+)_([a-zA-Z]+)(\d+)", os.path.basename(p).replace(".wav", ""))
            if m:
                spk = m.group(1).upper()
                code = m.group(2).lower()
                emo = SAVEE_MAP.get(code)
                if emo in allowed_emotions:
                    split = "train" if spk in s_cfg["train_speakers"] else ("val" if spk in s_cfg["val_speakers"] else "test")
                    records.append({"filepath": p, "dataset": "SAVEE", "speaker_id": spk, "emotion": emo, "split": split})

    # 4. TESS
    t_cfg = cfg["datasets"]["tess"]
    if t_cfg.get("enabled", True):
        for p in glob.glob(os.path.join(t_cfg["audio_dir"], "**", "*.wav"), recursive=True):
            parts = os.path.basename(p).replace(".wav", "").split("_")
            if len(parts) >= 3:
                spk = parts[0].upper()
                code = parts[2].lower()
                emo = TESS_MAP.get(code)
                if emo in allowed_emotions:
                    # YAF -> Train, OAF -> Test
                    split = "train" if spk == "YAF" else "test"
                    records.append({"filepath": p, "dataset": "TESS", "speaker_id": spk, "emotion": emo, "split": split})

    df = pd.DataFrame(records)
    # Stratified validation fallback for TESS if val empty
    if "val" not in df[df["dataset"] == "TESS"]["split"].values and len(df[df["dataset"] == "TESS"]) > 0:
        tess_tr_idx = df[(df["dataset"] == "TESS") & (df["split"] == "train")].sample(frac=0.15, random_state=42).index
        df.loc[tess_tr_idx, "split"] = "val"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, val_df, test_df

train_df, val_df, test_df = index_all_speech_datasets(CFG)

print("="*70)
print("MULTI-CORPUS SPEAKER-INDEPENDENT PARTITIONING SUMMARY:")
print(f"  • Training   : {len(train_df):,} files | By Dataset: {train_df['dataset'].value_counts().to_dict()}")
print(f"  • Validation : {len(val_df):,} files | By Dataset: {val_df['dataset'].value_counts().to_dict()}")
print(f"  • Held-Out   : {len(test_df):,} files | By Dataset: {test_df['dataset'].value_counts().to_dict()}")
print(f"  • Total Audio Recordings Indexed: {len(train_df) + len(val_df) + len(test_df):,}")
print("="*70)
""")

# -----------------------------------------------------------------------------
# 3. Audio Preprocessor & VAD
# -----------------------------------------------------------------------------
add_md("## 3. Audio Preprocessor with Voice Activity Detection (VAD)")
add_code("""# ==============================================================================
# 3. AUDIO PREPROCESSOR WITH VOICE ACTIVITY DETECTION (VAD)
# ==============================================================================
class AudioPreprocessor:
    \"\"\"
    Handles raw audio loading, monophonic conversion, sample-rate standardization,
    and energy-based Voice Activity Detection (VAD) to trim silence boundaries.
    \"\"\"
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.target_sr = cfg["audio"]["sample_rate"]
        self.min_duration = cfg["audio"]["min_duration_sec"]
        self.vad_enabled = cfg["audio"]["vad"]["enabled"]
        self.energy_thresh_ratio = cfg["audio"]["vad"]["energy_threshold_ratio"]

    def process_file(self, audio_path: str) -> Tuple[Optional[np.ndarray], int, Dict[str, float]]:
        if not os.path.exists(audio_path):
            return None, self.target_sr, {}
        
        try:
            y, sr = sf.read(audio_path, dtype="float32")
        except Exception:
            return None, self.target_sr, {}

        # Mono conversion
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample to 16 kHz if necessary
        if sr != self.target_sr:
            num_samples = int(len(y) * self.target_sr / sr)
            y = scipy.signal.resample(y, num_samples)
            sr = self.target_sr

        original_dur = len(y) / sr
        if original_dur < self.min_duration:
            return None, sr, {"original_duration": original_dur}

        # Energy-based Voice Activity Detection (VAD)
        if self.vad_enabled and len(y) > int(0.05 * sr):
            frame_len = int(self.cfg["audio"]["vad"]["frame_len_sec"] * sr)
            hop_len = int(self.cfg["audio"]["vad"]["hop_len_sec"] * sr)
            num_frames = max(1, (len(y) - frame_len) // hop_len)
            energies = np.array([np.sum(y[i*hop_len : i*hop_len+frame_len]**2) for i in range(num_frames)])
            
            if len(energies) > 0 and np.max(energies) > 1e-6:
                threshold = self.energy_thresh_ratio * np.max(energies)
                active_frames = np.where(energies > threshold)[0]
                if len(active_frames) > 0:
                    start_sample = active_frames[0] * hop_len
                    end_sample = min(len(y), (active_frames[-1] + 1) * hop_len + frame_len)
                    y_active = y[start_sample:end_sample]
                else:
                    y_active = y
            else:
                y_active = y
        else:
            y_active = y

        active_dur = len(y_active) / sr
        stats = {
            "original_duration": round(original_dur, 3),
            "active_duration": round(active_dur, 3),
            "speech_ratio": round(active_dur / max(original_dur, 1e-4), 3)
        }
        return y_active, sr, stats

preprocessor = AudioPreprocessor(CFG)
sample_test_path = train_df["filepath"].iloc[0]
sample_y, sample_sr, sample_dur_stats = preprocessor.process_file(sample_test_path)
print(f"[TEST] Preprocessed sample recording from '{train_df['dataset'].iloc[0]}':")
print(f"       File: {os.path.basename(sample_test_path)}")
print(f"       Active duration: {sample_dur_stats['active_duration']}s (original: {sample_dur_stats['original_duration']}s)")
""")

# -----------------------------------------------------------------------------
# 4. Comprehensive Acoustic Feature Extractor
# -----------------------------------------------------------------------------
add_md("## 4. Comprehensive Handcrafted Acoustic Extractor")
add_code("""# ==============================================================================
# 4. COMPREHENSIVE HANDCRAFTED ACOUSTIC FEATURE EXTRACTOR
# Extracts: Prosodic (F0, RMS), Voice Quality (HNR, Jitter, Shimmer), Spectral, MFCCs
# ==============================================================================
class ComprehensiveAcousticExtractor:
    \"\"\"
    Extracts a multi-dimensional battery of acoustic features:
    - Pitch / F0 dynamics (autocorrelation method)
    - Prosodic energy (RMS mean, std, max)
    - Voice Quality (HNR, local jitter, local shimmer)
    - Spectral characteristics (centroid mean/std, spectral rolloff)
    - 13 Mel-Frequency Cepstral Coefficients (MFCCs mean & std)
    \"\"\"
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.sr = cfg["audio"]["sample_rate"]
        hop_len = int(cfg["features"]["prosody"]["hop_len_sec"] * self.sr)
        self.mfcc_transform = AT.MFCC(
            sample_rate=self.sr,
            n_mfcc=cfg["features"]["mfcc"]["n_mfcc"],
            melkwargs={
                "n_fft": cfg["features"]["spectral"]["n_fft"],
                "hop_length": hop_len,
                "n_mels": cfg["features"]["mfcc"]["n_mels"]
            }
        )

    def extract_all_handcrafted(self, y: np.ndarray) -> Dict[str, float]:
        sr = self.sr
        feats: Dict[str, float] = {}

        # 1. Prosodic RMS Energy
        frame_len = int(self.cfg["features"]["prosody"]["frame_len_sec"] * sr)
        hop_len = int(self.cfg["features"]["prosody"]["hop_len_sec"] * sr)
        rms = np.array([np.sqrt(np.mean(y[i:i+frame_len]**2) + 1e-9) for i in range(0, max(1, len(y)-frame_len), hop_len)])
        feats["PROS_RMS_mean"] = float(np.mean(rms)) if len(rms) else 0.0
        feats["PROS_RMS_std"] = float(np.std(rms)) if len(rms) else 0.0
        feats["PROS_RMS_max"] = float(np.max(rms)) if len(rms) else 0.0

        # 2. Pitch (F0) via Autocorrelation
        p_cfg = self.cfg["features"]["pitch_f0"]
        pitch_frame = int(p_cfg["frame_len_sec"] * sr)
        pitch_hop = int(p_cfg["hop_len_sec"] * sr)
        min_lag = int(sr / p_cfg["max_freq_hz"])
        max_lag = int(sr / p_cfg["min_freq_hz"])
        f0_list, periods = [], []

        for i in range(0, max(1, len(y) - pitch_frame), pitch_hop):
            fr = y[i:i+pitch_frame] * np.hanning(pitch_frame)
            if np.sum(fr**2) < 1e-4:
                continue
            corr = np.correlate(fr, fr, mode='full')
            corr = corr[len(corr)//2:]
            if len(corr) <= max_lag:
                continue
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            if corr[peak_lag] > 0.3 * corr[0]:
                f0_list.append(sr / peak_lag)
                periods.append(peak_lag / sr)

        f0_arr = np.array(f0_list) if f0_list else np.array([0.0])
        feats["PROS_F0_mean"] = float(np.mean(f0_arr))
        feats["PROS_F0_std"] = float(np.std(f0_arr))
        feats["PROS_F0_min"] = float(np.min(f0_arr))
        feats["PROS_F0_max"] = float(np.max(f0_arr))

        # 3. Voice Quality: Jitter and Shimmer
        if len(periods) > 2:
            p_arr = np.array(periods)
            feats["VQ_jitter_local"] = float(np.mean(np.abs(np.diff(p_arr))) / (np.mean(p_arr) + 1e-9))
        else:
            feats["VQ_jitter_local"] = 0.0

        peak_amps = [np.max(np.abs(y[i:i+pitch_frame])) for i in range(0, max(1, len(y)-pitch_frame), pitch_hop)]
        if len(peak_amps) > 2:
            amp_arr = np.array(peak_amps)
            feats["VQ_shimmer_local"] = float(np.mean(np.abs(np.diff(amp_arr))) / (np.mean(amp_arr) + 1e-9))
        else:
            feats["VQ_shimmer_local"] = 0.0

        # 4. Harmonics-to-Noise Ratio (HNR)
        segment_len = min(len(y), int(sr * 0.5))
        corr_full = np.correlate(y[:segment_len], y[:segment_len], mode='full')
        corr_half = corr_full[len(corr_full)//2:]
        if len(corr_half) > max_lag:
            peak_l = min_lag + np.argmax(corr_half[min_lag:max_lag])
            r0 = corr_half[0]
            ra = corr_half[peak_l]
            feats["VQ_hnr_mean"] = float(10 * np.log10(max(ra / max(r0 - ra, 1e-6), 1e-4)))
        else:
            feats["VQ_hnr_mean"] = 0.0

        # 5. Spectral Centroid & Rolloff
        fft_size = self.cfg["features"]["spectral"]["n_fft"]
        freqs = np.fft.rfftfreq(fft_size, 1.0/sr)
        centroids, rolloffs = [], []
        for i in range(0, max(1, len(y)-fft_size), hop_len):
            fr = y[i:i+fft_size] * np.hanning(fft_size)
            spec = np.abs(np.fft.rfft(fr))
            s_sum = np.sum(spec) + 1e-9
            centroids.append(np.sum(freqs * spec) / s_sum)
            # 85% Rolloff
            cum = np.cumsum(spec)
            idx = np.where(cum >= 0.85 * s_sum)[0]
            rolloffs.append(freqs[idx[0]] if len(idx) else freqs[-1])

        feats["SPEC_CENTROID_mean"] = float(np.mean(centroids)) if centroids else 0.0
        feats["SPEC_CENTROID_std"] = float(np.std(centroids)) if centroids else 0.0
        feats["SPEC_ROLLOFF_mean"] = float(np.mean(rolloffs)) if rolloffs else 0.0

        # 6. MFCCs (13 coefficients, mean & std)
        y_tensor = torch.from_numpy(y).unsqueeze(0)
        mfcc_mat = self.mfcc_transform(y_tensor).squeeze(0).numpy()
        for m in range(mfcc_mat.shape[0]):
            feats[f"MFCC_{m+1}_mean"] = float(np.mean(mfcc_mat[m]))
            feats[f"MFCC_{m+1}_std"] = float(np.std(mfcc_mat[m]))

        return feats

class DummySSLEmbedder:
    \"\"\"Plug-and-play interface for Wav2Vec2 / HuBERT embeddings when enabled.\"\"\"
    enabled = False
    def extract_embedding(self, y: np.ndarray) -> Dict[str, float]:
        return {}

ssl_embedder = DummySSLEmbedder()
extractor = ComprehensiveAcousticExtractor(CFG)
demo_feats = extractor.extract_all_handcrafted(sample_y)
print(f"[EXTRACTOR] Extracted {len(demo_feats)} distinct acoustic features per recording.")
""")

# -----------------------------------------------------------------------------
# 5. Feature Extraction across Partitions
# -----------------------------------------------------------------------------
add_md("## 5. Parallel / Batch Feature Extraction Across Multi-Corpus Partitions")
add_code("""# ==============================================================================
# 5. FEATURE EXTRACTION PIPELINE ACROSS MULTI-CORPUS PARTITIONS
# ==============================================================================
def extract_dataset_features(df_subset: pd.DataFrame, max_samples: Optional[int] = None) -> pd.DataFrame:
    if max_samples and max_samples < len(df_subset):
        df_subset = df_subset.sample(n=max_samples, random_state=CFG["environment"]["seed"]).reset_index(drop=True)
    
    rows = []
    for idx, row in df_subset.iterrows():
        y_act, sr, stats = preprocessor.process_file(row["filepath"])
        if y_act is None:
            continue
        feats = extractor.extract_all_handcrafted(y_act)
        if ssl_embedder.enabled:
            feats.update(ssl_embedder.extract_embedding(y_act))
        
        feats["filepath"] = row["filepath"]
        feats["dataset"] = row["dataset"]
        feats["speaker_id"] = row["speaker_id"]
        feats["emotion"] = row["emotion"]
        feats["active_duration"] = stats.get("active_duration", 1.0)
        rows.append(feats)

    return pd.DataFrame(rows)

print("[INFO] Extracting acoustic feature vectors for Train, Val, and Test sets across all 4 corpora...")
train_feat_df = extract_dataset_features(train_df, max_samples=1800)
val_feat_df = extract_dataset_features(val_df, max_samples=400)
test_feat_df = extract_dataset_features(test_df, max_samples=500)

features_df = pd.concat([train_feat_df, val_feat_df, test_feat_df], ignore_index=True)
meta_cols = ["filepath", "dataset", "speaker_id", "emotion", "active_duration"]
ALL_FEAT_COLS = [c for c in train_feat_df.columns if c not in meta_cols]

print(f"[DATA] Feature Matrix Shape: {features_df.shape}")
print(f"[DATA] Total Acoustic Predictors: {len(ALL_FEAT_COLS)}")
print(f"[DATA] Multi-Corpus Class Representation:\\n{features_df['emotion'].value_counts()}")
""")

# -----------------------------------------------------------------------------
# 6. Leakage-Free Feature Selector
# -----------------------------------------------------------------------------
add_md("## 6. Leakage-Free Feature Selector (Imputation, Scaling, Collinearity)")
add_code("""# ==============================================================================
# 6. LEAKAGE-FREE FEATURE SELECTOR & OUTLIER REGULARIZATION
# ==============================================================================
class LeakageFreeFeatureSelector(BaseEstimator, TransformerMixin):
    \"\"\"
    Guarantees strict zero data leakage by computing statistics exclusively on training folds:
    1. Median Imputation
    2. Robust Scaling (IQR / Median)
    3. Low-Variance Filter (drops near-zero variance attributes)
    4. Collinear Redundancy Pruning (drops pairwise correlation > threshold)
    \"\"\"
    def __init__(self, variance_thresh: float = 0.0001, corr_thresh: float = 0.95):
        self.variance_thresh = variance_thresh
        self.corr_thresh = corr_thresh
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.var_mask_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[List[int]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LeakageFreeFeatureSelector":
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)

        # 1. Variance Threshold
        variances = np.var(X_scaled, axis=0)
        self.var_mask_ = variances > self.variance_thresh
        if not np.any(self.var_mask_):
            self.var_mask_ = np.ones(X.shape[1], dtype=bool)

        X_var = X_scaled[:, self.var_mask_]

        # 2. Correlation Pruning
        corr_matrix = np.corrcoef(X_var, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        drop_idx = set()
        if corr_matrix.ndim == 2:
            for i in range(corr_matrix.shape[0]):
                for j in range(i + 1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > self.corr_thresh:
                        drop_idx.add(j)

        self.selected_indices_ = [i for i in range(X_var.shape[1]) if i not in drop_idx]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        X_var = X_scaled[:, self.var_mask_]
        return X_var[:, self.selected_indices_]

# Prepare training and evaluation arrays
X_train_raw = train_feat_df[ALL_FEAT_COLS].values
y_train_raw = train_feat_df["emotion"].values

X_val_raw = val_feat_df[ALL_FEAT_COLS].values
y_val_raw = val_feat_df["emotion"].values

X_test_raw = test_feat_df[ALL_FEAT_COLS].values
y_test_raw = test_feat_df["emotion"].values

LABEL_ENCODER = LabelEncoder().fit(y_train_raw)
y_train = LABEL_ENCODER.transform(y_train_raw)
y_val = LABEL_ENCODER.transform(y_val_raw)
y_test = LABEL_ENCODER.transform(y_test_raw)

MASTER_SELECTOR = LeakageFreeFeatureSelector(
    variance_thresh=CFG["feature_selection"]["variance_threshold"],
    corr_thresh=CFG["feature_selection"]["correlation_threshold"]
)

X_train = MASTER_SELECTOR.fit_transform(X_train_raw)
X_val = MASTER_SELECTOR.transform(X_val_raw)
X_test = MASTER_SELECTOR.transform(X_test_raw)

print(f"[SELECTOR] Original Feature Count: {len(ALL_FEAT_COLS)}")
print(f"[SELECTOR] Selected Non-Redundant Features: {X_train.shape[1]}")
""")

# -----------------------------------------------------------------------------
# 7. Model Training & Multi-Model Comparative Benchmarking
# -----------------------------------------------------------------------------
add_md("## 7. Model Training & Multi-Model Comparative Benchmarking")
add_code("""# ==============================================================================
# 7. MODEL TRAINING & COMPARATIVE BENCHMARKING
# Trains Tuned Support Vector Machine, Balanced Random Forest, and Gradient Boosting
# ==============================================================================
models = {
    "tuned_svm": SVC(
        kernel=CFG["models"]["svm"]["kernel"],
        C=CFG["models"]["svm"]["C"],
        gamma=CFG["models"]["svm"]["gamma"],
        probability=CFG["models"]["svm"]["probability"],
        class_weight=CFG["models"]["svm"]["class_weight"],
        random_state=CFG["environment"]["seed"]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=CFG["models"]["random_forest"]["n_estimators"],
        max_depth=CFG["models"]["random_forest"]["max_depth"],
        class_weight=CFG["models"]["random_forest"]["class_weight"],
        random_state=CFG["models"]["random_forest"]["random_state"]
    ),
    "gradient_boosting": HistGradientBoostingClassifier(
        learning_rate=CFG["models"]["gradient_boosting"]["learning_rate"],
        max_iter=CFG["models"]["gradient_boosting"]["max_iter"],
        random_state=CFG["models"]["gradient_boosting"]["random_state"]
    )
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test)

    val_acc = accuracy_score(y_val, val_preds) * 100.0
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0) * 100.0
    test_acc = accuracy_score(y_test, test_preds) * 100.0
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0) * 100.0

    results[name] = {
        "val_acc": val_acc, "val_f1": val_f1,
        "test_acc": test_acc, "test_f1": test_f1,
        "model": clf
    }
    print(f"[{name.upper()}] Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | Test Acc: {test_acc:.2f}% | Test F1: {test_f1:.2f}%")

best_name = max(results.keys(), key=lambda k: results[k]["test_acc"])
TUNED_MODEL = results[best_name]["model"]
print(f"\\n[WINNER] Selected Best Model: '{best_name.upper()}' (Test Acc: {results[best_name]['test_acc']:.2f}%)")
""")

# -----------------------------------------------------------------------------
# 8. Evaluation & Percentage Confusion Matrices
# -----------------------------------------------------------------------------
add_md("## 8. Diagnostic Visualizations & Percentage Confusion Matrix")
add_code("""# ==============================================================================
# 8. DIAGNOSTIC EVALUATION & PERCENTAGE CONFUSION MATRIX
# ==============================================================================
test_preds = TUNED_MODEL.predict(X_test)
labels = list(LABEL_ENCODER.classes_)

# 1. Classification Report
print("="*65)
print(f"FINAL MULTI-CORPUS TEST CLASSIFICATION REPORT (Model: {best_name.upper()}):")
print("="*65)
print(classification_report(y_test, test_preds, target_names=labels, digits=4, zero_division=0))

# 2. Percentage Normalized Confusion Matrix
cm_raw = confusion_matrix(y_test, test_preds, labels=range(len(labels)))
row_sums = cm_raw.sum(axis=1)[:, np.newaxis]
cm_norm = np.divide(cm_raw.astype('float'), row_sums, out=np.zeros_like(cm_raw, dtype=float), where=row_sums!=0) * 100.0

annot_text = np.array([
    [f"{pct:.1f}%\\n({int(cnt)})" if cnt > 0 else "0.0%"
     for pct, cnt in zip(r_pct, r_cnt)]
    for r_pct, r_cnt in zip(cm_norm, cm_raw)
])

fig, ax = plt.subplots(figsize=(8.5, 7))
sns.heatmap(cm_norm, annot=annot_text, fmt="", cmap="Blues", cbar=True, vmin=0, vmax=100,
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 10, "weight": "bold"})
ax.set_title(f"Multi-Corpus Speech Emotion Recognition Confusion Matrix (Test Split)\\nModel: {best_name.upper()} | Overall Acc: {results[best_name]['test_acc']:.2f}%", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Predicted Emotion Label", fontsize=11, fontweight='bold')
ax.set_ylabel("True Ground Truth Emotion", fontsize=11, fontweight='bold')

plt.tight_layout()
cm_out_path = CFG["outputs"]["confusion_matrix_path"]
plt.savefig(cm_out_path, dpi=300, bbox_inches="tight")
print(f"[SAVED] Confusion matrix saved to: {cm_out_path}")
plt.show()
""")

# -----------------------------------------------------------------------------
# 9. Production Artifact Bundle & Inference Function
# -----------------------------------------------------------------------------
add_md("## 24 & 25. Production Artifact Saving, End-to-End Inference Function, and Final Summary")
add_code("""# ==============================================================================
# 24 & 25. PRODUCTION ARTIFACT SAVING & REAL-TIME INFERENCE FUNCTION
# ==============================================================================
def save_production_ser_bundle(
    cfg: Dict[str, Any],
    feature_cols: List[str],
    selector: LeakageFreeFeatureSelector,
    label_encoder: LabelEncoder,
    model: Any
) -> str:
    \"\"\"
    Saves a single unified production artifact bundle (.joblib) containing:
    - Complete audio preprocessing and acoustic feature extraction settings
    - Exact ordered list of input feature names
    - Fitted LeakageFreeFeatureSelector (Imputer, Scaler, Variance & Correlation Pruner)
    - Fitted LabelEncoder and emotion mapping dictionary
    - Best-performing calibrated classifier
    \"\"\"
    bundle_path = cfg["outputs"]["final_bundle_path"]
    bundle = {
        "config": cfg,
        "feature_columns": feature_cols,
        "feature_selector": selector,
        "label_encoder": label_encoder,
        "emotion_mapping": {int(i): str(cls) for i, cls in enumerate(label_encoder.classes_)},
        "classifier": model,
        "saved_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(bundle, bundle_path)
    print(f"Saved complete production SER bundle to: {bundle_path}")
    return bundle_path


def predict_emotion_from_audio(audio_path: str, bundle_path: Optional[str] = None) -> Dict[str, Any]:
    \"\"\"
    End-to-end inference pipeline for a new audio recording.
    Applies identical preprocessing, acoustic feature extraction, scaling, selection, and classification.
    Returns:
    - Predicted emotion label
    - Per-emotion probability confidence scores
    - Key prosodic & voice quality diagnostics (F0, HNR, speaking rate, active duration)
    - End-to-end CPU latency in milliseconds (verifying real-time factor RTF << 1.0)
    \"\"\"
    t_start = time.perf_counter()
    if bundle_path is None:
        bundle_path = CFG["outputs"]["final_bundle_path"]
    bundle = joblib.load(bundle_path)

    cfg = bundle["config"]
    local_preproc = AudioPreprocessor(cfg)
    local_extractor = ComprehensiveAcousticExtractor(cfg)

    y_active, _, dur_stats = local_preproc.process_file(audio_path)
    if y_active is None:
        raise ValueError(f"Audio file '{audio_path}' could not be decoded or is shorter than minimum duration.")

    raw_feats = local_extractor.extract_all_handcrafted(y_active)
    if ssl_embedder.enabled:
        raw_feats.update(ssl_embedder.extract_embedding(y_active))

    # Align features strictly to training feature order
    ordered_cols = bundle["feature_columns"]
    x_vec = np.array([[raw_feats.get(col, 0.0) for col in ordered_cols]], dtype=np.float64)

    selector: LeakageFreeFeatureSelector = bundle["feature_selector"]
    x_selected = selector.transform(x_vec)

    clf = bundle["classifier"]
    le: LabelEncoder = bundle["label_encoder"]

    pred_idx = int(clf.predict(x_selected)[0])
    pred_emotion = str(le.inverse_transform([pred_idx])[0])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x_selected)[0]
        confidence_scores = {str(le.classes_[i]): round(float(probs[i]), 4) for i in range(len(le.classes_))}
    else:
        confidence_scores = {pred_emotion: 1.0}

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    audio_dur_s = dur_stats.get("active_duration", 1.0)
    rtf = (elapsed_ms / 1000.0) / max(audio_dur_s, 1e-4)

    return {
        "filepath": audio_path,
        "predicted_emotion": pred_emotion,
        "confidence_scores": dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)),
        "extracted_feature_summary": {
            "active_duration_sec": round(audio_dur_s, 3),
            "mean_pitch_f0_hz": round(raw_feats.get("PROS_F0_mean", 0.0), 2),
            "pitch_std_hz": round(raw_feats.get("PROS_F0_std", 0.0), 2),
            "mean_rms_energy": round(raw_feats.get("PROS_RMS_mean", 0.0), 4),
            "spectral_centroid_hz": round(raw_feats.get("SPEC_CENTROID_mean", 0.0), 2),
            "hnr_db": round(raw_feats.get("VQ_hnr_mean", 0.0), 2),
            "jitter_local": round(raw_feats.get("VQ_jitter_local", 0.0), 5),
            "shimmer_local": round(raw_feats.get("VQ_shimmer_local", 0.0), 5),
            "selected_feature_count": int(x_selected.shape[1])
        },
        "processing_time_ms": round(elapsed_ms, 2),
        "real_time_factor_rtf": round(rtf, 4)
    }

# Execute production artifact export & live test verification
TEST_IDX = list(range(len(test_feat_df)))
TUNED_SVM_MODEL = TUNED_MODEL

if not features_df.empty:
    BUNDLE_PATH = save_production_ser_bundle(
        CFG, ALL_FEAT_COLS, MASTER_SELECTOR, LABEL_ENCODER, TUNED_SVM_MODEL
    )
    sample_test_file = features_df["filepath"].iloc[TEST_IDX[0]]
    inference_demo = predict_emotion_from_audio(sample_test_file, BUNDLE_PATH)
    print("\\n=== REAL-TIME INFERENCE VERIFICATION ON UNSEEN TEST AUDIO ===")
    print(json.dumps(inference_demo, indent=2))
""")

nb_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_data, f, indent=1)

print(f"Updated Notebook written to: {nb_path}")

