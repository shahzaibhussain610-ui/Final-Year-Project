"""
Facial Emotion Recognition (FER) - Standalone Training Script
Runs experiments directly from config.yaml without Jupyter notebook caching issues.
"""

import os
import sys
import math
import random
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import timm

# 1. LOAD CONFIG
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Could not find config: {CONFIG_PATH}")

with CONFIG_PATH.open(encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

def _project_path(value):
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

ds_cfg = CONFIG["datasets"]
PATHS = {
    "affectnet": _project_path(ds_cfg["affectnet"]["root"]),
    "fer2013_folders": _project_path(ds_cfg["fer2013"]["root"]),
    "fer2013_csv": _project_path(ds_cfg["fer2013"]["csv"]["file"]),
    "ckplus": _project_path(ds_cfg["ckplus"]["root"]) / ds_cfg["ckplus"]["csv_extended"]["file"],
    "ravdess": _project_path(ds_cfg["ravdess"]["root"]),
}

CANONICAL_CLASSES = list(CONFIG["label_mapping"]["canonical_classes"])
CLASS2IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

# Device detection with compatibility test for RTX 5090 (sm_120)
cfg_dev = CONFIG["environment"]["device"]
if cfg_dev == "auto":
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            DEVICE = torch.device("cuda")
        except Exception:
            print("Notice: GPU detected but current PyTorch build lacks sm_120 kernels. Using CPU.")
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(cfg_dev)

print(f"Project root : {PROJECT_ROOT}")
print(f"Device       : {DEVICE}")
print(f"Taxonomy     : {CANONICAL_CLASSES}")

# 2. DATA LOADERS
AFFECTNET_LABEL_MAP = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise", 4: "fear", 5: "disgust", 6: "angry", 7: None}
FER2013_CSV_MAP = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
CKPLUS_CSV_MAP = {0: "neutral", 1: "angry", 2: None, 3: "disgust", 4: "fear", 5: "happy", 6: "sad", 7: "surprise"}

def load_affectnet(root, split="train", max_samples=None):
    split_name = ds_cfg["affectnet"]["json_per_image"]["train_dir"] if split == "train" else ds_cfg["affectnet"]["json_per_image"]["val_dir"]
    split_dir = root / split_name
    images_dir = split_dir / ds_cfg["affectnet"]["json_per_image"]["images_subdir"]
    annot_dir = split_dir / ds_cfg["affectnet"]["json_per_image"]["annotations_subdir"]
    if not annot_dir.exists():
        return []
    import json
    samples = []
    for ann_file in sorted(annot_dir.glob("*.json")):
        img_path = images_dir / (ann_file.stem + ".jpg")
        if not img_path.exists():
            continue
        try:
            ann = json.loads(ann_file.read_text(encoding="utf-8"))
            lc = ann.get(ds_cfg["affectnet"]["json_per_image"]["label_field"], -1)
            code = int(lc[0]) if isinstance(lc, (list, tuple)) else int(lc)
        except Exception:
            continue
        canonical = AFFECTNET_LABEL_MAP.get(code)
        if canonical is None or canonical not in CLASS2IDX:
            continue
        samples.append((str(img_path), CLASS2IDX[canonical]))
        if max_samples and len(samples) >= max_samples:
            break
    print(f"  [AffectNet {split}] {len(samples)} samples loaded")
    return samples

def load_fer2013_folders(root, split="train", max_samples=None):
    folder_map = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "sad": "sad", "surprise": "surprise", "neutral": "neutral"}
    split_dir = root / split
    if not split_dir.exists():
        return []
    samples = []
    for cd in sorted(split_dir.iterdir()):
        if not cd.is_dir():
            continue
        c = folder_map.get(cd.name.lower())
        if c is None:
            continue
        for img_file in sorted(cd.glob("*")):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append((str(img_file), CLASS2IDX[c]))
    if max_samples and len(samples) > max_samples:
        random.seed(CONFIG["environment"]["seed"])
        samples = random.sample(samples, max_samples)
    print(f"  [FER2013-Folders {split}] {len(samples)} samples loaded")
    return samples

def _load_pixel_csv(csv_path, label_map, split="train", train_u="Training", test_u="PrivateTest", max_samples=None, name="CSV"):
    df = pd.read_csv(csv_path)
    if "Usage" in df.columns:
        u = train_u if split == "train" else test_u
        df = df[df["Usage"] == u]
    else:
        split_idx = int(len(df) * 0.8)
        df = df.iloc[:split_idx] if split == "train" else df.iloc[split_idx:]
    samples = []
    for _, row in df.iterrows():
        c = label_map.get(int(row["emotion"]))
        if c is None or c not in CLASS2IDX:
            continue
        try:
            pxs = list(map(int, str(row["pixels"]).split()))
            sz = int(np.sqrt(len(pxs)))
            if sz * sz != len(pxs):
                continue
            img = np.array(pxs, dtype=np.uint8).reshape(sz, sz)
        except Exception:
            continue
        samples.append((img, CLASS2IDX[c]))
        if max_samples and len(samples) >= max_samples:
            break
    print(f"  [{name} {split}] {len(samples)} samples loaded")
    return samples

def load_fer2013_csv(csv_path, split="train", max_samples=None):
    uv = ds_cfg["fer2013"]["csv"].get("usage_values", {})
    return _load_pixel_csv(csv_path, FER2013_CSV_MAP, split, uv.get("train", "Training"), uv.get("test", "PrivateTest"), max_samples, "FER2013-CSV")

def load_ckplus_csv(csv_path, split="train", max_samples=None):
    uv = ds_cfg["ckplus"]["csv_extended"].get("usage_values", {})
    return _load_pixel_csv(csv_path, CKPLUS_CSV_MAP, split, uv.get("train", "Training"), uv.get("test", "PrivateTest"), max_samples, "CK+")

# 3. TRANSFORMS & DATASET
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class SafeRandomErasing:
    def __init__(self, p=0.25, scale=(0.02, 0.1)):
        self.eraser = transforms.RandomErasing(p=p, scale=scale)
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.functional.to_tensor(img)
        return self.eraser(img)

def get_transforms(input_size, split, augmentation="standard", normalization="imagenet"):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalization == "imagenet" else transforms.Lambda(lambda x: x)
    aug = {
        "none": [],
        "light": [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ],
        "standard": [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        ],
    }.get(augmentation, [])

    if split == "train":
        tf_list = [transforms.Resize((input_size, input_size))] + aug + [transforms.ToTensor()]
        if augmentation == "standard":
            tf_list.append(SafeRandomErasing(p=0.25, scale=(0.02, 0.1)))
        tf_list.append(norm)
    else:
        tf_list = [transforms.Resize((input_size, input_size)), transforms.ToTensor(), norm]
    return transforms.Compose(tf_list)

class FERDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        if isinstance(sample, np.ndarray):
            img = Image.fromarray(sample, mode='L').convert('RGB')
        elif isinstance(sample, str):
            try:
                img = Image.open(sample).convert('RGB')
            except Exception:
                img = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
        else:
            img = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        if not isinstance(img, torch.Tensor):
            img = transforms.functional.to_tensor(img)
        return img, label

def make_loaders(train_data, test_data, input_size, batch_size, augmentation="standard", normalization="imagenet", num_workers=0):
    train_ds = FERDataset([(s, l) for s, l, _ in train_data], get_transforms(input_size, "train", augmentation, normalization))
    test_ds = FERDataset([(s, l) for s, l, _ in test_data], get_transforms(input_size, "test", "none", normalization))
    pin = (DEVICE.type == "cuda")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
        DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=num_workers, pin_memory=pin),
    )

# 4. MODEL
class FERModel(nn.Module):
    def __init__(self, timm_name, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        in_feat = self.backbone.num_features
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, num_classes))
        self.backbone_frozen = False
    def forward(self, x):
        return self.classifier(self.backbone(x))
    def freeze_backbone(self):
        self.backbone_frozen = True
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
    def unfreeze_backbone(self):
        self.backbone_frozen = False
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()
    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "backbone_frozen", False):
            self.backbone.eval()
        return self

# 5. TRAINING ENGINE
def compute_class_weights(labels, n):
    c = np.bincount(labels, minlength=n).astype(float)
    c = np.where(c == 0, 1, c)
    return torch.tensor(c.sum() / (n * c), dtype=torch.float32)

def get_loss_fn(name, weights=None, label_smoothing=0.0, focal_gamma=2.0):
    if name == "focal":
        def fl(logits, targets):
            ce = F.cross_entropy(logits, targets, weight=weights, reduction='none')
            return ((1 - torch.exp(-ce)) ** focal_gamma * ce).mean()
        return fl
    elif name == "weighted_cross_entropy":
        return lambda logits, t: F.cross_entropy(logits, t, weight=weights, label_smoothing=label_smoothing)
    else:
        return lambda logits, t: F.cross_entropy(logits, t, label_smoothing=label_smoothing)

def build_optimizer(model, cfg):
    groups = [
        {"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": cfg["lr"]},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg["backbone_lr"]},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    opt_name = cfg.get("optimizer", "adamw").lower()
    wd = cfg.get("weight_decay", 0.05)
    if opt_name == "adam":
        return torch.optim.Adam(groups, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.get("momentum", 0.9), nesterov=cfg.get("nesterov", True), weight_decay=wd)
    return torch.optim.AdamW(groups, weight_decay=wd)

def cosine_warmup_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate(model, loader, n_classes, loss_fn=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels_dev = labels.to(DEVICE)
        logits = model(imgs)
        if loss_fn is not None:
            total_loss += loss_fn(logits, labels_dev).item()
            n_batches += 1
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    ap, al = np.array(all_preds), np.array(all_labels)
    res = {
        "accuracy": accuracy_score(al, ap),
        "macro_f1": f1_score(al, ap, average='macro', zero_division=0),
        "per_class_f1": f1_score(al, ap, average=None, zero_division=0, labels=list(range(n_classes))).tolist(),
        "preds": ap, "labels": al,
    }
    if loss_fn is not None and n_batches > 0:
        res["loss"] = total_loss / n_batches
    return res

def train_model(model_name, timm_name, input_size, train_data, test_data, cfg, n_classes=7, save_dir=None):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    pretrained = cfg.get("pretrained", True)
    dropout = cfg.get("dropout", 0.3)
    model = FERModel(timm_name, n_classes, pretrained=pretrained, dropout=dropout).to(DEVICE)

    bs = cfg.get("batch_size_override", cfg["batch_size"])
    train_loader, test_loader = make_loaders(
        train_data, test_data, input_size, bs,
        augmentation=CONFIG["augmentation"]["preset"],
        normalization=CONFIG["preprocessing"]["normalization"],
        num_workers=cfg.get("num_workers", 0)
    )

    cw = compute_class_weights([l for _, l, _ in train_data], n_classes).to(DEVICE)
    loss_weights = cw if (cfg["loss"] == "weighted_cross_entropy" or (cfg["loss"] == "focal" and cfg.get("focal_use_class_weights", False))) else None
    loss_fn = get_loss_fn(cfg["loss"], weights=loss_weights, label_smoothing=cfg.get("label_smoothing", 0.0), focal_gamma=cfg.get("focal_gamma", 2.0))

    if save_dir is None:
        save_dir = _project_path(CONFIG["paths"]["checkpoint_dir"]) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}
    sel_metric_name = cfg.get("selection_metric", "val_macro_f1")
    best_metric = float("inf") if sel_metric_name == "val_loss" else -1.0
    patience_ctr, best_state = 0, None

    total_epochs = cfg["epochs"]
    mode = cfg.get("mode", "head_then_full")
    head_epochs = cfg.get("head_epochs", 2) if mode == "head_then_full" else 0
    if mode == "head_only":
        head_epochs = total_epochs

    steps_per_ep = len(train_loader)

    def setup_phase(phase_name, phase_epochs):
        if phase_name == "HEAD":
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        optimizer = build_optimizer(model, cfg)
        total_phase_steps = phase_epochs * steps_per_ep
        warmup_phase_steps = min(cfg.get("warmup_epochs", 1) * steps_per_ep, total_phase_steps)
        sched = cosine_warmup_scheduler(optimizer, total_steps=total_phase_steps, warmup_steps=warmup_phase_steps, min_lr_ratio=cfg.get("min_lr_ratio", 0.01))
        return optimizer, sched

    if head_epochs > 0:
        current_phase = "HEAD"
        optimizer, sched = setup_phase("HEAD", head_epochs)
    else:
        current_phase = "FULL"
        optimizer, sched = setup_phase("FULL", total_epochs)

    for epoch in range(total_epochs):
        if mode == "head_then_full" and head_epochs > 0 and epoch == head_epochs:
            current_phase = "FULL"
            remaining_epochs = total_epochs - head_epochs
            optimizer, sched = setup_phase("FULL", remaining_epochs)

        model.train()
        ep_loss, n_bat = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            if cfg.get("gradient_clip", 0.0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            sched.step()
            ep_loss += loss.item()
            n_bat += 1

        avg_loss = ep_loss / max(1, n_bat)
        cur_lr = optimizer.param_groups[0]["lr"]

        val = evaluate(model, test_loader, n_classes, loss_fn=loss_fn)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val.get("loss", 0.0))
        history["val_acc"].append(val["accuracy"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["lr"].append(cur_lr)

        print(f"  [{model_name}|{current_phase}] Ep {epoch+1:3d}/{total_epochs} "
              f"loss={avg_loss:.4f} val_loss={val.get('loss', 0.0):.4f} "
              f"acc={val['accuracy']:.4f} f1={val['macro_f1']:.4f} lr={cur_lr:.2e}")

        if sel_metric_name == "val_loss":
            sel_metric = val.get("loss", avg_loss)
            is_better = (best_state is None) or (sel_metric < best_metric - cfg.get("early_stopping_min_delta", 0.001))
        elif sel_metric_name == "val_accuracy":
            sel_metric = val["accuracy"]
            is_better = (best_state is None) or (sel_metric > best_metric + cfg.get("early_stopping_min_delta", 0.001))
        else:
            sel_metric = val["macro_f1"]
            is_better = (best_state is None) or (sel_metric > best_metric + cfg.get("early_stopping_min_delta", 0.001))

        if is_better:
            best_metric = sel_metric
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_dir / "best.pt")
        else:
            patience_ctr += 1
            if cfg.get("early_stopping_enabled", True) and patience_ctr >= cfg.get("early_stopping_patience", 6):
                print(f"  Early stopping triggered at epoch {epoch+1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, test_loader, n_classes, loss_fn=loss_fn)
    return {
        "model_name": model_name, "timm_name": timm_name, "history": history,
        "best_val_metric": best_metric, "test_accuracy": final["accuracy"],
        "test_macro_f1": final["macro_f1"], "test_per_class_f1": final["per_class_f1"],
        "checkpoint": str(save_dir / "best.pt"),
    }

t_cfg = CONFIG["training"]
TRAIN_CFG = {
    "mode": t_cfg.get("mode", "head_then_full"),
    "epochs": t_cfg.get("epochs", 30),
    "head_epochs": t_cfg.get("head_epochs", 2),
    "batch_size": CONFIG["environment"]["batch_size"],
    "lr": t_cfg["optimizer"]["lr"],
    "backbone_lr": t_cfg["optimizer"].get("backbone_lr", 1e-4),
    "weight_decay": t_cfg["optimizer"]["weight_decay"],
    "optimizer": t_cfg["optimizer"]["name"],
    "momentum": t_cfg["optimizer"].get("momentum", 0.9),
    "nesterov": t_cfg["optimizer"].get("nesterov", True),
    "warmup_epochs": t_cfg["scheduler"].get("warmup_epochs", 1),
    "min_lr_ratio": t_cfg["scheduler"].get("min_lr_ratio", 0.01),
    "early_stopping_enabled": t_cfg.get("early_stopping", {}).get("enabled", True),
    "early_stopping_patience": t_cfg.get("early_stopping", {}).get("patience", 6),
    "early_stopping_min_delta": t_cfg.get("early_stopping", {}).get("min_delta", 0.001),
    "selection_metric": t_cfg.get("selection_metric", "val_macro_f1"),
    "label_smoothing": CONFIG["loss"].get("label_smoothing", 0.0),
    "loss": CONFIG["loss"]["name"],
    "focal_gamma": CONFIG["loss"].get("focal_gamma", 2.0),
    "gradient_clip": CONFIG["environment"]["gradient_clipping"]["max_norm"],
    "seed": CONFIG["environment"]["seed"],
    "num_workers": 0,
}

if __name__ == "__main__":
    print("\n=== LOADING DATASETS ===")
    max_s = t_cfg.get("max_samples_per_dataset")

    print("\n=== STARTING TRAINING ===")
    res = train_model(
        model_name="efficientnet_b0",
        timm_name="efficientnet_b0",
        input_size=224,
        train_data=ALL_TRAIN,
        test_data=ALL_TEST,
        cfg=TRAIN_CFG,
        n_classes=len(CANONICAL_CLASSES)
    )
    print(f"\nTraining Complete! Best Test Accuracy: {res['test_accuracy']:.4f}, Test Macro F1: {res['test_macro_f1']:.4f}")
