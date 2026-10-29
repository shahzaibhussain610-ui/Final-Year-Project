"""
Patch the infrastructure cell in-place to fix:
  1. FER2013/CK+/RAVDESS val-split carving (no 'val/' folder exists)
  2. Deprecated torch.cuda.amp -> torch.amp API
  3. autocast device_type awareness
"""
import json, re, sys
sys.stdout.reconfigure(encoding='utf-8')

NB = r"c:\Users\Shahzaib Hussain\Desktop\Shahzaib Hussain\FYP\New Notebook\fer_research\notebooks\facial_emotion_research.ipynb"

nb = json.load(open(NB, encoding='utf-8'))

# ── Find the infrastructure code cell by its id ──────────────
infra_idx = None
for i, c in enumerate(nb['cells']):
    if c.get('id') == 'exp_infra_code':
        infra_idx = i
        break

if infra_idx is None:
    print("ERROR: could not find exp_infra_code cell")
    sys.exit(1)

print(f"Found infra cell at index {infra_idx}")

# ── Build the complete replacement source ────────────────────
NEW_SOURCE = r"""# ============================================================
# Experiment runner infrastructure  (GPU + config-driven)
# ============================================================
import copy, json, math, time, random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import pandas as pd
import timm

# ── AMP helpers (new torch.amp API, backward-compat wrapper) ──
USE_AMP = False
AMP_DTYPE = torch.float16

def _make_autocast():
    """Return the right autocast context for the current device."""
    if not USE_AMP:
        return torch.amp.autocast(device_type=device.type, enabled=False)
    return torch.amp.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=True)

def _make_scaler():
    """Return a GradScaler (CUDA only; no-op otherwise)."""
    if device.type == "cuda" and USE_AMP:
        return torch.amp.GradScaler('cuda')
    return torch.amp.GradScaler('cpu', enabled=False)

# ──────────────────────────────────────────────────────────────
# GPU / mixed-precision config
# ──────────────────────────────────────────────────────────────
mp_cfg = CONFIG["environment"]["mixed_precision"]
_amp_enabled = mp_cfg["enabled"]
USE_AMP = (
    device.type == "cuda"
    if _amp_enabled == "auto"
    else bool(_amp_enabled)
)
AMP_DTYPE = torch.float16 if mp_cfg.get("dtype", "float16") == "float16" else torch.bfloat16
print(f"Mixed precision: {USE_AMP}  |  dtype: {AMP_DTYPE}  |  device: {device}")

# ──────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────
def set_seed(seed):
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = CONFIG["environment"]["deterministic"]
        torch.backends.cudnn.benchmark = (
            False if CONFIG["environment"]["deterministic"]
            else CONFIG["environment"]["cudnn_benchmark"]
        )

# ──────────────────────────────────────────────────────────────
# Dataset loading helpers
# Keys understood: "affectnet", "fer2013", "ckplus", "ravdess"
# Splits: "train" | "val" | "test"
# ──────────────────────────────────────────────────────────────
def _split_off_val(samples, val_frac, seed):
    """Carve val_frac of samples as validation; return (train, val)."""
    _random.seed(seed)
    idx = list(range(len(samples)))
    _random.shuffle(idx)
    n_val = max(1, int(len(samples) * val_frac))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]
    val   = [samples[i] for i in val_idx]
    train = [samples[i] for i in train_idx]
    return train, val

# Cache so we only call load_fer2013_folders once per max_s value
_FER_TRAIN_CACHE  = {}
_CKP_TRAIN_CACHE  = {}
_RAV_TRAIN_CACHE  = {}

def _load_split(dataset_name, split, max_s):
    """
    Return list of (sample, label) tuples.
    Handles the val-split carve for datasets without an explicit val folder.
    """
    split_seed = CONFIG["splits"]["seed"]

    if dataset_name == "affectnet":
        # AffectNet has real train_set / validation_set folders
        return load_affectnet(PATHS["affectnet"], split, max_s)

    elif dataset_name == "fer2013":
        # FER2013 folders: has train/ and test/ but NO val/
        # val = val_fraction_from_train carved out of train
        fer_val_frac = ds_cfg["fer2013"]["folders"].get("val_fraction_from_train", 0.1)
        cache_key = max_s
        if cache_key not in _FER_TRAIN_CACHE:
            all_tr = load_fer2013_folders(PATHS["fer2013_folders"], "train", max_s)
            tr, va = _split_off_val(all_tr, fer_val_frac, split_seed)
            _FER_TRAIN_CACHE[cache_key] = {"train": tr, "val": va}
        if split == "train":
            return _FER_TRAIN_CACHE[cache_key]["train"]
        elif split == "val":
            return _FER_TRAIN_CACHE[cache_key]["val"]
        else:  # test
            return load_fer2013_folders(PATHS["fer2013_folders"], "test", max_s)

    elif dataset_name == "ckplus":
        # CK+ csv: Usage column may exist (Training/PrivateTest) or may need 80/20 split
        cache_key = max_s
        if cache_key not in _CKP_TRAIN_CACHE:
            all_tr = load_ckplus_csv(PATHS["ckplus"], "train", max_s)
            all_te = load_ckplus_csv(PATHS["ckplus"], "test",  max_s)
            if all_tr:
                tr, va = _split_off_val(all_tr, 0.15, split_seed)
            else:
                tr, va = [], []
            _CKP_TRAIN_CACHE[cache_key] = {"train": tr, "val": va, "test": all_te}
        return _CKP_TRAIN_CACHE[cache_key].get(split, [])

    elif dataset_name == "ravdess":
        # RAVDESS: actor holdout split (already implemented in load_ravdess)
        return load_ravdess(PATHS["ravdess"], split, max_s)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_data(train_datasets, val_datasets, max_s=None):
    """Combine multiple datasets for train / val splits."""
    train_data, val_data = [], []
    for ds in train_datasets:
        samples = _load_split(ds, "train", max_s)
        train_data.extend([(s, l, ds) for s, l in samples])
    for ds in val_datasets:
        samples = _load_split(ds, "val", max_s)
        val_data.extend([(s, l, ds) for s, l in samples])
    return train_data, val_data


# ──────────────────────────────────────────────────────────────
# Augmentation helpers (config-driven)
# ──────────────────────────────────────────────────────────────
def _make_augmentation_list(aug_name, input_size):
    aug_cfg = CONFIG["augmentation"]["presets"].get(aug_name, {})
    aug_list = []
    if not aug_cfg:
        return aug_list
    if "horizontal_flip" in aug_cfg:
        aug_list.append(transforms.RandomHorizontalFlip(aug_cfg["horizontal_flip"]["p"]))
    if "random_resized_crop" in aug_cfg:
        c = aug_cfg["random_resized_crop"]
        aug_list.append(transforms.RandomResizedCrop(
            input_size, scale=c["scale"], ratio=c["ratio"]))
    if "rotation" in aug_cfg:
        aug_list.append(transforms.RandomRotation(aug_cfg["rotation"]["degrees"]))
    if "affine" in aug_cfg:
        c = aug_cfg["affine"]
        aug_list.append(transforms.RandomAffine(
            0, translate=(c["translate"], c["translate"]),
            scale=c["scale"], shear=c["shear"]))
    if "color_jitter" in aug_cfg:
        c = aug_cfg["color_jitter"]
        aug_list.append(transforms.ColorJitter(
            brightness=c.get("brightness", 0), contrast=c.get("contrast", 0),
            saturation=c.get("saturation", 0), hue=c.get("hue", 0)))
    if "randaugment" in aug_cfg:
        c = aug_cfg["randaugment"]
        try:
            aug_list.append(transforms.RandAugment(
                num_ops=c["num_ops"], magnitude=c["magnitude"]))
        except AttributeError:
            pass  # older torchvision
    return aug_list


def get_transforms_v2(input_size, split, aug_preset="standard", normalization="imagenet"):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    norm = (transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            if normalization == "imagenet"
            else transforms.Lambda(lambda x: x))
    if split == "train":
        aug_list = _make_augmentation_list(aug_preset, input_size)
        tf_list = [transforms.Resize((input_size, input_size))] + aug_list
        tf_list.append(transforms.ToTensor())
        re_cfg = CONFIG["augmentation"]["presets"].get(aug_preset, {}).get("random_erasing", {})
        if re_cfg:
            tf_list.append(transforms.RandomErasing(p=re_cfg["p"], scale=re_cfg["scale"]))
        tf_list.append(norm)
    else:
        tf_list = [transforms.Resize((input_size, input_size)), transforms.ToTensor(), norm]
    return transforms.Compose(tf_list)


# ──────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────
def compute_class_weights(train_data, num_classes, method="inverse_frequency"):
    labels = [l for _, l, _ in train_data]
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1)
    if method == "inverse_frequency":
        w = 1.0 / counts
    elif method == "inverse_sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        w = np.ones(num_classes)
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(loss_cfg, train_data, num_classes, dev):
    name = loss_cfg["name"]
    ls   = loss_cfg.get("label_smoothing", 0.0)
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    elif name == "weighted_cross_entropy":
        w = compute_class_weights(train_data, num_classes,
                                   loss_cfg.get("class_weighting", "inverse_frequency")).to(dev)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    elif name == "focal":
        alpha = None
        if loss_cfg.get("focal_use_class_weights", False):
            alpha = compute_class_weights(train_data, num_classes,
                                          loss_cfg.get("class_weighting", "inverse_frequency")).to(dev)
        return FocalLoss(gamma=loss_cfg.get("focal_gamma", 2.0), alpha=alpha, label_smoothing=ls)
    elif name == "class_balanced":
        beta   = loss_cfg.get("class_balanced_beta", 0.999)
        labels = [l for _, l, _ in train_data]
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        En = (1.0 - beta ** counts) / (1.0 - beta)
        w  = 1.0 / np.maximum(En, 1e-6)
        w  = w / w.sum() * num_classes
        w  = torch.tensor(w, dtype=torch.float32).to(dev)
        base = loss_cfg.get("class_balanced_base", "cross_entropy")
        if base == "focal":
            return FocalLoss(gamma=loss_cfg.get("focal_gamma", 2.0), alpha=w, label_smoothing=ls)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    else:
        raise ValueError(f"Unknown loss: {name}")


# ──────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────
def build_model(model_key, num_classes, cfg, training_mode="head_then_full"):
    arch      = cfg["models"]["architectures"][model_key]
    defaults  = cfg["models"]["defaults"]
    timm_name = arch["timm_name"]
    pretrained = defaults["pretrained"] and (training_mode != "scratch")
    drop_path  = defaults.get("drop_path_rate", 0.0)
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path,
    )
    return model


def freeze_backbone(model):
    """Freeze all params except the final classifier head."""
    for name, param in model.named_parameters():
        if not any(k in name for k in ("classifier", "head", "fc")):
            param.requires_grad_(False)


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad_(True)


def gradual_unfreeze_step(model, step, units_per_step=2):
    children = list(model.children())
    total = len(children)
    keep_frozen = max(0, total - (step + 1) * units_per_step)
    for i, child in enumerate(children):
        for param in child.parameters():
            param.requires_grad_(i >= keep_frozen)


# ──────────────────────────────────────────────────────────────
# Optimiser & Scheduler builders
# ──────────────────────────────────────────────────────────────
def build_optimizer(model, opt_cfg):
    name  = opt_cfg["name"]
    lr    = opt_cfg["lr"]
    wd    = opt_cfg["weight_decay"]
    blr   = opt_cfg.get("backbone_lr") or lr
    head_params, backbone_params = [], []
    for nm, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in nm for k in ("classifier", "head", "fc")):
            head_params.append(p)
        else:
            backbone_params.append(p)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": blr})
    if not param_groups:
        param_groups = list(model.parameters())
    if name == "adamw":
        return optim.AdamW(param_groups, weight_decay=wd,
                           betas=tuple(opt_cfg.get("betas", [0.9, 0.999])))
    elif name == "adam":
        return optim.Adam(param_groups, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(param_groups,
                         momentum=opt_cfg.get("momentum", 0.9),
                         nesterov=opt_cfg.get("nesterov", True),
                         weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, sched_cfg, epochs, steps_per_epoch):
    name = sched_cfg["name"]
    if name == "cosine_warmup":
        warmup_steps = sched_cfg.get("warmup_epochs", 1) * steps_per_epoch
        total_steps  = epochs * steps_per_epoch
        min_lr_ratio = sched_cfg.get("min_lr_ratio", 0.01)
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            prog = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size_epochs", 10),
            gamma=sched_cfg.get("gamma", 0.1))
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=sched_cfg.get("plateau_patience", 3),
            factor=sched_cfg.get("gamma", 0.1),
            mode="max")
    else:
        return None


# ──────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, dev, num_classes):
    if len(loader.dataset) == 0:
        print("  Warning: val loader is empty; skipping evaluation.")
        return {"accuracy": 0.0, "macro_f1": 0.0}
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(dev, non_blocking=True)
        with _make_autocast():
            logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    if not all_preds:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        zero_division=0, labels=list(range(num_classes)))
    return {"accuracy": acc, "macro_f1": macro_f1}


# ──────────────────────────────────────────────────────────────
# Core training loop
# ──────────────────────────────────────────────────────────────
def train_experiment(exp_id, model_key, train_data, val_data, cfg_overrides=None):
    """
    Train one experiment run.
    Returns (result_dict, best_model).
    """
    # ── merge overrides ──────────────────────────────────────
    cfg = copy.deepcopy(CONFIG)
    if cfg_overrides:
        for dotted_key, value in cfg_overrides.items():
            if dotted_key.startswith("_"):
                continue  # internal flags
            keys = dotted_key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

    num_classes   = len(cfg["label_mapping"]["canonical_classes"])
    t_cfg         = cfg["training"]
    opt_cfg       = t_cfg["optimizer"]
    sched_cfg     = t_cfg["scheduler"]
    es_cfg        = t_cfg.get("early_stopping", {})
    loss_cfg      = cfg["loss"]
    aug_preset    = cfg["augmentation"]["preset"]
    norm_mode     = cfg["preprocessing"]["normalization"]
    training_mode = t_cfg.get("mode", "head_then_full")
    epochs        = t_cfg["epochs"]
    head_epochs   = t_cfg.get("head_epochs", 2)
    bs            = cfg["environment"]["batch_size"]
    arch_cfg      = cfg["models"]["architectures"].get(model_key, {})
    bs            = arch_cfg.get("batch_size", bs)
    input_size    = arch_cfg.get("input_size", 224)
    gc_norm       = (cfg["environment"]["gradient_clipping"]["max_norm"]
                     if cfg["environment"]["gradient_clipping"]["enabled"] else 0.0)
    sel_metric    = t_cfg.get("selection_metric", "val_macro_f1")
    ema_enabled   = cfg["optimization"]["ema"]["enabled"]
    gradual_cfg   = t_cfg.get("gradual_unfreeze", {})
    num_workers   = TRAINING.get("num_workers", 0)

    print(f"\n{'='*60}")
    print(f"Experiment : {exp_id}  |  Model: {model_key}")
    print(f"  Mode     : {training_mode}   |  Epochs: {epochs} (head: {head_epochs})")
    print(f"  BS       : {bs}  |  Input: {input_size}px")
    print(f"  Loss     : {loss_cfg['name']}  |  Aug: {aug_preset}")
    print(f"  Train    : {len(train_data)}  |  Val: {len(val_data)}")
    print(f"{'='*60}")

    set_seed(cfg["environment"]["seed"])

    # ── DataLoaders ──────────────────────────────────────────
    from torch.utils.data import DataLoader
    tr_tf = get_transforms_v2(input_size, "train", aug_preset, norm_mode)
    va_tf = get_transforms_v2(input_size, "val",   "none",     norm_mode)
    train_ds = FERDataset([(s, l) for s, l, _ in train_data], tr_tf)
    val_ds   = FERDataset([(s, l) for s, l, _ in val_data],   va_tf)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    # ── Model ────────────────────────────────────────────────
    model = build_model(model_key, num_classes, cfg, training_mode)
    if training_mode in ("head_only", "head_then_full", "gradual"):
        freeze_backbone(model)
    model = model.to(device)

    ema_model = AveragedModel(model, device=device) if ema_enabled else None

    # ── Loss & optimiser ─────────────────────────────────────
    criterion = build_loss(loss_cfg, train_data, num_classes, device)
    optimizer = build_optimizer(model, opt_cfg)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, sched_cfg, epochs, steps_per_epoch)
    scaler    = _make_scaler()

    # ── Training loop ────────────────────────────────────────
    best_metric = -1.0
    best_state  = None
    es_counter  = 0
    es_patience  = es_cfg.get("patience", 6)
    es_min_delta = es_cfg.get("min_delta", 0.001)
    es_enabled   = es_cfg.get("enabled", True)
    history      = []

    for epoch in range(1, epochs + 1):
        # ── Unfreeze logic ───────────────────────────────────
        if training_mode == "head_then_full" and epoch == head_epochs + 1:
            unfreeze_all(model)
            optimizer = build_optimizer(model, opt_cfg)
            scheduler = build_scheduler(optimizer, sched_cfg,
                                        epochs - head_epochs, steps_per_epoch)
        elif training_mode == "gradual":
            uf_every = gradual_cfg.get("unfreeze_every_epochs", 2)
            if epoch > 1 and (epoch - 1) % uf_every == 0:
                step_n = (epoch - 1) // uf_every
                gradual_unfreeze_step(model, step_n,
                                      gradual_cfg.get("units_per_step", 2))
                optimizer = build_optimizer(model, opt_cfg)
                scheduler = build_scheduler(optimizer, sched_cfg,
                                            epochs, steps_per_epoch)

        # ── Train epoch ──────────────────────────────────────
        model.train()
        ep_loss, n_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _make_autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            if gc_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gc_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler and not isinstance(scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            ep_loss  += loss.item()
            n_batches += 1
        ep_loss /= max(1, n_batches)

        if ema_model:
            ema_model.update_parameters(model)

        # ── Validate ─────────────────────────────────────────
        eval_model = (ema_model
                      if (ema_model and cfg["optimization"]["ema"]["use_for_selection"])
                      else model)
        val_m = evaluate(eval_model, val_loader, device, num_classes)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_m["macro_f1"])

        sel_val = val_m["macro_f1"] if "f1" in sel_metric else val_m["accuracy"]
        history.append({
            "epoch":        epoch,
            "train_loss":   ep_loss,
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
        })

        improved = sel_val > best_metric + es_min_delta
        if improved:
            best_metric = sel_val
            best_state  = copy.deepcopy(model.state_dict())
            es_counter  = 0
        else:
            es_counter += 1

        print(f"  Ep {epoch:3d}/{epochs} | loss={ep_loss:.4f} | "
              f"val_acc={val_m['accuracy']:.4f} | val_f1={val_m['macro_f1']:.4f}"
              f"{' *' if improved else ''}")

        if es_enabled and es_counter >= es_patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # ── Restore best weights ─────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    final_val = evaluate(model, val_loader, device, num_classes)
    result = {
        "id":           exp_id,
        "model":        model_key,
        "best_val_acc": final_val["accuracy"],
        "best_val_f1":  final_val["macro_f1"],
        "history":      history,
    }
    print(f"  BEST  val_acc={final_val['accuracy']:.4f}  val_f1={final_val['macro_f1']:.4f}")
    return result, model


# ──────────────────────────────────────────────────────────────
# Results persistence
# ──────────────────────────────────────────────────────────────
ALL_RESULTS = {}

def save_results(results_dir=RESULTS_DIR):
    rows = [{
        "id":           r["id"],
        "model":        r["model"],
        "best_val_acc": r["best_val_acc"],
        "best_val_f1":  r["best_val_f1"],
    } for r in ALL_RESULTS.values()]
    df = pd.DataFrame(rows)
    (results_dir / "all_experiments.csv").write_text(df.to_csv(index=False))
    (results_dir / "all_experiments.json").write_text(
        json.dumps(ALL_RESULTS, indent=2, default=str))
    print(f"Results saved → {results_dir}")
    return df

# Clear split caches (so re-running cell re-loads fresh)
_FER_TRAIN_CACHE.clear()
_CKP_TRAIN_CACHE.clear()
_RAV_TRAIN_CACHE.clear()
print("Experiment infrastructure ready. device =", device)
"""

# Convert multi-line string to notebook source list
lines = NEW_SOURCE.splitlines(keepends=True)
nb['cells'][infra_idx]['source'] = lines
nb['cells'][infra_idx]['outputs'] = []
nb['cells'][infra_idx]['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Patched cell {infra_idx} successfully.")
print(f"New source: {len(lines)} lines")
