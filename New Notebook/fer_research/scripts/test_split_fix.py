import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Load config
import yaml
CONFIG_PATH = Path("config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Mock notebook environment
PROJECT_ROOT = Path(".").resolve()
PATHS = {
    "affectnet": Path(CONFIG["datasets"]["affectnet"]["root"]),
    "fer2013_folders": Path(CONFIG["datasets"]["fer2013"]["root"]),
    "fer2013_csv": Path(CONFIG["datasets"]["fer2013"]["csv"]["file"]),
    "ckplus": Path(CONFIG["datasets"]["ckplus"]["root"]),
    "ravdess": Path(CONFIG["datasets"]["ravdess"]["root"]),
}
ds_cfg = CONFIG["datasets"]
CANONICAL_CLASSES = CONFIG["label_mapping"]["canonical_classes"]
CLASS2IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
TRAINING = {"seed": 42}
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load notebook and exec cell 8 (loaders) and cell 23 (infra)
nb = json.load(open("notebooks/facial_emotion_research.ipynb", encoding="utf-8"))

cell_8_code = "".join([c["source"] for c in nb["cells"] if c.get("id") == "00000009"][0])
cell_23_code = "".join([c["source"] for c in nb["cells"] if c.get("id") == "exp_infra_code"][0])

g = globals()
exec(cell_8_code, g)
exec(cell_23_code, g)

# Test build_data for FER2013
tr, va = g["build_data"](["fer2013"], ["fer2013"], max_s=100)
print(f"FER2013 test (max_s=100) -> Train: {len(tr)}, Val: {len(va)}")
assert len(tr) > 0, "Train is empty!"
assert len(va) > 0, "Val is empty!"

# Test build_data for CK+
tr_ck, va_ck = g["build_data"](["ckplus"], ["ckplus"])
print(f"CK+ test -> Train: {len(tr_ck)}, Val: {len(va_ck)}")
assert len(tr_ck) > 0, "CK+ Train is empty!"
assert len(va_ck) > 0, "CK+ Val is empty!"

print("\nSUCCESS: All dataset split checks passed! Val data is properly carved and non-empty.")
