# Facial Emotion Recognition — research framework

A modular PyTorch research framework for evaluating, comparing, and finding the best **validated and generalisable**
facial-emotion model across **AffectNet, FER2013, CK+ and RAVDESS**.

Everything is driven by one file, `config.yaml`. Experiments are defined there, run resumably, compared only
within identical evaluation protocols, and reported from measured results.

> **Status.** The framework has been verified end-to-end (59 passing tests, smoke runs, notebook execution) on
> *synthetic* datasets that mimic each dataset's on-disk format. **No results on the real datasets exist yet** —
> every number in `results/` is produced by your own runs. Nothing in the reports is estimated or filled in.

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -r requirements.txt                          # GPU: install torch/torchvision for your CUDA first
python -m src.cli check-env                              # reports CPU/GPU/MPS, library versions
```

**Face detector model (needed for CK+ and RAVDESS):** download `face_detection_yunet_2023mar.onnx` from the
OpenCV Zoo (`github.com/opencv/opencv_zoo`, folder `models/face_detection_yunet`) into `models/`.
Haar cascades are an alternative (`preprocessing.face_detector.backend: haar`) but **OpenCV 5 removed
`CascadeClassifier` from the main package** — use `opencv-python<5` or `opencv-contrib-python` for Haar.

**Pretrained weights** are fetched by `timm` from the Hugging Face hub on first use. Offline machines: copy the HF
cache, or point `models.defaults.pretrained_source` at a local weight file. If weights cannot be loaded the run
**fails** unless `models.defaults.allow_scratch_fallback: true`, in which case it is recorded as scratch.

**Google Colab:** upload/clone the repo to Drive, open `notebooks/facial_emotion_research.ipynb`, uncomment the
Drive-mount and `pip install` lines in the first cell.

### Tested environment
Python 3.13, PyTorch 2.14 (CPU), torchvision 0.29, timm 1.0.30, OpenCV 5.0, onnxruntime 1.29, ONNX 1.23,
Optuna 5.0, pandas 3.0, scikit-learn 1.9 — Linux, CPU only.

---

## 2. Datasets (not downloaded automatically — each requires its own licence)

Set `paths.data_root` (or each `datasets.<name>.root`) in `config.yaml`. Supported layouts:

| Dataset | `format` | Expected layout |
|---|---|---|
| AffectNet | `official_csv` | `Manually_Annotated_Images/…` + `Manually_Annotated_file_lists/{training,validation}.csv` (header-less `validation.csv` handled) |
| | `folders` | `train/<class>/*.jpg`, `val/<class>/*.jpg` (class folders may be names or codes 0–10) |
| | `npy` | `train_set/images/<id>.jpg` + `train_set/annotations/<id>_exp.npy` (same for `val_set`) |
| FER2013 | `csv` | `fer2013.csv` with `emotion, pixels, Usage` (also `icml_face_data.csv` with spaced headers) |
| | `folders` | `train/<class>/*.png`, `test/<class>/*.png` (Kaggle) |
| CK+ | `original` | `cohn-kanade-images/S###/###/*.png` + `Emotion/S###/###/*_emotion.txt` |
| | `folders` | `<dir>/<class>/S###_###_########.png` (e.g. CK+48) |
| RAVDESS | — | any tree of `MM-VV-EE-II-SS-RR-AA.mp4` files (`video_glob`) |

```bash
python -m src.cli validate        # names every missing path and shows the expected layout
python -m src.cli manifest        # builds manifests + reports (cached; --force to rebuild)
```

Disable a dataset you do not have with `datasets.<name>.enabled: false`; experiments that need it are marked
**skipped** (with the reason), never run on substitute data.

---

## 3. Evaluation protocols and leakage control

| Dataset | Train / validation / test | Subject leakage |
|---|---|---|
| FER2013 (csv) | Training / PublicTest / PrivateTest (official) | not verifiable (no IDs) |
| FER2013 (folders) | train minus stratified val / val / `test` folder (= Public+Private; **not** the official protocol) | not verifiable |
| AffectNet | official training minus stratified 5 % val / val / **official validation as test** (test labels are not public) | not verifiable (no IDs) |
| CK+ | subject-independent 10-fold (test fold *k*, val fold *k+1*) or holdout | enforced |
| RAVDESS | actor-independent holdout or explicit actor lists | enforced |

* All frames of one CK+ sequence / RAVDESS clip stay in one split. RAVDESS **audio-video (01) and video-only (02)
  files of a performance contain identical frames**; they share a `clip_id` so they can never be split apart.
* Near-duplicate images are found with dHash candidates confirmed by pixel correlation (≥0.95); copies are
  removed from **train** (a val↔test copy is removed from val). The test set is never altered.
  FER2013 is known to contain such cross-split duplicates.
* Run-level checks repeat this across every dataset a run touches **including held-out cross-dataset targets**,
  so an image shared between e.g. AffectNet-train and FER2013-test is removed from training before training.
* Automated assertions (`splits.leakage_checks`) raise `LeakageError`; checks without metadata are reported as
  `not_verifiable`, never as passed.
* The split seed (`splits.seed`) is independent of training seeds: multi-seed runs vary initialisation and data
  order only.
* Model selection, early stopping, HPO, ensemble membership, TTA use and quantisation calibration use
  **train/validation data only**; the test split is evaluated once per run after selection.

### Label mapping
Seven canonical classes (`angry, disgust, fear, happy, sad, surprise, neutral`). Defaults: contempt, AffectNet
none/uncertain/no-face and RAVDESS *calm* are **excluded with a recorded reason** (calm is a posed low-arousal
state, not neutral). CK+ peak frames get the sequence label; the first frame(s) are labelled neutral
(`frame_selection.neutral_frames`). An alternative `six_class` set (no disgust) is available via
`label_mapping.active_label_set`. See `results/manifests/label_mapping_report.md` after building manifests.

---

## 4. Configuration (`config.yaml`)

Sections: `environment, paths, datasets, label_mapping, preprocessing, splits, models, training, loss,
augmentation, optimization, experiments, evaluation, cross_dataset, logging`.

* **Strict schema**: unknown keys or invalid values abort with a list of every problem
  (e.g. `training.learning_rat: unsupported configuration option`). Semantic checks catch e.g. class weights +
  balanced sampling together (needs `allow_combined_balancing: true`), alignment without landmarks, or a
  cross-dataset target that is also used for training.
* **Overrides** without editing the file: `python -m src.cli run --set training.epochs=10 --set loss.name=focal`.
* **Per-model input resolution** (`models.architectures.<m>.input_size`) — EfficientNet-B2 uses 256,
  others 224; ViT position embeddings are interpolated by timm for other sizes. 48/112/160 px are possible.
* `preprocessing.color_mode: grayscale_3ch` (default) removes the colour cue between grayscale FER2013/CK+ and
  colour AffectNet/RAVDESS; gray is replicated to 3 channels so ImageNet weights still apply.
* **Experiments** (`experiments.definitions`): each has `id, group, model, train, val, test` (val/test default to
  `train` when omitted), optional `seeds`, `overrides` (any config key), `grid` (expands models, `dataset`,
  `dataset_combo` or any key), `cross_eval` (held-out targets), `depends_on`.
  Groups A–F mirror the project experiments matrix (baselines, architectures, single-dataset benchmarks, combinations, training factors, and cross-dataset generalisation).
* Each run stores `config_snapshot.yaml`; a run is skipped when completed with an identical config hash.

---

## 5. Workflow

```bash
python -m src.cli check-env                           # Phase 1
python -m src.cli validate                            # Phase 2
python -m src.cli manifest                            # Phase 3 (reports in results/manifests/)
python -m src.cli smoke                               # Phase 5: 1 epoch x 3 steps on real data
python -m src.cli list                                # all runs + status
python -m src.cli estimate --groups A B               # measured step time x planned steps (upper bound)
python -m src.cli run --groups A                      # Phase 5 baselines (resumable)
python -m src.cli run --only "B_arch_affectnet__*"    # Phase 6
python -m src.cli run --groups C D E                  # Phases 6-7
python -m src.cli hpo                                 # Phase 7 (optimization.hpo; resumable sqlite study)
python -m src.cli run --groups F                      # Phase 8 cross-dataset
python -m src.cli cross-eval --source A2_effb0_affectnet --targets ckplus ravdess
python -m src.cli ensemble --members A1_effb0_fer2013 A3_r50_fer2013
python -m src.cli report                              # Phase 9 -> results/report.md
```

Interrupting a run (Ctrl-C, preemption) is safe: rerun the same command; it resumes from `last.pt` when the
configuration is unchanged. Failed runs are recorded with their traceback and the runner continues
(`experiments.continue_on_failure`); runs depending on a failed run are skipped.

**Notebook:** `notebooks/facial_emotion_research.ipynb` walks through the same 14 steps. Set
`FER_NOTEBOOK_DEMO=1` (or `USE_SYNTHETIC_DEMO = True`) to try it on generated data.

### Outputs
```
results/manifests/          manifests (parquet+csv), label_mapping_report.md, validation_report.json
results/experiments/<run>/  metrics.json, history.csv, curves.png, confusion_*.png, predictions_*.npz,
                            config_snapshot.yaml, data_report.json, cross_dataset.json, status.json
results/all_results.csv, experiment_status.csv, cross_dataset_results.csv, seed_aggregates.csv, report.md
checkpoints/<run>/best.pt   selected weights + metadata (classes, input size, normalisation, provenance)
checkpoints/<run>/last.pt   full training state for resuming
logs/<run>.log
```

`metrics.json` contains every metric from the brief: model/architecture, datasets, scratch vs. pretrained and
weight source, loss, optimiser, scheduler, batch size, epochs completed, best epoch, validation accuracy and
macro-F1, test accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix, training time,
CPU batch-1 latency, images/s, parameter count, model size, peak GPU memory. Macro-F1 is averaged over the
classes present in the evaluated ground truth.

---

## 6. Tests

```bash
pytest                       # test suite, synthetic data generated in a temp dir
FER_HAAR_CASCADE=/path/haarcascade_frontalface_default.xml pytest   # also exercise Haar detection
```
Covers config parsing/validation, path validation, label mapping, image loading, every architecture's output
shape, freezing, training/validation execution, checkpoint save/reload, resume, seed reproducibility, leakage
detection (subject/clip/sequence/duplicate), metric correctness vs. scikit-learn, losses, experiment
expansion, skip/resume bookkeeping, cross-dataset evaluation, HPO (test split never evaluated), ensembles,
distillation, gradual unfreezing, and staged training.
The synthetic dataset generator in the research notebook documents the deliberate format traps used for verification.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Process killed / CUDA OOM | lower `environment.batch_size` (or per-model `batch_size`) and raise `gradient_accumulation_steps` |
| `PretrainedWeightsError` | no internet/HF access: copy the HF cache or set `pretrained_source`; or allow scratch fallback explicitly |
| `Haar cascade not found` / `no CascadeClassifier` | use YuNet (default) or install `opencv-python<5` / `opencv-contrib-python` |
| `LeakageError` | read the message: a subject/clip/duplicate crosses splits; check custom `actor_lists` or set `cross_split_policy: drop_from_train` |
| "completed with a DIFFERENT configuration" | results are kept; use `--force` or `experiments.rerun_on_config_change: true` |
| Manifest seems stale after moving data | `python -m src.cli manifest --force` |
| Many CK+/RAVDESS frames "no face" | check the detector model, lower `min_face_size`, or `on_no_face: center_crop` (recorded) |
| DataLoader hangs (Windows/notebooks) | `environment.dataloader.num_workers: 0` |
| ViT/ConvNeXt extremely slow | expected on CPU; run `estimate` first, use a GPU, or restrict `B` to small models |

---

## 8. Reproducing a result

1. Use the run's `results/experiments/<run>/config_snapshot.yaml` as `--config` (it contains the run definition
   under `_run`), the same manifests (`results/manifests/*.parquet`, keyed by a config hash) and the recorded
   library versions (`python -m src.cli check-env`).
2. `environment.deterministic: true` enables deterministic cuDNN and `torch.use_deterministic_algorithms`
   (warn-only). Bit-identical results are guaranteed only on the same hardware/software stack.
3. Report mean ± std over seeds (`seeds: [42, 43, 44]`, aggregated in `seed_aggregates.csv`) for key comparisons.

## 9. Known limitations

* FER2013 and AffectNet have no subject identifiers — identity leakage across their splits cannot be excluded.
* AffectNet results use the official validation set as test; they are not comparable to papers that report on
  other subsets or class sets (8-class, balanced subsets).
* CK+ and RAVDESS are small, posed, lab datasets; per-fold variance is large — use several folds/seeds.
* RAVDESS emotion labels describe the *performed* emotion of the whole clip; individual frames (especially
  early ones, skipped via `skip_start_seconds`) may look neutral.
* Cross-dataset drops reflect domain shift (resolution, colour, demographics, posed vs. spontaneous expressions,
  annotation conventions) as much as model quality.
