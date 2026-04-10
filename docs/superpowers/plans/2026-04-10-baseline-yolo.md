# Baseline YOLO Single-Stage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a baseline YOLO digit detection notebook — train YOLOv11 on utility-meter dataset, evaluate with combined score (0.8*mAP50 + 0.2*FSA), measure inference time, test cross-dataset on waterMeterDataset.

**Architecture:** Single Jupyter notebook with configurable `MODEL_SIZE` parameter. Uses existing shared infra: `models/data/unified_loader.py`, `models/metrics/evaluation.py`, `models/utils/visualization.py`. Outputs weights, metrics JSON, comparison CSV, prediction visualizations.

**Tech Stack:** Python 3.13, uv, PyTorch, Ultralytics YOLOv11, matplotlib, pandas

---

## File Map

```
Notebooks/01_baseline/yolo_single_stage.ipynb  # CREATE — baseline experiment notebook
models/weights/baseline_yolo/data.yaml          # CREATE (at runtime) — fixed data.yaml for YOLO
results/baseline_metrics.json                    # CREATE (at runtime) — all metrics
results/baseline_predictions.png                 # CREATE (at runtime) — 8 sample predictions
results/baseline_comparison.csv                  # CREATE (at runtime) — one row per model size
```

No modifications to existing shared code. All runtime outputs are gitignored.

---

## Task 1: Notebook Foundation — Setup, Config, Dataset Verification

**Files:**
- Create: `Notebooks/01_baseline/yolo_single_stage.ipynb`

- [ ] **Step 1: Create notebook with title cell**

```markdown
# 01 — Baseline: YOLO Single-Stage Digit Detection

Direct digit detection using YOLOv11. Each digit is a separate bounding box (classes 0–9).
Sorted left-to-right → reconstructed meter reading.
```

- [ ] **Step 2: Add Setup cell (cell 1)**

```python
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path("../..").resolve()
if not (ROOT / "models").exists():
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/UrranQx/WaterMeterCV.git"], check=True)
    ROOT = Path("WaterMeterCV").resolve()

sys.path.insert(0, str(ROOT))

import yaml
import json
import time
import csv
import torch
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from models.data.unified_loader import load_utility_meter_dataset, load_water_meter_dataset
from models.metrics.evaluation import full_string_accuracy, per_digit_accuracy, character_error_rate
from models.utils.visualization import draw_digit_bboxes

print(f"ROOT: {ROOT}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

- [ ] **Step 3: Add Config cell (cell 2)**

```python
# ═══════════════════════════════════════════════════════════════════
# MODEL SIZE — change this to switch between nano/small/medium:
#   "yolo11n" — nano  (~2.6M params, ~15 ms/img)  ← start here
#   "yolo11s" — small (~9.6M params, ~45 ms/img)  ← better accuracy
#   "yolo11m" — medium (~20M params, ~120 ms/img) ← max quality, batch↓
# ═══════════════════════════════════════════════════════════════════
MODEL_SIZE = "yolo11n"

# Paths
DATASET_PATH = ROOT / "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
WM_PATH = ROOT / "WaterMetricsDATA/waterMeterDataset/WaterMeters"
DATA_YAML = DATASET_PATH / "data.yaml"
WEIGHTS_DIR = ROOT / "models/weights/baseline_yolo"
RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 16  # reduce to 8–12 for yolo11m if OOM
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 10

RUN_NAME = f"{MODEL_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Model: {MODEL_SIZE}")
print(f"Device: {DEVICE}")
print(f"Run name: {RUN_NAME}")
```

- [ ] **Step 4: Add Dataset Verification cell (cell 3)**

```python
# Load original data.yaml
with open(DATA_YAML) as f:
    data_config = yaml.safe_load(f)

# Fix paths: original uses ../train/images (relative to yaml),
# which breaks depending on CWD. Set absolute path to be safe.
data_config['path'] = str(DATASET_PATH)
data_config['train'] = 'train/images'
data_config['val'] = 'valid/images'
data_config['test'] = 'test/images'

FIXED_YAML = WEIGHTS_DIR / "data.yaml"
with open(FIXED_YAML, 'w') as f:
    yaml.dump(data_config, f)

print(f"Classes ({data_config['nc']}): {data_config['names']}")
print(f"Fixed data.yaml saved to {FIXED_YAML}\n")

# Count images per split
for split_name, split_dir in [("train", "train"), ("valid", "valid"), ("test", "test")]:
    img_dir = DATASET_PATH / split_dir / "images"
    count = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    print(f"  {split_name}: {count} images")
```

- [ ] **Step 5: Run cells 1–3 to verify**

Run cells in Jupyter. Expected output:
- ROOT path printed, CUDA available: True, GPU name shown
- Model/Device/Run name printed
- 14 classes listed, image counts: ~1552 train, ~194 valid, ~194 test

- [ ] **Step 6: Commit**

```bash
git add Notebooks/01_baseline/yolo_single_stage.ipynb
git commit -m "feat: baseline YOLO notebook — setup, config, dataset verification"
```

---

## Task 2: Training Cell

**Files:**
- Modify: `Notebooks/01_baseline/yolo_single_stage.ipynb`

- [ ] **Step 1: Add Training markdown header**

```markdown
## Training
```

- [ ] **Step 2: Add Training cell (cell 4)**

```python
model = YOLO(f"{MODEL_SIZE}.pt")

results = model.train(
    data=str(FIXED_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=str(WEIGHTS_DIR),
    name=RUN_NAME,
    device=DEVICE,
    patience=PATIENCE,
    save=True,
)

print(f"\nTraining complete. Best weights: {WEIGHTS_DIR / RUN_NAME / 'weights' / 'best.pt'}")
```

- [ ] **Step 3: Commit**

```bash
git add Notebooks/01_baseline/yolo_single_stage.ipynb
git commit -m "feat: baseline YOLO notebook — training cell"
```

---

## Task 3: Primary Evaluation + Inference Time

**Files:**
- Modify: `Notebooks/01_baseline/yolo_single_stage.ipynb`

- [ ] **Step 1: Add Evaluation markdown header**

```markdown
## Evaluation
```

- [ ] **Step 2: Add Evaluation cell (cell 5)**

This cell loads the best model, runs YOLO built-in validation for mAP, then runs a timed prediction loop for custom metrics (FSA, per-digit, CER, inference time).

```python
# Load best model
best_weights = WEIGHTS_DIR / RUN_NAME / "weights" / "best.pt"
best_model = YOLO(str(best_weights))

# ── YOLO built-in validation → mAP ──────────────────────────────
val_results = best_model.val(data=str(FIXED_YAML), split="test")
mAP50 = val_results.box.map50
mAP50_95 = val_results.box.map

print(f"mAP50:    {mAP50:.4f}")
print(f"mAP50-95: {mAP50_95:.4f}")
print(f"Per-class AP50:")
for i, name in enumerate(data_config["names"]):
    if i < len(val_results.box.ap50):
        print(f"  {name}: {val_results.box.ap50[i]:.4f}")


def predict_value(model, image_path):
    """Run YOLO and reconstruct digit string from detections.

    Returns (predicted_string, raw_result).
    """
    result = model.predict(str(image_path), verbose=False)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        digit_mask = result.boxes.cls <= 9
        digit_boxes = result.boxes[digit_mask]
        if len(digit_boxes) > 0:
            sorted_idx = digit_boxes.xywh[:, 0].argsort()
            pred_str = "".join(str(int(digit_boxes.cls[i].item())) for i in sorted_idx)
            return pred_str, result
    return "", result


# ── Custom metrics + inference timing ────────────────────────────
test_samples = load_utility_meter_dataset(DATASET_PATH, split="test")

predictions = []
ground_truths = []

t_start = time.perf_counter()
for sample in test_samples:
    pred_str, _ = predict_value(best_model, sample.image_path)
    predictions.append(pred_str)

    gt_str = ""
    if sample.value is not None:
        gt_str = str(int(sample.value)) if sample.value == int(sample.value) else str(sample.value)
    ground_truths.append(gt_str)
t_total_ms = (time.perf_counter() - t_start) * 1000
avg_inference_ms = t_total_ms / len(test_samples)

# Compute metrics
fsa = full_string_accuracy(predictions, ground_truths)

pda_pairs = [(p, g) for p, g in zip(predictions, ground_truths) if g]
pda = sum(per_digit_accuracy(p, g) for p, g in pda_pairs) / len(pda_pairs) if pda_pairs else 0.0

cer_pairs = [(p, g) for p, g in zip(predictions, ground_truths) if g]
cer = sum(character_error_rate(p, g) for p, g in cer_pairs) / len(cer_pairs) if cer_pairs else 0.0

combined = 0.8 * mAP50 + 0.2 * fsa

print(f"\n{'='*50}")
print(f"Full-string accuracy: {fsa:.4f}")
print(f"Per-digit accuracy:   {pda:.4f}")
print(f"CER:                  {cer:.4f}")
print(f"Avg inference:        {avg_inference_ms:.1f} ms/image")
print(f"{'='*50}")
print(f"Combined Score (0.8×mAP50 + 0.2×FSA): {combined:.4f}")
```

- [ ] **Step 3: Commit**

```bash
git add Notebooks/01_baseline/yolo_single_stage.ipynb
git commit -m "feat: baseline YOLO notebook — evaluation + inference timing"
```

---

## Task 4: Visualization, Cross-Dataset, Results Saving, Conclusions

**Files:**
- Modify: `Notebooks/01_baseline/yolo_single_stage.ipynb`

- [ ] **Step 1: Add Visualization markdown header**

```markdown
## Predictions
```

- [ ] **Step 2: Add Visualization cell (cell 6)**

```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, sample in zip(axes.flat, test_samples[:8]):
    img = cv2.imread(str(sample.image_path))
    if img is None:
        ax.set_title("not found")
        ax.axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    pred_str, result = predict_value(best_model, sample.image_path)

    # Draw predicted digit bboxes
    if result.boxes is not None and len(result.boxes) > 0:
        digit_mask = result.boxes.cls <= 9
        digit_boxes = result.boxes[digit_mask]
        if len(digit_boxes) > 0:
            bboxes = []
            for i in range(len(digit_boxes)):
                cls_id = int(digit_boxes.cls[i].item())
                cx = digit_boxes.xywh[i, 0].item() / w_img
                cy = digit_boxes.xywh[i, 1].item() / h_img
                bw = digit_boxes.xywh[i, 2].item() / w_img
                bh = digit_boxes.xywh[i, 3].item() / h_img
                bboxes.append((cls_id, cx, cy, bw, bh))
            img = draw_digit_bboxes(img, bboxes)

    gt_str = ""
    if sample.value is not None:
        gt_str = str(int(sample.value)) if sample.value == int(sample.value) else str(sample.value)

    ax.imshow(img)
    ax.set_title(f"GT={gt_str or '—'} | Pred={pred_str or '—'}", fontsize=10)
    ax.axis("off")

plt.suptitle(f"Baseline YOLO ({MODEL_SIZE}) — Test Predictions", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_predictions.png", dpi=150)
plt.close()
print("Saved to results/baseline_predictions.png")
```

- [ ] **Step 3: Add Cross-Dataset markdown header**

```markdown
## Cross-Dataset Evaluation (waterMeterDataset)

Per-digit accuracy only — full-string skipped because WM ground truth has decimals
(e.g. `78.677`) while YOLO predicts digit sequences only (`78677`). Decimal handling deferred.
```

- [ ] **Step 4: Add Cross-Dataset Evaluation cell (cell 7)**

```python
wm_samples = load_water_meter_dataset(WM_PATH)

wm_pda_scores = []
for sample in wm_samples:
    pred_str, _ = predict_value(best_model, sample.image_path)

    # Strip decimal point from GT for digit-only comparison
    gt_str = ""
    if sample.value is not None:
        gt_str = str(sample.value).replace(".", "")

    if gt_str:
        wm_pda_scores.append(per_digit_accuracy(pred_str, gt_str))

wm_pda = sum(wm_pda_scores) / len(wm_pda_scores) if wm_pda_scores else 0.0
print(f"waterMeterDataset — per-digit accuracy: {wm_pda:.4f}  (N={len(wm_pda_scores)})")
```

- [ ] **Step 5: Add Results Saving markdown header**

```markdown
## Save Results
```

- [ ] **Step 6: Add Results Saving cell (cell 8)**

```python
# Save full metrics to JSON
metrics = {
    "model_size": MODEL_SIZE,
    "run_name": RUN_NAME,
    "primary_eval": {
        "mAP50": round(float(mAP50), 4),
        "mAP50_95": round(float(mAP50_95), 4),
        "full_string_accuracy": round(fsa, 4),
        "per_digit_accuracy": round(pda, 4),
        "CER": round(cer, 4),
        "combined_score": round(combined, 4),
        "avg_inference_ms": round(avg_inference_ms, 1),
    },
    "cross_dataset_eval": {
        "wm_per_digit_accuracy": round(wm_pda, 4),
    },
    "config": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "patience": PATIENCE,
    },
    "run_date": datetime.now().isoformat(),
}

with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Append to comparison CSV (one row per model size run)
csv_path = RESULTS_DIR / "baseline_comparison.csv"
csv_exists = csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow([
            "model_size", "mAP50", "mAP50_95", "full_string_acc",
            "per_digit_acc", "CER", "combined_score", "inference_ms", "run_date",
        ])
    writer.writerow([
        MODEL_SIZE,
        round(float(mAP50), 4), round(float(mAP50_95), 4),
        round(fsa, 4), round(pda, 4), round(cer, 4),
        round(combined, 4), round(avg_inference_ms, 1),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    ])

print(f"Metrics → {RESULTS_DIR / 'baseline_metrics.json'}")
print(f"CSV    → {csv_path}")
```

- [ ] **Step 7: Add Conclusions cell**

```markdown
## Conclusions

*(Fill after running)*

- **Model:** yolo11n / yolo11s / yolo11m
- **Combined Score:** ...
- **mAP50:** ...
- **Full-string accuracy:** ...
- **Per-digit accuracy:** ...
- **CER:** ...
- **Inference:** ... ms/image
- **Cross-dataset (WM per-digit):** ...
- **Next step:** upgrade model / proceed to ROI experiments
```

- [ ] **Step 8: Commit**

```bash
git add Notebooks/01_baseline/yolo_single_stage.ipynb
git commit -m "feat: baseline YOLO notebook — visualization, cross-dataset eval, results saving"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|---|---|
| MODEL_SIZE config with comments | Task 1, Step 3 |
| Dataset verification | Task 1, Step 4 |
| data.yaml path fix | Task 1, Step 4 |
| Training with early stopping | Task 2, Step 2 |
| mAP50 via YOLO .val() | Task 3, Step 2 |
| Full-string accuracy | Task 3, Step 2 |
| Per-digit accuracy | Task 3, Step 2 |
| CER | Task 3, Step 2 |
| Combined score (0.8*mAP50 + 0.2*FSA) | Task 3, Step 2 |
| Inference time (full test set avg) | Task 3, Step 2 |
| 8-image prediction grid | Task 4, Step 2 |
| WM cross-dataset (per-digit only) | Task 4, Step 4 |
| Save baseline_metrics.json | Task 4, Step 6 |
| Save baseline_comparison.csv | Task 4, Step 6 |
| Conclusions markdown | Task 4, Step 7 |
| Iteration plan (nano→small→medium) | Task 1, Step 3 (MODEL_SIZE comments) |
