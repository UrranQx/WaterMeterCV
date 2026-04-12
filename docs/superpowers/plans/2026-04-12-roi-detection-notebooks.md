# ROI Detection Notebooks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 3 ROI detection notebooks + shared data helpers. Each notebook trains/evals on both datasets independently (utility-meter and waterMeterDataset).

**Architecture:** Shared ROI data helpers in `models/data/roi_dataset.py`, WM split in `unified_loader.py`. Three notebooks (`faster_rcnn.ipynb`, `yolo_roi.ipynb`, `segmentation_unet.ipynb`) each follow a 21-cell structure: setup, UM experiment, WM experiment, visualization, comparison, save results. WM dataset is the primary focus (1244 images with full ROI polygons); UM is secondary (45 images with ROI).

**Tech Stack:** PyTorch, Detectron2, Ultralytics YOLOv11, segmentation-models-pytorch, albumentations

**Spec:** `docs/superpowers/specs/2026-04-11-roi-detection-notebooks-design.md`

**Key conventions:**
- Notebook format: `nbformat: 4`, `nbformat_minor: 4`, python3 kernel, `language_info.version: "3.13.0"`
- IN_COLAB setup pattern from `docs/colab-workflow.md` with `BRANCH = "feature/roi-detection"`
- WORKERS: 0 on Windows, 2 in Colab
- Weights: local `models/weights/<method>/`, Colab `WEIGHTS_BASE/<method>/`
- Results: `results/roi_<method>_metrics.json` + append to `results/roi_comparison.csv`

---

## File Structure

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `models/data/unified_loader.py` | Add `load_water_meter_dataset_split()` |
| Create | `models/data/roi_dataset.py` | Shared ROI helpers: polygon_to_bbox, dataset prep |
| Modify | `tests/test_unified_loader.py` | Tests for WM split |
| Create | `tests/test_roi_dataset.py` | Tests for ROI helpers |
| Create | `Notebooks/02_roi_detection/faster_rcnn.ipynb` | Detectron2 Faster R-CNN notebook |
| Create | `Notebooks/02_roi_detection/yolo_roi.ipynb` | YOLOv11 single-class ROI notebook |
| Create | `Notebooks/02_roi_detection/segmentation_unet.ipynb` | U-Net segmentation notebook |

---

## Task 1: WM dataset split in unified_loader

**Files:**
- Modify: `models/data/unified_loader.py`
- Modify: `tests/test_unified_loader.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_unified_loader.py`:

```python
from models.data.unified_loader import load_water_meter_dataset_split


class TestWaterMeterSplit:
    def test_split_returns_two_lists(self):
        train, test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        assert isinstance(train, list)
        assert isinstance(test, list)
        assert len(train) > 0
        assert len(test) > 0

    def test_split_ratio_approximately_correct(self):
        train, test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        total = len(train) + len(test)
        assert abs(len(train) / total - 0.7) < 0.01

    def test_split_is_deterministic(self):
        t1, _ = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        t2, _ = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        assert [s.image_path for s in t1] == [s.image_path for s in t2]

    def test_split_no_overlap(self):
        train, test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        train_paths = {s.image_path for s in train}
        test_paths = {s.image_path for s in test}
        assert train_paths.isdisjoint(test_paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_unified_loader.py::TestWaterMeterSplit -v`
Expected: FAIL — `ImportError: cannot import name 'load_water_meter_dataset_split'`

- [ ] **Step 3: Implement**

Append to `models/data/unified_loader.py`:

```python
import random


def load_water_meter_dataset_split(
    root: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[UnifiedSample], list[UnifiedSample]]:
    """Deterministic train/test split of waterMeterDataset.

    Returns (train_samples, test_samples).
    """
    all_samples = load_water_meter_dataset(root)
    shuffled = all_samples.copy()
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_unified_loader.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add models/data/unified_loader.py tests/test_unified_loader.py
git commit -m "feat: add load_water_meter_dataset_split with deterministic 70/30 split"
```

---

## Task 2: ROI dataset helpers

**Files:**
- Create: `models/data/roi_dataset.py`
- Create: `tests/test_roi_dataset.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_roi_dataset.py`:

```python
import pytest
from pathlib import Path
from models.data.roi_dataset import (
    polygon_to_bbox,
    filter_utility_meter_roi_samples,
    prepare_yolo_roi_dataset,
)

DATA_ROOT = Path("WaterMetricsDATA")
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"


class TestPolygonToBbox:
    def test_square_polygon(self):
        polygon = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
        cx, cy, w, h = polygon_to_bbox(polygon)
        assert abs(cx - 0.2) < 1e-6
        assert abs(cy - 0.2) < 1e-6
        assert abs(w - 0.2) < 1e-6
        assert abs(h - 0.2) < 1e-6

    def test_triangle_polygon(self):
        polygon = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        cx, cy, w, h = polygon_to_bbox(polygon)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6


class TestFilterUtilityMeterRoi:
    def test_returns_only_images_with_roi(self):
        samples = filter_utility_meter_roi_samples(UM_YOLO_PATH, "train")
        assert len(samples) > 0
        for img_path, bbox in samples:
            assert img_path.exists()
            assert len(bbox) == 4
            cx, cy, w, h = bbox
            assert 0 <= cx <= 1 and 0 <= cy <= 1

    def test_train_has_45_roi_images(self):
        samples = filter_utility_meter_roi_samples(UM_YOLO_PATH, "train")
        assert len(samples) == 45


class TestPrepareYoloRoiDataset:
    def test_creates_single_class_dataset(self, tmp_path):
        dst = tmp_path / "roi_yolo"
        prepare_yolo_roi_dataset(UM_YOLO_PATH, dst)
        assert (dst / "data.yaml").exists()
        # Check a label file has only class 0
        label_files = list((dst / "train" / "labels").glob("*.txt"))
        assert len(label_files) > 0
        with open(label_files[0]) as f:
            for line in f:
                parts = line.strip().split()
                assert parts[0] == "0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_roi_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.data.roi_dataset'`

- [ ] **Step 3: Implement `models/data/roi_dataset.py`**

```python
"""Shared ROI detection helpers for all 02_roi_detection notebooks."""
from pathlib import Path
import shutil
import yaml


def polygon_to_bbox(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Convert normalized polygon [(x, y), ...] to (cx, cy, w, h) normalized bbox."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    return (cx, cy, w, h)


def filter_utility_meter_roi_samples(
    dataset_path: Path,
    split: str,
) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """Return (image_path, roi_bbox) for images that have class 10 (ROI).

    roi_bbox is (cx, cy, w, h) normalized.
    """
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / split / "labels"
    images_dir = dataset_path / split / "images"
    results = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 10:
                    bbox = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
                    # Find image
                    for ext in (".jpg", ".jpeg", ".png"):
                        img_path = images_dir / (label_file.stem + ext)
                        if img_path.exists():
                            results.append((img_path, bbox))
                            break
                    break  # one ROI per image
    return results


def prepare_yolo_roi_dataset(src_path: Path, dst_path: Path) -> None:
    """Create single-class YOLO dataset: filter class 10 -> class 0, copy images.

    Writes data.yaml with nc=1, names=["ROI"].
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    for split in ["train", "valid", "test"]:
        (dst_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / split / "labels").mkdir(parents=True, exist_ok=True)

        src_labels = src_path / split / "labels"
        src_images = src_path / split / "images"

        for label_file in src_labels.glob("*.txt"):
            roi_lines = []
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == 10:
                        roi_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

            if roi_lines:
                for ext in (".jpg", ".jpeg", ".png"):
                    src_img = src_images / (label_file.stem + ext)
                    if src_img.exists():
                        shutil.copy(src_img, dst_path / split / "images" / src_img.name)
                        break
                with open(dst_path / split / "labels" / label_file.name, "w") as f:
                    f.write("\n".join(roi_lines))

    data_yaml = {
        "path": str(dst_path),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["ROI"],
    }
    with open(dst_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)


def prepare_wm_yolo_roi_dataset(
    train_samples: list,
    test_samples: list,
    dst_path: Path,
) -> None:
    """Convert waterMeterDataset samples to YOLO single-class ROI dataset.

    Each sample must have roi_polygon. Polygon is converted to bbox.
    Creates train/ and test/ splits (no valid — use test for val during training).
    """
    dst_path = Path(dst_path)

    for split_name, samples in [("train", train_samples), ("test", test_samples)]:
        (dst_path / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / split_name / "labels").mkdir(parents=True, exist_ok=True)

        for sample in samples:
            if sample.roi_polygon is None:
                continue
            cx, cy, w, h = polygon_to_bbox(sample.roi_polygon)
            img_name = sample.image_path.name
            stem = sample.image_path.stem

            # Symlink or copy image
            dst_img = dst_path / split_name / "images" / img_name
            if not dst_img.exists():
                shutil.copy(sample.image_path, dst_img)

            with open(dst_path / split_name / "labels" / f"{stem}.txt", "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = {
        "path": str(dst_path),
        "train": "train/images",
        "val": "test/images",
        "test": "test/images",
        "nc": 1,
        "names": ["ROI"],
    }
    with open(dst_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_roi_dataset.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add models/data/roi_dataset.py tests/test_roi_dataset.py
git commit -m "feat: ROI data helpers — polygon_to_bbox, UM filter, YOLO dataset prep"
```

---

## Task 3: Notebook — YOLO ROI (`yolo_roi.ipynb`)

Starting with YOLO because it reuses the most from baseline and has the simplest setup.

**Files:**
- Create: `Notebooks/02_roi_detection/yolo_roi.ipynb`

- [ ] **Step 1: Create notebook with all cells**

Notebook format: `nbformat: 4`, `nbformat_minor: 4`, kernel `python3`, `language_info.version: "3.13.0"`.

**Cell 0 (markdown):**
```markdown
# 02 — ROI Detection: YOLO Single-Class

Detect the "Reading Digit" window (ROI) using YOLOv11 trained on a single class.
Two experiments: utility-meter dataset (sparse ROI) and waterMeterDataset (full polygon ROI).
```

**Cell 1 (code) — Setup:**
```python
import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB:
    try:
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        pass

if IN_COLAB:
    from google.colab import drive, userdata

    drive.mount("/content/drive")

    token = userdata.get("GITHUB_TOKEN", "")
    base = f"https://{token}@github.com" if token else "https://github.com"
    if not Path("/content/WaterMeterCV").exists():
        subprocess.run(
            ["git", "clone", f"{base}/UrranQx/WaterMeterCV.git", "/content/WaterMeterCV"],
            check=True
        )

    BRANCH = "feature/roi-detection"
    subprocess.run(["git", "-C", "/content/WaterMeterCV", "checkout", BRANCH], check=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "albumentations", "rapidfuzz", "shapely"],
        check=True
    )

    ROOT         = Path("/content/WaterMeterCV")
    DATA_ROOT    = Path("/content/drive/MyDrive/WaterMeterCV/WaterMetricsDATA")
    WEIGHTS_BASE = Path("/content/drive/MyDrive/WaterMeterCV/weights")
    RESULTS_DIR  = Path("/content/drive/MyDrive/WaterMeterCV/results")
    WORKERS = 2
else:
    ROOT         = Path("../..").resolve()
    DATA_ROOT    = ROOT / "WaterMetricsDATA"
    WEIGHTS_BASE = ROOT / "models/weights"
    RESULTS_DIR  = ROOT / "results"
    WORKERS = 0

sys.path.insert(0, str(ROOT))
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yaml
import json
import time
import csv
import torch
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from models.data.unified_loader import load_water_meter_dataset_split
from models.data.roi_dataset import (
    filter_utility_meter_roi_samples,
    prepare_yolo_roi_dataset,
    prepare_wm_yolo_roi_dataset,
    polygon_to_bbox,
)
from models.metrics.evaluation import compute_iou_bbox

print(f"ROOT: {ROOT}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 2 (code) — Config:**
```python
MODEL_SIZE = "yolo11n"

# Dataset paths
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
WM_PATH = DATA_ROOT / "waterMeterDataset/WaterMeters"

# Working dirs
WEIGHTS_DIR = WEIGHTS_BASE / "roi_yolo"
UM_ROI_DATASET = DATA_ROOT / "_roi_only_yolo_um"
WM_ROI_DATASET = DATA_ROOT / "_roi_only_yolo_wm"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparams
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 10

RUN_NAME = f"yolo_roi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Model: {MODEL_SIZE}")
print(f"Device: {DEVICE}")
print(f"Workers: {WORKERS}")
```

**Cell 3 (code) — Data Prep:**
```python
# UM: filter to ROI-only single-class YOLO dataset
prepare_yolo_roi_dataset(UM_YOLO_PATH, UM_ROI_DATASET)
um_train_roi = filter_utility_meter_roi_samples(UM_YOLO_PATH, "train")
um_test_roi = filter_utility_meter_roi_samples(UM_YOLO_PATH, "test")
print(f"UM ROI: {len(um_train_roi)} train, {len(um_test_roi)} test")

# WM: 70/30 split, convert polygons to YOLO labels
wm_train, wm_test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
prepare_wm_yolo_roi_dataset(wm_train, wm_test, WM_ROI_DATASET)
print(f"WM ROI: {len(wm_train)} train, {len(wm_test)} test")
```

**Cell 4 (markdown):**
```markdown
## Experiment 1: utility-meter dataset
```

**Cell 5 (code) — Verify UM:**
```python
for split in ["train", "valid", "test"]:
    img_dir = UM_ROI_DATASET / split / "images"
    if img_dir.exists():
        count = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        print(f"  {split}: {count} images with ROI")
```

**Cell 6 (markdown):**
```markdown
## Training (utility-meter)
```

**Cell 7 (code) — Train UM:**
```python
um_model = YOLO(f"{MODEL_SIZE}.pt")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

um_results = um_model.train(
    data=str(UM_ROI_DATASET / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=str(WEIGHTS_DIR),
    name=f"um_{RUN_NAME}",
    device=DEVICE,
    patience=PATIENCE,
    workers=WORKERS,
    save=True,
)

um_best = WEIGHTS_DIR / f"um_{RUN_NAME}" / "weights" / "best.pt"
print(f"Best weights: {um_best}")
```

**Cell 8 (markdown):**
```markdown
## Evaluation (utility-meter)
```

**Cell 9 (code) — Eval UM:**
```python
um_best_model = YOLO(str(um_best))

um_ious = []
um_inference_times = []
for img_path, gt_bbox in um_test_roi:
    t0 = time.perf_counter()
    result = um_best_model.predict(str(img_path), verbose=False)[0]
    um_inference_times.append((time.perf_counter() - t0) * 1000)

    if result.boxes is not None and len(result.boxes) > 0:
        # Top-1 by confidence
        best_idx = result.boxes.conf.argmax()
        box = result.boxes.xywhn[best_idx]
        pred_bbox = (box[0].item(), box[1].item(), box[2].item(), box[3].item())
        um_ious.append(compute_iou_bbox(pred_bbox, gt_bbox))
    else:
        um_ious.append(0.0)

um_mean_iou = np.mean(um_ious) if um_ious else 0.0
um_det_rate = sum(1 for v in um_ious if v > 0) / len(um_ious) if um_ious else 0.0
um_avg_ms = np.mean(um_inference_times) if um_inference_times else 0.0

print(f"UM — Mean IoU: {um_mean_iou:.4f}")
print(f"UM — Detection rate: {um_det_rate:.4f} ({sum(1 for v in um_ious if v > 0)}/{len(um_ious)})")
print(f"UM — Avg inference: {um_avg_ms:.1f} ms/image")
```

**Cell 10 (markdown):**
```markdown
## Experiment 2: waterMeterDataset
```

**Cell 11 (code) — Train WM:**
```python
wm_model = YOLO(f"{MODEL_SIZE}.pt")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

wm_results = wm_model.train(
    data=str(WM_ROI_DATASET / "data.yaml"),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=str(WEIGHTS_DIR),
    name=f"wm_{RUN_NAME}",
    device=DEVICE,
    patience=PATIENCE,
    workers=WORKERS,
    save=True,
)

wm_best = WEIGHTS_DIR / f"wm_{RUN_NAME}" / "weights" / "best.pt"
print(f"Best weights: {wm_best}")
```

**Cell 12 (code) — Eval WM:**
```python
wm_best_model = YOLO(str(wm_best))

wm_ious = []
wm_inference_times = []
for sample in wm_test:
    if sample.roi_polygon is None:
        continue
    gt_bbox = polygon_to_bbox(sample.roi_polygon)

    t0 = time.perf_counter()
    result = wm_best_model.predict(str(sample.image_path), verbose=False)[0]
    wm_inference_times.append((time.perf_counter() - t0) * 1000)

    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = result.boxes.conf.argmax()
        box = result.boxes.xywhn[best_idx]
        pred_bbox = (box[0].item(), box[1].item(), box[2].item(), box[3].item())
        wm_ious.append(compute_iou_bbox(pred_bbox, gt_bbox))
    else:
        wm_ious.append(0.0)

wm_mean_iou = np.mean(wm_ious) if wm_ious else 0.0
wm_det_rate = sum(1 for v in wm_ious if v > 0) / len(wm_ious) if wm_ious else 0.0
wm_avg_ms = np.mean(wm_inference_times) if wm_inference_times else 0.0

print(f"WM — Mean IoU: {wm_mean_iou:.4f}")
print(f"WM — Detection rate: {wm_det_rate:.4f} ({sum(1 for v in wm_ious if v > 0)}/{len(wm_ious)})")
print(f"WM — Avg inference: {wm_avg_ms:.1f} ms/image")
```

**Cell 13 (markdown):**
```markdown
## Predictions
```

**Cell 14 (code) — Visualization:**
```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Top row: UM test samples
for i, ax in enumerate(axes[0]):
    if i >= len(um_test_roi):
        ax.axis("off")
        continue
    img_path, gt_bbox = um_test_roi[i]
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    # GT box (green)
    cx, cy, w, h = gt_bbox
    x1, y1 = int((cx - w/2) * w_img), int((cy - h/2) * h_img)
    x2, y2 = int((cx + w/2) * w_img), int((cy + h/2) * h_img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Pred box (red)
    result = um_best_model.predict(str(img_path), verbose=False)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = result.boxes.conf.argmax()
        box = result.boxes.xywhn[best_idx]
        px, py, pw, ph = box[0].item(), box[1].item(), box[2].item(), box[3].item()
        px1, py1 = int((px - pw/2) * w_img), int((py - ph/2) * h_img)
        px2, py2 = int((px + pw/2) * w_img), int((py + ph/2) * h_img)
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    ax.imshow(img)
    iou_val = um_ious[i] if i < len(um_ious) else 0
    ax.set_title(f"UM IoU={iou_val:.2f}", fontsize=10)
    ax.axis("off")

# Bottom row: WM test samples
wm_test_with_roi = [s for s in wm_test if s.roi_polygon is not None]
for i, ax in enumerate(axes[1]):
    if i >= len(wm_test_with_roi):
        ax.axis("off")
        continue
    sample = wm_test_with_roi[i]
    img = cv2.imread(str(sample.image_path))
    if img is None:
        ax.axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    # GT polygon -> bbox (green)
    gt_bbox = polygon_to_bbox(sample.roi_polygon)
    cx, cy, w, h = gt_bbox
    x1, y1 = int((cx - w/2) * w_img), int((cy - h/2) * h_img)
    x2, y2 = int((cx + w/2) * w_img), int((cy + h/2) * h_img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Pred box (red)
    result = wm_best_model.predict(str(sample.image_path), verbose=False)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = result.boxes.conf.argmax()
        box = result.boxes.xywhn[best_idx]
        px, py, pw, ph = box[0].item(), box[1].item(), box[2].item(), box[3].item()
        px1, py1 = int((px - pw/2) * w_img), int((py - ph/2) * h_img)
        px2, py2 = int((px + pw/2) * w_img), int((py + ph/2) * h_img)
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    ax.imshow(img)
    iou_val = wm_ious[i] if i < len(wm_ious) else 0
    ax.set_title(f"WM IoU={iou_val:.2f}", fontsize=10)
    ax.axis("off")

plt.suptitle(f"YOLO ROI ({MODEL_SIZE}) — Green=GT, Red=Pred", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roi_yolo_predictions.png", dpi=150)
plt.close()
print("Saved to results/roi_yolo_predictions.png")
```

**Cell 15 (markdown):**
```markdown
## Comparison
```

**Cell 16 (code) — Compare:**
```python
print(f"{'='*60}")
print(f"{'Metric':<25} {'utility-meter':>15} {'waterMeter':>15}")
print(f"{'='*60}")
print(f"{'Mean IoU':<25} {um_mean_iou:>15.4f} {wm_mean_iou:>15.4f}")
print(f"{'Detection rate':<25} {um_det_rate:>15.4f} {wm_det_rate:>15.4f}")
print(f"{'Avg inference (ms)':<25} {um_avg_ms:>15.1f} {wm_avg_ms:>15.1f}")
print(f"{'N test':<25} {len(um_test_roi):>15d} {len(wm_test):>15d}")
print(f"{'='*60}")
```

**Cell 17 (markdown):**
```markdown
## Save Results
```

**Cell 18 (code) — Save:**
```python
metrics = {
    "method": "yolo_roi",
    "model_size": MODEL_SIZE,
    "utility_meter": {
        "mean_iou": round(um_mean_iou, 4),
        "detection_rate": round(um_det_rate, 4),
        "avg_inference_ms": round(um_avg_ms, 1),
        "n_train": len(um_train_roi),
        "n_test": len(um_test_roi),
    },
    "water_meter": {
        "mean_iou": round(wm_mean_iou, 4),
        "detection_rate": round(wm_det_rate, 4),
        "avg_inference_ms": round(wm_avg_ms, 1),
        "n_train": len(wm_train),
        "n_test": len(wm_test),
    },
    "config": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "patience": PATIENCE,
    },
    "run_date": datetime.now().isoformat(),
}

with open(RESULTS_DIR / "roi_yolo_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

csv_path = RESULTS_DIR / "roi_comparison.csv"
csv_exists = csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow([
            "method", "um_mean_iou", "um_detection_rate", "um_inference_ms",
            "wm_mean_iou", "wm_detection_rate", "wm_inference_ms", "run_date",
        ])
    writer.writerow([
        "yolo_roi",
        round(um_mean_iou, 4), round(um_det_rate, 4), round(um_avg_ms, 1),
        round(wm_mean_iou, 4), round(wm_det_rate, 4), round(wm_avg_ms, 1),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    ])

print(f"Metrics -> {RESULTS_DIR / 'roi_yolo_metrics.json'}")
print(f"CSV    -> {csv_path}")
```

**Cell 19 (markdown):**
```markdown
## Conclusions

*(Fill after running)*

- **YOLO ROI (utility-meter):** Mean IoU=..., Detection rate=...
- **YOLO ROI (waterMeter):** Mean IoU=..., Detection rate=...
- **Inference:** ... ms/image
- **Next step:** compare with Faster R-CNN and U-Net
```

- [ ] **Step 2: Remove `.gitkeep`**

```bash
rm Notebooks/02_roi_detection/.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add Notebooks/02_roi_detection/yolo_roi.ipynb
git add -u Notebooks/02_roi_detection/.gitkeep
git commit -m "feat: ROI notebook — YOLO single-class detection"
```

---

## Task 4: Notebook — Faster R-CNN (`faster_rcnn.ipynb`)

**Files:**
- Create: `Notebooks/02_roi_detection/faster_rcnn.ipynb`

- [ ] **Step 1: Create notebook with all cells**

Same format conventions as Task 3.

**Cell 0 (markdown):**
```markdown
# 02 — ROI Detection: Faster R-CNN (Detectron2)

Detect the "Reading Digit" window (ROI) using Faster R-CNN with FPN backbone (Detectron2).
Two experiments: utility-meter dataset (sparse ROI, COCO format) and waterMeterDataset (polygon ROI).
```

**Cell 1 (code) — Setup:**
```python
import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB:
    try:
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        pass

if IN_COLAB:
    from google.colab import drive, userdata

    drive.mount("/content/drive")

    token = userdata.get("GITHUB_TOKEN", "")
    base = f"https://{token}@github.com" if token else "https://github.com"
    if not Path("/content/WaterMeterCV").exists():
        subprocess.run(
            ["git", "clone", f"{base}/UrranQx/WaterMeterCV.git", "/content/WaterMeterCV"],
            check=True
        )

    BRANCH = "feature/roi-detection"
    subprocess.run(["git", "-C", "/content/WaterMeterCV", "checkout", BRANCH], check=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/facebookresearch/detectron2.git",
         "rapidfuzz", "shapely"],
        check=True
    )

    ROOT         = Path("/content/WaterMeterCV")
    DATA_ROOT    = Path("/content/drive/MyDrive/WaterMeterCV/WaterMetricsDATA")
    WEIGHTS_BASE = Path("/content/drive/MyDrive/WaterMeterCV/weights")
    RESULTS_DIR  = Path("/content/drive/MyDrive/WaterMeterCV/results")
    WORKERS = 2
else:
    ROOT         = Path("../..").resolve()
    DATA_ROOT    = ROOT / "WaterMetricsDATA"
    WEIGHTS_BASE = ROOT / "models/weights"
    RESULTS_DIR  = ROOT / "results"
    WORKERS = 0

sys.path.insert(0, str(ROOT))
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import time
import csv
import torch
import cv2
import numpy as np
from datetime import datetime

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from models.data.unified_loader import load_water_meter_dataset_split
from models.data.roi_dataset import polygon_to_bbox
from models.metrics.evaluation import compute_iou_bbox

print(f"ROOT: {ROOT}")
print(f"Detectron2: {detectron2.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 2 (code) — Config:**
```python
COCO_ROOT = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.coco"
WM_PATH = DATA_ROOT / "waterMeterDataset/WaterMeters"

WEIGHTS_DIR = WEIGHTS_BASE / "roi_faster_rcnn"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Detectron2 hyperparams
MAX_ITER_UM = 3000
MAX_ITER_WM = 5000
BASE_LR = 0.0025
IMS_PER_BATCH = 4
ROI_BATCH_SIZE = 128

RUN_NAME = f"frcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"Run: {RUN_NAME}")
```

**Cell 3 (code) — Data Prep:**
```python
def get_um_roi_dicts(split: str):
    """Load COCO annotations, filter to 'Reading Digit' (category 11)."""
    json_path = COCO_ROOT / split / "_annotations.coco.json"
    with open(json_path) as f:
        coco = json.load(f)

    roi_cat_id = next(c["id"] for c in coco["categories"] if c["name"] == "Reading Digit")
    img_lookup = {img["id"]: img for img in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] == roi_cat_id:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    dataset_dicts = []
    for img_id, anns in anns_by_img.items():
        img_info = img_lookup[img_id]
        record = {
            "file_name": str(COCO_ROOT / split / img_info["file_name"]),
            "image_id": img_id,
            "height": img_info["height"],
            "width": img_info["width"],
            "annotations": [],
        }
        for ann in anns:
            record["annotations"].append({
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
            })
        dataset_dicts.append(record)
    return dataset_dicts


def get_wm_roi_dicts(samples):
    """Convert WM samples to Detectron2-format dicts."""
    dataset_dicts = []
    for i, s in enumerate(samples):
        if s.roi_polygon is None:
            continue
        img = cv2.imread(str(s.image_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        cx, cy, bw, bh = polygon_to_bbox(s.roi_polygon)
        x = (cx - bw / 2) * w
        y = (cy - bh / 2) * h

        record = {
            "file_name": str(s.image_path),
            "image_id": i,
            "height": h,
            "width": w,
            "annotations": [{
                "bbox": [x, y, bw * w, bh * h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
            }],
        }
        dataset_dicts.append(record)
    return dataset_dicts


# Register UM datasets
for split in ["train", "valid", "test"]:
    name = f"um_roi_{split}"
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, lambda s=split: get_um_roi_dicts(s))
    MetadataCatalog.get(name).set(thing_classes=["ROI"])

# WM split
wm_train, wm_test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)

for split_name, samples in [("train", wm_train), ("test", wm_test)]:
    name = f"wm_roi_{split_name}"
    if name in DatasetCatalog:
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, lambda s=samples: get_wm_roi_dicts(s))
    MetadataCatalog.get(name).set(thing_classes=["ROI"])

print(f"UM: {len(get_um_roi_dicts('train'))} train, {len(get_um_roi_dicts('test'))} test")
print(f"WM: {len(wm_train)} train, {len(wm_test)} test")
```

**Cell 4 (markdown):**
```markdown
## Experiment 1: utility-meter dataset
```

**Cell 5 (code) — Train UM:**
```python
cfg_um = get_cfg()
cfg_um.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_um.DATASETS.TRAIN = ("um_roi_train",)
cfg_um.DATASETS.TEST = ("um_roi_valid",)
cfg_um.DATALOADER.NUM_WORKERS = WORKERS
cfg_um.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg_um.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg_um.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg_um.SOLVER.BASE_LR = BASE_LR
cfg_um.SOLVER.MAX_ITER = MAX_ITER_UM
cfg_um.SOLVER.STEPS = (2000, 2500)
cfg_um.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = ROI_BATCH_SIZE
cfg_um.OUTPUT_DIR = str(WEIGHTS_DIR / f"um_{RUN_NAME}")

import os
os.makedirs(cfg_um.OUTPUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer_um = DefaultTrainer(cfg_um)
trainer_um.resume_or_load(resume=False)
trainer_um.train()
print(f"UM training complete. Weights: {cfg_um.OUTPUT_DIR}")
```

**Cell 6 (markdown):**
```markdown
## Evaluation (utility-meter)
```

**Cell 7 (code) — Eval UM:**
```python
cfg_um.MODEL.WEIGHTS = os.path.join(cfg_um.OUTPUT_DIR, "model_final.pth")
cfg_um.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
um_predictor = DefaultPredictor(cfg_um)

um_test_dicts = get_um_roi_dicts("test")
um_ious = []
um_inference_times = []
for d in um_test_dicts:
    img = cv2.imread(d["file_name"])
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    outputs = um_predictor(img)
    um_inference_times.append((time.perf_counter() - t0) * 1000)

    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    if len(pred_boxes) > 0:
        px1, py1, px2, py2 = pred_boxes[0]  # top-1 by confidence
        pred_cxcywh = ((px1+px2)/(2*w), (py1+py2)/(2*h), (px2-px1)/w, (py2-py1)/h)

        gt = d["annotations"][0]["bbox"]  # [x, y, w, h] absolute
        gt_cxcywh = ((gt[0]+gt[2]/2)/w, (gt[1]+gt[3]/2)/h, gt[2]/w, gt[3]/h)

        um_ious.append(compute_iou_bbox(pred_cxcywh, gt_cxcywh))
    else:
        um_ious.append(0.0)

um_mean_iou = np.mean(um_ious) if um_ious else 0.0
um_det_rate = sum(1 for v in um_ious if v > 0) / len(um_ious) if um_ious else 0.0
um_avg_ms = np.mean(um_inference_times) if um_inference_times else 0.0

print(f"UM — Mean IoU: {um_mean_iou:.4f}")
print(f"UM — Detection rate: {um_det_rate:.4f} ({sum(1 for v in um_ious if v > 0)}/{len(um_ious)})")
print(f"UM — Avg inference: {um_avg_ms:.1f} ms/image")
```

**Cell 8 (markdown):**
```markdown
## Experiment 2: waterMeterDataset
```

**Cell 9 (code) — Train WM:**
```python
cfg_wm = get_cfg()
cfg_wm.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_wm.DATASETS.TRAIN = ("wm_roi_train",)
cfg_wm.DATASETS.TEST = ("wm_roi_test",)
cfg_wm.DATALOADER.NUM_WORKERS = WORKERS
cfg_wm.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg_wm.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg_wm.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg_wm.SOLVER.BASE_LR = BASE_LR
cfg_wm.SOLVER.MAX_ITER = MAX_ITER_WM
cfg_wm.SOLVER.STEPS = (3000, 4000)
cfg_wm.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = ROI_BATCH_SIZE
cfg_wm.OUTPUT_DIR = str(WEIGHTS_DIR / f"wm_{RUN_NAME}")

os.makedirs(cfg_wm.OUTPUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer_wm = DefaultTrainer(cfg_wm)
trainer_wm.resume_or_load(resume=False)
trainer_wm.train()
print(f"WM training complete. Weights: {cfg_wm.OUTPUT_DIR}")
```

**Cell 10 (code) — Eval WM:**
```python
cfg_wm.MODEL.WEIGHTS = os.path.join(cfg_wm.OUTPUT_DIR, "model_final.pth")
cfg_wm.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
wm_predictor = DefaultPredictor(cfg_wm)

wm_test_dicts = get_wm_roi_dicts(wm_test)
wm_ious = []
wm_inference_times = []
for d in wm_test_dicts:
    img = cv2.imread(d["file_name"])
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    outputs = wm_predictor(img)
    wm_inference_times.append((time.perf_counter() - t0) * 1000)

    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    if len(pred_boxes) > 0:
        px1, py1, px2, py2 = pred_boxes[0]
        pred_cxcywh = ((px1+px2)/(2*w), (py1+py2)/(2*h), (px2-px1)/w, (py2-py1)/h)

        gt = d["annotations"][0]["bbox"]
        gt_cxcywh = ((gt[0]+gt[2]/2)/w, (gt[1]+gt[3]/2)/h, gt[2]/w, gt[3]/h)

        wm_ious.append(compute_iou_bbox(pred_cxcywh, gt_cxcywh))
    else:
        wm_ious.append(0.0)

wm_mean_iou = np.mean(wm_ious) if wm_ious else 0.0
wm_det_rate = sum(1 for v in wm_ious if v > 0) / len(wm_ious) if wm_ious else 0.0
wm_avg_ms = np.mean(wm_inference_times) if wm_inference_times else 0.0

print(f"WM — Mean IoU: {wm_mean_iou:.4f}")
print(f"WM — Detection rate: {wm_det_rate:.4f} ({sum(1 for v in wm_ious if v > 0)}/{len(wm_ious)})")
print(f"WM — Avg inference: {wm_avg_ms:.1f} ms/image")
```

**Cell 11 (markdown):**
```markdown
## Predictions
```

**Cell 12 (code) — Visualization:**
```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Top row: UM test
for i, ax in enumerate(axes[0]):
    if i >= len(um_test_dicts):
        ax.axis("off")
        continue
    d = um_test_dicts[i]
    img = cv2.imread(d["file_name"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    # GT (green)
    gt = d["annotations"][0]["bbox"]
    gx, gy, gw, gh = int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])
    cv2.rectangle(img, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 2)

    # Pred (red)
    outputs = um_predictor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    if len(pred_boxes) > 0:
        px1, py1, px2, py2 = [int(v) for v in pred_boxes[0]]
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    ax.imshow(img)
    iou_val = um_ious[i] if i < len(um_ious) else 0
    ax.set_title(f"UM IoU={iou_val:.2f}", fontsize=10)
    ax.axis("off")

# Bottom row: WM test
for i, ax in enumerate(axes[1]):
    if i >= len(wm_test_dicts):
        ax.axis("off")
        continue
    d = wm_test_dicts[i]
    img = cv2.imread(d["file_name"])
    if img is None:
        ax.axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img.shape[:2]

    gt = d["annotations"][0]["bbox"]
    gx, gy, gw, gh = int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])
    cv2.rectangle(img, (gx, gy), (gx+gw, gy+gh), (0, 255, 0), 2)

    outputs = wm_predictor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    if len(pred_boxes) > 0:
        px1, py1, px2, py2 = [int(v) for v in pred_boxes[0]]
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    ax.imshow(img)
    iou_val = wm_ious[i] if i < len(wm_ious) else 0
    ax.set_title(f"WM IoU={iou_val:.2f}", fontsize=10)
    ax.axis("off")

plt.suptitle("Faster R-CNN ROI — Green=GT, Red=Pred", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roi_faster_rcnn_predictions.png", dpi=150)
plt.close()
print("Saved to results/roi_faster_rcnn_predictions.png")
```

**Cell 13 (markdown):**
```markdown
## Comparison
```

**Cell 14 (code) — Compare:**
```python
print(f"{'='*60}")
print(f"{'Metric':<25} {'utility-meter':>15} {'waterMeter':>15}")
print(f"{'='*60}")
print(f"{'Mean IoU':<25} {um_mean_iou:>15.4f} {wm_mean_iou:>15.4f}")
print(f"{'Detection rate':<25} {um_det_rate:>15.4f} {wm_det_rate:>15.4f}")
print(f"{'Avg inference (ms)':<25} {um_avg_ms:>15.1f} {wm_avg_ms:>15.1f}")
print(f"{'N test':<25} {len(um_test_dicts):>15d} {len(wm_test_dicts):>15d}")
print(f"{'='*60}")
```

**Cell 15 (markdown):**
```markdown
## Save Results
```

**Cell 16 (code) — Save:**
```python
metrics = {
    "method": "faster_rcnn",
    "utility_meter": {
        "mean_iou": round(um_mean_iou, 4),
        "detection_rate": round(um_det_rate, 4),
        "avg_inference_ms": round(um_avg_ms, 1),
        "n_train": len(get_um_roi_dicts("train")),
        "n_test": len(um_test_dicts),
    },
    "water_meter": {
        "mean_iou": round(wm_mean_iou, 4),
        "detection_rate": round(wm_det_rate, 4),
        "avg_inference_ms": round(wm_avg_ms, 1),
        "n_train": len(wm_train),
        "n_test": len(wm_test),
    },
    "config": {
        "max_iter_um": MAX_ITER_UM,
        "max_iter_wm": MAX_ITER_WM,
        "base_lr": BASE_LR,
        "ims_per_batch": IMS_PER_BATCH,
    },
    "run_date": datetime.now().isoformat(),
}

with open(RESULTS_DIR / "roi_faster_rcnn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

csv_path = RESULTS_DIR / "roi_comparison.csv"
csv_exists = csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow([
            "method", "um_mean_iou", "um_detection_rate", "um_inference_ms",
            "wm_mean_iou", "wm_detection_rate", "wm_inference_ms", "run_date",
        ])
    writer.writerow([
        "faster_rcnn",
        round(um_mean_iou, 4), round(um_det_rate, 4), round(um_avg_ms, 1),
        round(wm_mean_iou, 4), round(wm_det_rate, 4), round(wm_avg_ms, 1),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    ])

print(f"Metrics -> {RESULTS_DIR / 'roi_faster_rcnn_metrics.json'}")
print(f"CSV    -> {csv_path}")
```

**Cell 17 (markdown):**
```markdown
## Conclusions

*(Fill after running)*

- **Faster R-CNN (utility-meter):** Mean IoU=..., Detection rate=...
- **Faster R-CNN (waterMeter):** Mean IoU=..., Detection rate=...
- **Inference:** ... ms/image
- **Next step:** compare with YOLO and U-Net
```

- [ ] **Step 2: Commit**

```bash
git add Notebooks/02_roi_detection/faster_rcnn.ipynb
git commit -m "feat: ROI notebook — Faster R-CNN (Detectron2)"
```

---

## Task 5: Notebook — U-Net Segmentation (`segmentation_unet.ipynb`)

**Files:**
- Create: `Notebooks/02_roi_detection/segmentation_unet.ipynb`

- [ ] **Step 1: Create notebook with all cells**

**Cell 0 (markdown):**
```markdown
# 02 — ROI Detection: U-Net Segmentation

Binary segmentation of the ROI (reading window) using U-Net with ResNet34 encoder.
Predicted mask is converted to bounding box for IoU comparison with detection-based methods.
Two experiments: utility-meter dataset (bbox -> mask) and waterMeterDataset (polygon -> mask).
```

**Cell 1 (code) — Setup:**
```python
import sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB:
    try:
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        pass

if IN_COLAB:
    from google.colab import drive, userdata

    drive.mount("/content/drive")

    token = userdata.get("GITHUB_TOKEN", "")
    base = f"https://{token}@github.com" if token else "https://github.com"
    if not Path("/content/WaterMeterCV").exists():
        subprocess.run(
            ["git", "clone", f"{base}/UrranQx/WaterMeterCV.git", "/content/WaterMeterCV"],
            check=True
        )

    BRANCH = "feature/roi-detection"
    subprocess.run(["git", "-C", "/content/WaterMeterCV", "checkout", BRANCH], check=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "segmentation-models-pytorch", "albumentations",
         "rapidfuzz", "shapely"],
        check=True
    )

    ROOT         = Path("/content/WaterMeterCV")
    DATA_ROOT    = Path("/content/drive/MyDrive/WaterMeterCV/WaterMetricsDATA")
    WEIGHTS_BASE = Path("/content/drive/MyDrive/WaterMeterCV/weights")
    RESULTS_DIR  = Path("/content/drive/MyDrive/WaterMeterCV/results")
    WORKERS = 2
else:
    ROOT         = Path("../..").resolve()
    DATA_ROOT    = ROOT / "WaterMetricsDATA"
    WEIGHTS_BASE = ROOT / "models/weights"
    RESULTS_DIR  = ROOT / "results"
    WORKERS = 0

sys.path.insert(0, str(ROOT))
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from datetime import datetime
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.data.unified_loader import (
    load_utility_meter_dataset,
    load_water_meter_dataset_split,
)
from models.data.roi_dataset import polygon_to_bbox, filter_utility_meter_roi_samples
from models.metrics.evaluation import compute_iou_bbox

print(f"ROOT: {ROOT}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 2 (code) — Config:**
```python
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
WM_PATH = DATA_ROOT / "waterMeterDataset/WaterMeters"

WEIGHTS_DIR = WEIGHTS_BASE / "roi_unet"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_NAME = f"unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"Device: {DEVICE}, Run: {RUN_NAME}")
```

**Cell 3 (code) — Dataset class + transforms:**
```python
class ROISegmentationDataset(Dataset):
    """Binary segmentation dataset: image + ROI mask."""

    def __init__(self, samples, img_size=256, transform=None, source="bbox"):
        """
        Args:
            samples: list of (image_path, roi_bbox) tuples OR UnifiedSample objects
            source: "bbox" — samples are (path, bbox) tuples
                    "polygon" — samples are UnifiedSample with roi_polygon
        """
        self.img_size = img_size
        self.transform = transform
        self.source = source
        if source == "bbox":
            self.items = [(p, b) for p, b in samples]
        else:
            self.items = [(s.image_path, s.roi_polygon) for s in samples
                          if s.roi_polygon is not None]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, roi = self.items[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        if self.source == "bbox":
            cx, cy, bw, bh = roi
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            mask[max(0, y1):y2, max(0, x1):x2] = 1
        else:  # polygon
            pts = np.array([(int(x * w), int(y * h)) for x, y in roi], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        return img, mask.unsqueeze(0).float()


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

**Cell 4 (code) — Data Prep:**
```python
# UM
um_train_roi = filter_utility_meter_roi_samples(UM_YOLO_PATH, "train")
um_test_roi = filter_utility_meter_roi_samples(UM_YOLO_PATH, "test")
print(f"UM ROI: {len(um_train_roi)} train, {len(um_test_roi)} test")

# WM
wm_train, wm_test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
print(f"WM: {len(wm_train)} train, {len(wm_test)} test")
```

**Cell 5 (code) — Train helper:**
```python
def train_unet(train_dataset, val_dataset, tag, epochs=EPOCHS):
    """Train U-Net and return best model path."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=WORKERS)

    best_val_loss = float("inf")
    best_path = WEIGHTS_DIR / f"{tag}_best.pt"

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = bce(preds, masks) + dice(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_loss += (bce(preds, masks) + dice(preds, masks)).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} — train: {avg_train:.4f}, val: {avg_val:.4f}")

    print(f"  Best val loss: {best_val_loss:.4f}, saved to {best_path}")
    return best_path


def mask_to_bbox_normalized(mask_np):
    """Convert binary mask (H, W) to (cx, cy, w, h) normalized bbox."""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    h, w = mask_np.shape
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return (cx, cy, bw, bh)


def eval_unet(model, dataset, gt_bboxes):
    """Evaluate model, return (ious, inference_times_ms)."""
    model.eval()
    ious = []
    times_ms = []

    for i in range(len(dataset)):
        img_tensor, _ = dataset[i]
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(img_batch)
        times_ms.append((time.perf_counter() - t0) * 1000)

        pred_mask = (torch.sigmoid(pred[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
        pred_bbox = mask_to_bbox_normalized(pred_mask)

        if pred_bbox is not None and gt_bboxes[i] is not None:
            ious.append(compute_iou_bbox(pred_bbox, gt_bboxes[i]))
        else:
            ious.append(0.0)

    return ious, times_ms
```

**Cell 6 (markdown):**
```markdown
## Experiment 1: utility-meter dataset
```

**Cell 7 (code) — Train + Eval UM:**
```python
um_train_ds = ROISegmentationDataset(um_train_roi, IMG_SIZE, train_transform, source="bbox")
um_test_ds = ROISegmentationDataset(um_test_roi, IMG_SIZE, val_transform, source="bbox")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Training U-Net on UM...")
um_best_path = train_unet(um_train_ds, um_test_ds, f"um_{RUN_NAME}")

um_model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1).to(DEVICE)
um_model.load_state_dict(torch.load(um_best_path, map_location=DEVICE, weights_only=True))

# GT bboxes for test
um_gt_bboxes = [bbox for _, bbox in um_test_roi]
um_ious, um_inference_times = eval_unet(um_model, um_test_ds, um_gt_bboxes)

um_mean_iou = np.mean(um_ious) if um_ious else 0.0
um_det_rate = sum(1 for v in um_ious if v > 0) / len(um_ious) if um_ious else 0.0
um_avg_ms = np.mean(um_inference_times) if um_inference_times else 0.0

print(f"\nUM — Mean IoU: {um_mean_iou:.4f}")
print(f"UM — Detection rate: {um_det_rate:.4f}")
print(f"UM — Avg inference: {um_avg_ms:.1f} ms/image")
```

**Cell 8 (markdown):**
```markdown
## Experiment 2: waterMeterDataset
```

**Cell 9 (code) — Train + Eval WM:**
```python
wm_train_ds = ROISegmentationDataset(wm_train, IMG_SIZE, train_transform, source="polygon")
wm_test_ds = ROISegmentationDataset(wm_test, IMG_SIZE, val_transform, source="polygon")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Training U-Net on WM...")
wm_best_path = train_unet(wm_train_ds, wm_test_ds, f"wm_{RUN_NAME}")

wm_model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1).to(DEVICE)
wm_model.load_state_dict(torch.load(wm_best_path, map_location=DEVICE, weights_only=True))

# GT bboxes from polygons
wm_gt_bboxes = [polygon_to_bbox(s.roi_polygon) if s.roi_polygon else None for s in wm_test]
wm_ious, wm_inference_times = eval_unet(wm_model, wm_test_ds, wm_gt_bboxes)

wm_mean_iou = np.mean(wm_ious) if wm_ious else 0.0
wm_det_rate = sum(1 for v in wm_ious if v > 0) / len(wm_ious) if wm_ious else 0.0
wm_avg_ms = np.mean(wm_inference_times) if wm_inference_times else 0.0

print(f"\nWM — Mean IoU: {wm_mean_iou:.4f}")
print(f"WM — Detection rate: {wm_det_rate:.4f}")
print(f"WM — Avg inference: {wm_avg_ms:.1f} ms/image")
```

**Cell 10 (markdown):**
```markdown
## Predictions
```

**Cell 11 (code) — Visualization:**
```python
def show_mask_predictions(model, dataset, gt_bboxes, axes_row, title_prefix):
    """Draw mask overlay + bbox on a row of axes."""
    for i, ax in enumerate(axes_row):
        if i >= len(dataset):
            ax.axis("off")
            continue
        img_tensor, gt_mask_tensor = dataset[i]

        # Denormalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 1)

        # Predict
        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(DEVICE))
        pred_mask = (torch.sigmoid(pred[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)

        # Overlay: red = pred mask
        overlay = img_np.copy()
        overlay[pred_mask > 0, 0] = np.minimum(overlay[pred_mask > 0, 0] + 0.3, 1.0)

        ax.imshow(overlay)
        ax.set_title(f"{title_prefix} #{i}", fontsize=10)
        ax.axis("off")


fig, axes = plt.subplots(2, 4, figsize=(20, 10))
show_mask_predictions(um_model, um_test_ds, um_gt_bboxes, axes[0], "UM")
show_mask_predictions(wm_model, wm_test_ds, wm_gt_bboxes, axes[1], "WM")

plt.suptitle("U-Net ROI Segmentation — Red overlay = predicted mask", fontsize=14)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "roi_unet_predictions.png", dpi=150)
plt.close()
print("Saved to results/roi_unet_predictions.png")
```

**Cell 12 (markdown):**
```markdown
## Comparison
```

**Cell 13 (code) — Compare:**
```python
print(f"{'='*60}")
print(f"{'Metric':<25} {'utility-meter':>15} {'waterMeter':>15}")
print(f"{'='*60}")
print(f"{'Mean IoU':<25} {um_mean_iou:>15.4f} {wm_mean_iou:>15.4f}")
print(f"{'Detection rate':<25} {um_det_rate:>15.4f} {wm_det_rate:>15.4f}")
print(f"{'Avg inference (ms)':<25} {um_avg_ms:>15.1f} {wm_avg_ms:>15.1f}")
print(f"{'N test':<25} {len(um_test_roi):>15d} {len(wm_test):>15d}")
print(f"{'='*60}")
```

**Cell 14 (markdown):**
```markdown
## Save Results
```

**Cell 15 (code) — Save:**
```python
metrics = {
    "method": "unet_segmentation",
    "utility_meter": {
        "mean_iou": round(um_mean_iou, 4),
        "detection_rate": round(um_det_rate, 4),
        "avg_inference_ms": round(um_avg_ms, 1),
        "n_train": len(um_train_roi),
        "n_test": len(um_test_roi),
    },
    "water_meter": {
        "mean_iou": round(wm_mean_iou, 4),
        "detection_rate": round(wm_det_rate, 4),
        "avg_inference_ms": round(wm_avg_ms, 1),
        "n_train": len(wm_train),
        "n_test": len(wm_test),
    },
    "config": {
        "encoder": "resnet34",
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    },
    "run_date": datetime.now().isoformat(),
}

with open(RESULTS_DIR / "roi_unet_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

csv_path = RESULTS_DIR / "roi_comparison.csv"
csv_exists = csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not csv_exists:
        writer.writerow([
            "method", "um_mean_iou", "um_detection_rate", "um_inference_ms",
            "wm_mean_iou", "wm_detection_rate", "wm_inference_ms", "run_date",
        ])
    writer.writerow([
        "unet_segmentation",
        round(um_mean_iou, 4), round(um_det_rate, 4), round(um_avg_ms, 1),
        round(wm_mean_iou, 4), round(wm_det_rate, 4), round(wm_avg_ms, 1),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    ])

print(f"Metrics -> {RESULTS_DIR / 'roi_unet_metrics.json'}")
print(f"CSV    -> {csv_path}")
```

**Cell 16 (markdown):**
```markdown
## Conclusions

*(Fill after running)*

- **U-Net (utility-meter):** Mean IoU=..., Detection rate=...
- **U-Net (waterMeter):** Mean IoU=..., Detection rate=...
- **Inference:** ... ms/image
- **Next step:** compare with YOLO and Faster R-CNN
```

- [ ] **Step 2: Commit**

```bash
git add Notebooks/02_roi_detection/segmentation_unet.ipynb
git commit -m "feat: ROI notebook — U-Net segmentation"
```

---

## Execution Order

1. **Task 1** — WM split (unified_loader) — required by all notebooks
2. **Task 2** — ROI helpers (roi_dataset.py) — required by all notebooks
3. **Task 3** — YOLO ROI notebook — simplest, reuses most from baseline
4. **Task 4** — Faster R-CNN notebook — Detectron2, Colab-only training likely
5. **Task 5** — U-Net notebook — custom training loop, most code
