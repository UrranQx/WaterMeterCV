# WaterMeterCV ML Research — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Последовательно протестировать 11 ML-подходов для считывания показаний водяных счётчиков и выбрать лучший пайплайн.

**Architecture:** Shared infrastructure (`models/data/`, `models/metrics/`, `models/utils/`) + изолированные Jupyter-ноутбуки для каждого эксперимента. Ноутбуки импортируют shared-код, пригодны для запуска в Colab.

**Tech Stack:** Python 3.13, uv, PyTorch, Ultralytics (YOLOv11), Detectron2, torchvision, pandas, matplotlib

**Datasets:**
- `WaterMetricsDATA/waterMeterDataset/WaterMeters/` — 1244 images, `data.csv` (value + ROI polygon, normalized coords)
- `WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i/` — 1555 train / 194 valid / 194 test, YOLO + COCO formats, 14 classes (0-9, Reading Digit, black, red, white), bbox annotations

---

## File Map

```
WaterMeterCV/
├── models/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── unified_loader.py      # Task 2: единый loader для обоих датасетов
│   │   └── augmentations.py       # Task 2: аугментации (albumentations)
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── evaluation.py          # Task 3: full-string acc, per-digit acc, CER, IoU, timing
│   └── utils/
│       ├── __init__.py
│       └── visualization.py       # Task 4: отрисовка предсказаний, comparison plots
├── Notebooks/
│   ├── 00_data_exploration.ipynb           # Task 5
│   ├── 01_baseline/
│   │   └── yolo_single_stage.ipynb         # Task 6
│   ├── 02_roi_detection/
│   │   ├── faster_rcnn.ipynb               # Task 8
│   │   ├── yolo_roi.ipynb                  # Task 9
│   │   └── segmentation_unet.ipynb         # Task 10
│   ├── 03_ocr/
│   │   ├── crnn_ctc.ipynb                  # Task 12
│   │   ├── transformer_ocr.ipynb           # Task 13
│   │   ├── cnn_ctc.ipynb                   # Task 14
│   │   └── per_digit_classifier.ipynb      # Task 15
│   └── 04_combinations/
│       ├── craft_crnn.ipynb                # Task 16
│       ├── maskrcnn_decoder.ipynb          # Task 17
│       └── detectron2_ocr.ipynb            # Task 18
├── configs/
│   └── default.yaml                        # Task 1: пути, гиперпараметры
├── results/
│   └── comparison.md                       # Task 19
├── tests/
│   ├── test_unified_loader.py              # Task 2
│   ├── test_evaluation.py                  # Task 3
│   └── test_visualization.py               # Task 4
├── models/weights/                         # gitignored
├── models/checkpoints/                     # gitignored
└── CLAUDE.md                               # Task 20
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `configs/default.yaml`, directory structure, update `.gitignore`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p models/data models/metrics models/utils models/weights models/checkpoints
mkdir -p Notebooks/01_baseline Notebooks/02_roi_detection Notebooks/03_ocr Notebooks/04_combinations
mkdir -p configs results tests
```

- [ ] **Step 2: Create `__init__.py` files**

```python
# models/__init__.py — empty
# models/data/__init__.py — empty
# models/metrics/__init__.py — empty
# models/utils/__init__.py — empty
```

- [ ] **Step 3: Update `.gitignore`**

Append:
```gitignore
models/weights/
models/checkpoints/
results/*.png
results/*.jpg
__pycache__/
*.pyc
.ipynb_checkpoints/
```

- [ ] **Step 4: Create `configs/default.yaml`**

```yaml
paths:
  water_meter_dataset: "WaterMetricsDATA/waterMeterDataset/WaterMeters"
  utility_meter_yolo: "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
  utility_meter_coco: "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.coco"
  weights_dir: "models/weights"
  checkpoints_dir: "models/checkpoints"
  results_dir: "results"

training:
  batch_size: 16
  epochs: 50
  learning_rate: 0.001
  img_size: 640
  device: "cuda"  # or "cpu"

evaluation:
  iou_threshold: 0.5
  confidence_threshold: 0.25

# COCO category mapping (utility-meter dataset)
# 0: Digits (parent), 1-10: digits 0-9, 11: Reading Digit, 12: black, 13: red, 14: white
utility_meter_digit_classes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # COCO ids for 0-9
utility_meter_roi_class: 11  # "Reading Digit" = ROI bbox
```

- [ ] **Step 5: Update `pyproject.toml` — add test + research dependencies**

```toml
[project]
name = "watermetercv"
version = "0.1.0"
description = "CV pipeline for water meter reading extraction"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "ipykernel>=7.2.0",
    "matplotlib>=3.10.8",
    "pandas>=3.0.2",
    "pip>=26.0.1",
    "torch>=2.11.0",
    "torchvision>=0.26.0",
    "ultralytics>=8.4.33",
    "albumentations>=2.0.0",
    "pyyaml>=6.0",
    "opencv-python>=4.10.0",
    "Pillow>=11.0.0",
    "editdistance>=0.8.0",
]

[project.optional-dependencies]
test = ["pytest>=8.0.0"]
detectron2 = ["detectron2"]
transformers = ["transformers>=4.40.0", "sentencepiece>=0.2.0"]
craft = ["craft-text-detector>=0.4.3"]
```

- [ ] **Step 6: Sync dependencies**

Run: `uv sync`
Expected: all dependencies installed without errors.

- [ ] **Step 7: Create `develop` branch and commit scaffolding**

```bash
git checkout -b develop
git add models/__init__.py models/data/__init__.py models/metrics/__init__.py models/utils/__init__.py
git add configs/default.yaml .gitignore pyproject.toml Notebooks/ results/ tests/
git commit -m "feat: project scaffolding — directory structure, config, dependencies"
```

---

## Task 2: Unified Data Loader

**Files:**
- Create: `models/data/unified_loader.py`, `models/data/augmentations.py`, `tests/test_unified_loader.py`

- [ ] **Step 1: Write failing tests for unified loader**

```python
# tests/test_unified_loader.py
import pytest
from pathlib import Path
from models.data.unified_loader import (
    load_water_meter_dataset,
    load_utility_meter_dataset,
    UnifiedSample,
)


DATA_ROOT = Path("WaterMetricsDATA")
WM_PATH = DATA_ROOT / "waterMeterDataset" / "WaterMeters"
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"


@pytest.fixture
def wm_samples():
    return load_water_meter_dataset(WM_PATH)


@pytest.fixture
def um_samples():
    return load_utility_meter_dataset(UM_YOLO_PATH, split="train")


class TestUnifiedSample:
    def test_wm_returns_list_of_unified_samples(self, wm_samples):
        assert len(wm_samples) > 0
        assert isinstance(wm_samples[0], UnifiedSample)

    def test_wm_sample_has_required_fields(self, wm_samples):
        s = wm_samples[0]
        assert s.image_path.exists()
        assert isinstance(s.value, float)
        assert s.roi_polygon is not None  # list of (x, y) normalized
        assert len(s.roi_polygon) >= 3

    def test_um_returns_list_of_unified_samples(self, um_samples):
        assert len(um_samples) > 0
        assert isinstance(um_samples[0], UnifiedSample)

    def test_um_sample_has_digit_bboxes(self, um_samples):
        s = um_samples[0]
        assert s.image_path.exists()
        assert s.digit_bboxes is not None  # list of (class_id, cx, cy, w, h)
        assert len(s.digit_bboxes) > 0

    def test_um_value_is_none_when_not_available(self, um_samples):
        """utility-meter dataset has digit bboxes but no single ground truth value."""
        s = um_samples[0]
        # value can be reconstructed from digit bboxes sorted by x-coordinate
        # but raw dataset doesn't provide it as a single field
        assert s.value is None or isinstance(s.value, float)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `uv run pytest tests/test_unified_loader.py -v`
Expected: ImportError — `models.data.unified_loader` does not exist yet.

- [ ] **Step 3: Implement unified loader**

```python
# models/data/unified_loader.py
from dataclasses import dataclass, field
from pathlib import Path
import csv
import ast


@dataclass
class UnifiedSample:
    """Unified representation of a single dataset sample."""
    image_path: Path
    value: float | None = None               # ground truth reading (e.g. 595.825)
    roi_polygon: list[tuple[float, float]] | None = None  # [(x, y), ...] normalized
    roi_bbox: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    digit_bboxes: list[tuple[int, float, float, float, float]] | None = None  # [(class_id, cx, cy, w, h), ...]
    mask_path: Path | None = None
    dataset_source: str = ""


def load_water_meter_dataset(root: Path) -> list[UnifiedSample]:
    """Load waterMeterDataset from data.csv.

    CSV columns: photo_name, value, location (polygon as Python dict string).
    """
    root = Path(root)
    csv_path = root / "data.csv"
    images_dir = root / "images"
    masks_dir = root / "masks"
    samples = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            photo_name = row["photo_name"]
            value = float(row["value"])

            # Parse polygon from string like "{'type': 'polygon', 'data': [{'x': ..., 'y': ...}, ...]}"
            location = ast.literal_eval(row["location"])
            polygon = [(pt["x"], pt["y"]) for pt in location["data"]]

            image_path = images_dir / photo_name
            mask_path = masks_dir / photo_name
            if not mask_path.exists():
                mask_path = None

            samples.append(UnifiedSample(
                image_path=image_path,
                value=value,
                roi_polygon=polygon,
                mask_path=mask_path,
                dataset_source="water_meter_dataset",
            ))

    return samples


def load_utility_meter_dataset(
    root: Path,
    split: str = "train",
) -> list[UnifiedSample]:
    """Load utility-meter dataset from YOLO format labels.

    Label format per line: class_id cx cy w h (normalized).
    Classes: 0=0, 1=1, ..., 9=9, 10=Reading Digit, 11=black, 12=red, 13=white
    (YOLO data.yaml remaps: COCO ids offset by 1 vs YOLO class indices)

    In YOLO format the class indices are:
    0-9: digits '0'-'9', 10: Reading Digit, 11: black, 12: red, 13: white
    """
    root = Path(root)
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    samples = []

    if not images_dir.exists():
        return samples

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        digit_bboxes = []
        roi_bbox = None

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    if cls_id == 10:  # Reading Digit = ROI
                        roi_bbox = (cx, cy, w, h)
                    elif cls_id <= 9:  # digit 0-9
                        digit_bboxes.append((cls_id, cx, cy, w, h))

        # Reconstruct value from digit bboxes sorted left-to-right by cx
        value = None
        if digit_bboxes:
            sorted_digits = sorted(digit_bboxes, key=lambda d: d[1])  # sort by cx
            value_str = "".join(str(d[0]) for d in sorted_digits)
            # Note: no decimal point info from bboxes alone — value is approximate
            # The "black"/"red"/"white" classes might distinguish integer vs decimal digits
            # but this is dataset-specific logic to refine in EDA
            try:
                value = float(value_str)
            except ValueError:
                value = None

        samples.append(UnifiedSample(
            image_path=img_path,
            value=value,
            roi_bbox=roi_bbox,
            digit_bboxes=digit_bboxes if digit_bboxes else None,
            dataset_source="utility_meter",
        ))

    return samples
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `uv run pytest tests/test_unified_loader.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Implement augmentations**

```python
# models/data/augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.HorizontalFlip(p=0.0),  # meters shouldn't be flipped
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
```

- [ ] **Step 6: Commit**

```bash
git add models/data/ tests/test_unified_loader.py
git commit -m "feat: unified data loader for both datasets + augmentations"
```

> **Note:** Tasks 2-5 all commit to the `feature/data-exploration` branch created in Task 1 step 7. This branch carries all shared infrastructure + EDA.

---

## Task 3: Evaluation Metrics

**Files:**
- Create: `models/metrics/evaluation.py`, `tests/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluation.py
import pytest
from models.metrics.evaluation import (
    full_string_accuracy,
    per_digit_accuracy,
    character_error_rate,
    compute_iou_polygon,
    compute_iou_bbox,
)


class TestFullStringAccuracy:
    def test_perfect(self):
        preds = ["123.456", "789.012"]
        gts = ["123.456", "789.012"]
        assert full_string_accuracy(preds, gts) == 1.0

    def test_half_correct(self):
        preds = ["123.456", "999.999"]
        gts = ["123.456", "789.012"]
        assert full_string_accuracy(preds, gts) == 0.5

    def test_empty(self):
        assert full_string_accuracy([], []) == 0.0


class TestPerDigitAccuracy:
    def test_perfect(self):
        assert per_digit_accuracy("12345", "12345") == 1.0

    def test_one_wrong(self):
        assert per_digit_accuracy("12345", "12346") == 4 / 5

    def test_different_lengths(self):
        # pad shorter with empty, count mismatches
        acc = per_digit_accuracy("123", "1234")
        assert 0.0 <= acc <= 1.0


class TestCER:
    def test_perfect(self):
        assert character_error_rate("12345", "12345") == 0.0

    def test_one_substitution(self):
        cer = character_error_rate("12345", "12346")
        assert 0.0 < cer < 1.0


class TestIoU:
    def test_identical_bbox(self):
        bbox = (0.5, 0.5, 0.2, 0.2)
        assert compute_iou_bbox(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap_bbox(self):
        a = (0.1, 0.1, 0.1, 0.1)
        b = (0.9, 0.9, 0.1, 0.1)
        assert compute_iou_bbox(a, b) == pytest.approx(0.0)

    def test_identical_polygon(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert compute_iou_polygon(poly, poly) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests — verify failure**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement metrics**

```python
# models/metrics/evaluation.py
import time
from typing import Callable
import editdistance


def full_string_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    """Fraction of predictions that exactly match ground truth."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
    return correct / len(predictions)


def per_digit_accuracy(pred: str, gt: str) -> float:
    """Per-character accuracy between two strings, aligned left-to-right."""
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0
    pred_padded = pred.ljust(max_len)
    gt_padded = gt.ljust(max_len)
    correct = sum(1 for p, g in zip(pred_padded, gt_padded) if p == g)
    return correct / max_len


def character_error_rate(pred: str, gt: str) -> float:
    """CER = edit_distance(pred, gt) / len(gt)."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)


def compute_iou_bbox(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU for two bboxes in (cx, cy, w, h) normalized format."""
    ax1 = a[0] - a[2] / 2
    ay1 = a[1] - a[3] / 2
    ax2 = a[0] + a[2] / 2
    ay2 = a[1] + a[3] / 2

    bx1 = b[0] - b[2] / 2
    by1 = b[1] - b[3] / 2
    bx2 = b[0] + b[2] / 2
    by2 = b[1] + b[3] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_iou_polygon(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> float:
    """IoU for two polygons using shapely."""
    from shapely.geometry import Polygon

    a = Polygon(poly_a)
    b = Polygon(poly_b)

    if not a.is_valid or not b.is_valid:
        return 0.0

    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def measure_inference_time(
    fn: Callable,
    *args,
    n_runs: int = 10,
    **kwargs,
) -> float:
    """Average inference time in milliseconds."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)
```

- [ ] **Step 4: Add `shapely` to `pyproject.toml` dependencies**

```toml
"shapely>=2.0.0",
```

Run: `uv sync`

- [ ] **Step 5: Run tests — verify pass**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add models/metrics/ tests/test_evaluation.py pyproject.toml
git commit -m "feat: evaluation metrics — full-string acc, per-digit acc, CER, IoU, timing"
```

---

## Task 4: Visualization Utilities

**Files:**
- Create: `models/utils/visualization.py`, `tests/test_visualization.py`

- [ ] **Step 1: Write test**

```python
# tests/test_visualization.py
import pytest
import numpy as np
from models.utils.visualization import draw_digit_bboxes, draw_roi_polygon


class TestVisualization:
    @pytest.fixture
    def dummy_image(self):
        return np.zeros((640, 640, 3), dtype=np.uint8)

    def test_draw_digit_bboxes_returns_image(self, dummy_image):
        bboxes = [(5, 0.3, 0.3, 0.05, 0.08), (3, 0.4, 0.3, 0.05, 0.08)]
        result = draw_digit_bboxes(dummy_image, bboxes)
        assert result.shape == dummy_image.shape

    def test_draw_roi_polygon_returns_image(self, dummy_image):
        polygon = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        result = draw_roi_polygon(dummy_image, polygon)
        assert result.shape == dummy_image.shape
```

- [ ] **Step 2: Run — verify fail**

Run: `uv run pytest tests/test_visualization.py -v`

- [ ] **Step 3: Implement**

```python
# models/utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


DIGIT_COLORS = {
    i: (0, 255, 0) for i in range(10)  # green for digits
}


def draw_digit_bboxes(
    image: np.ndarray,
    bboxes: list[tuple[int, float, float, float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw digit bounding boxes on image.

    Args:
        bboxes: list of (class_id, cx, cy, w, h) in normalized coords.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for cls_id, cx, cy, bw, bh in bboxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, str(cls_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def draw_roi_polygon(
    image: np.ndarray,
    polygon: list[tuple[float, float]],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw ROI polygon overlay on image.

    Args:
        polygon: list of (x, y) in normalized coords.
    """
    img = image.copy()
    h, w = img.shape[:2]
    pts = np.array([(int(x * w), int(y * h)) for x, y in polygon], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img


def plot_comparison_table(
    results: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> None:
    """Plot a comparison table of experiment results.

    Args:
        results: {experiment_name: {metric_name: value}}
    """
    experiments = list(results.keys())
    if not experiments:
        return

    metrics = list(results[experiments[0]].keys())
    data = [[results[exp].get(m, 0.0) for m in metrics] for exp in experiments]

    fig, ax = plt.subplots(figsize=(len(metrics) * 2, len(experiments) * 0.6 + 1))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{v:.4f}" for v in row] for row in data],
        rowLabels=experiments,
        colLabels=metrics,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
```

- [ ] **Step 4: Run — verify pass**

Run: `uv run pytest tests/test_visualization.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add models/utils/ tests/test_visualization.py
git commit -m "feat: visualization utilities — bbox/polygon drawing, comparison table"
```

---

## Task 5: Notebook 00 — Data Exploration (EDA)

**Files:**
- Create: `Notebooks/00_data_exploration.ipynb`

- [ ] **Step 1: Create EDA notebook with the following cells**

**Cell 1 — Setup:**
```python
import sys
from pathlib import Path

# Handle both local and Colab execution
ROOT = Path("..")
if not (ROOT / "models").exists():
    # Colab: clone repo
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/<USER>/WaterMeterCV.git"], check=True)
    ROOT = Path("WaterMeterCV")

sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from models.data.unified_loader import load_water_meter_dataset, load_utility_meter_dataset
from models.utils.visualization import draw_roi_polygon, draw_digit_bboxes
```

**Cell 2 — Load datasets:**
```python
wm_samples = load_water_meter_dataset(ROOT / "WaterMetricsDATA/waterMeterDataset/WaterMeters")
um_train = load_utility_meter_dataset(ROOT / "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11", split="train")
um_valid = load_utility_meter_dataset(ROOT / "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11", split="valid")
um_test = load_utility_meter_dataset(ROOT / "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11", split="test")

print(f"WaterMeter: {len(wm_samples)} samples")
print(f"UtilityMeter: {len(um_train)} train, {len(um_valid)} valid, {len(um_test)} test")
```

**Cell 3 — Value distribution (waterMeterDataset):**
```python
wm_values = [s.value for s in wm_samples if s.value is not None]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(wm_values, bins=50)
axes[0].set_title("waterMeterDataset — value distribution")
axes[0].set_xlabel("Meter reading")

# Number of digits distribution
digit_counts = [len(str(v).replace(".", "")) for v in wm_values]
axes[1].hist(digit_counts, bins=range(1, 12))
axes[1].set_title("Number of digits per reading")
plt.tight_layout()
plt.savefig(ROOT / "results" / "eda_value_distribution.png", dpi=150)
plt.show()
```

**Cell 4 — Visualize ROI samples (waterMeterDataset):**
```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, sample in zip(axes.flat, wm_samples[:8]):
    img = cv2.imread(str(sample.image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if sample.roi_polygon:
        img = draw_roi_polygon(img, sample.roi_polygon)
    ax.imshow(img)
    ax.set_title(f"val={sample.value}")
    ax.axis("off")
plt.suptitle("waterMeterDataset — ROI polygons")
plt.tight_layout()
plt.savefig(ROOT / "results" / "eda_roi_samples.png", dpi=150)
plt.show()
```

**Cell 5 — Visualize digit bboxes (utility-meter):**
```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, sample in zip(axes.flat, um_train[:8]):
    img = cv2.imread(str(sample.image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if sample.digit_bboxes:
        img = draw_digit_bboxes(img, sample.digit_bboxes)
    ax.imshow(img)
    ax.set_title(f"val={sample.value}")
    ax.axis("off")
plt.suptitle("utility-meter — digit bounding boxes")
plt.tight_layout()
plt.savefig(ROOT / "results" / "eda_digit_bboxes.png", dpi=150)
plt.show()
```

**Cell 6 — Digit class distribution (utility-meter):**
```python
from collections import Counter
all_digits = []
for s in um_train:
    if s.digit_bboxes:
        all_digits.extend([d[0] for d in s.digit_bboxes])

counts = Counter(all_digits)
plt.bar([str(k) for k in sorted(counts.keys())], [counts[k] for k in sorted(counts.keys())])
plt.title("Digit class distribution (utility-meter train)")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.savefig(ROOT / "results" / "eda_digit_distribution.png", dpi=150)
plt.show()
```

**Cell 7 — ROI size analysis:**
```python
# Polygon area distribution (waterMeterDataset)
from shapely.geometry import Polygon

areas = []
for s in wm_samples:
    if s.roi_polygon and len(s.roi_polygon) >= 3:
        areas.append(Polygon(s.roi_polygon).area)

plt.hist(areas, bins=30)
plt.title("ROI polygon area distribution (normalized)")
plt.xlabel("Area")
plt.savefig(ROOT / "results" / "eda_roi_areas.png", dpi=150)
plt.show()
```

**Cell 8 — Conclusions:**
```markdown
## Выводы EDA
- Размер датасетов: waterMeter=..., utility-meter=...
- Распределение значений: ...
- Распределение классов цифр: ...
- Типичный размер ROI: ...
- Потенциальные проблемы: ...
```

- [ ] **Step 2: Run notebook locally, verify all cells execute**

Run: `uv run jupyter execute Notebooks/00_data_exploration.ipynb`
Expected: all cells execute, PNGs saved to `results/`.

- [ ] **Step 3: Commit**

```bash
git add Notebooks/00_data_exploration.ipynb results/*.png
git commit -m "feat: EDA notebook — dataset analysis, ROI visualization, digit distribution"
```

---

## Task 6: Notebook 01 — Baseline YOLO Single-Stage

**Files:**
- Create: `Notebooks/01_baseline/yolo_single_stage.ipynb`

Этот ноутбук использует Ultralytics YOLOv11 для одностадийной детекции цифр (каждая цифра = отдельный bounding box). Датасет: `utility-meter...v4i.yolov11` (уже в YOLO формате с train/valid/test splits).

- [ ] **Step 1: Create notebook**

**Cell 1 — Setup:**
```python
import sys
from pathlib import Path

ROOT = Path("../..")
if not (ROOT / "models").exists():
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/<USER>/WaterMeterCV.git"], check=True)
    ROOT = Path("WaterMeterCV")

sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import yaml
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 2 — Config:**
```python
DATASET_PATH = ROOT / "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
DATA_YAML = DATASET_PATH / "data.yaml"
WEIGHTS_DIR = ROOT / "models/weights/baseline_yolo"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Verify dataset
with open(DATA_YAML) as f:
    data_config = yaml.safe_load(f)
print(f"Classes ({data_config['nc']}): {data_config['names']}")
```

**Cell 3 — Train:**
```python
model = YOLO("yolo11n.pt")  # nano for fast iteration, swap to yolo11s/m later

results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    project=str(WEIGHTS_DIR),
    name="run1",
    device="cuda" if torch.cuda.is_available() else "cpu",
    patience=10,
    save=True,
)
```

**Cell 4 — Evaluate:**
```python
best_model = YOLO(WEIGHTS_DIR / "run1/weights/best.pt")
val_results = best_model.val(data=str(DATA_YAML), split="test")

print(f"mAP50: {val_results.box.map50:.4f}")
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"Per-class AP50:")
for i, name in enumerate(data_config["names"]):
    print(f"  {name}: {val_results.box.ap50[i]:.4f}")
```

**Cell 5 — Full-string accuracy:**
```python
from models.metrics.evaluation import full_string_accuracy, measure_inference_time
from models.data.unified_loader import load_utility_meter_dataset

test_samples = load_utility_meter_dataset(DATASET_PATH, split="test")

predictions = []
ground_truths = []

for sample in test_samples:
    result = best_model.predict(str(sample.image_path), verbose=False)[0]
    # Sort detections left-to-right by x-coordinate
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        # Filter to digit classes only (0-9)
        digit_mask = boxes.cls <= 9
        digit_boxes = boxes[digit_mask]

        if len(digit_boxes) > 0:
            sorted_indices = digit_boxes.xywh[:, 0].argsort()
            pred_str = "".join(str(int(digit_boxes.cls[i].item())) for i in sorted_indices)
        else:
            pred_str = ""
    else:
        pred_str = ""

    predictions.append(pred_str)
    if sample.value is not None:
        ground_truths.append(str(int(sample.value)) if sample.value == int(sample.value) else str(sample.value))
    else:
        ground_truths.append("")

acc = full_string_accuracy(predictions, ground_truths)
print(f"Full-string accuracy: {acc:.4f}")
```

**Cell 6 — Inference time:**
```python
import cv2

sample_img = cv2.imread(str(test_samples[0].image_path))
avg_time = measure_inference_time(best_model.predict, sample_img, verbose=False)
print(f"Average inference time: {avg_time:.1f} ms")
```

**Cell 7 — Visual results:**
```python
from models.utils.visualization import draw_digit_bboxes
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, sample in zip(axes.flat, test_samples[:8]):
    img = cv2.imread(str(sample.image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = best_model.predict(str(sample.image_path), verbose=False)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        bboxes = [(int(c), *xywh.tolist()) for c, xywh in zip(result.boxes.cls, result.boxes.xywhn)]
        img = draw_digit_bboxes(img, bboxes)

    ax.imshow(img)
    ax.set_title(f"GT={sample.value}")
    ax.axis("off")

plt.suptitle(f"Baseline YOLO — Full-string acc: {acc:.4f}")
plt.tight_layout()
plt.savefig(ROOT / "results" / "baseline_yolo_predictions.png", dpi=150)
plt.show()
```

**Cell 8 — Conclusions:**
```markdown
## Результаты Baseline YOLO Single-Stage
- mAP50: ...
- mAP50-95: ...
- Full-string accuracy: ...
- Inference time: ... ms
- Наблюдения: ...
```

- [ ] **Step 2: Commit (don't run yet — needs GPU)**

```bash
git add Notebooks/01_baseline/
git commit -m "feat: baseline YOLO single-stage digit detection notebook"
```

- [ ] **Step 3: Merge to develop**

```bash
git checkout develop
git merge feature/data-exploration
git branch -d feature/data-exploration
```

---

## Task 7: Merge baseline, tag v0.1, create ROI branch

- [ ] **Step 1: After running baseline on GPU, merge and tag**

```bash
git checkout develop
git checkout -b feature/baseline-yolo develop
# (notebook already committed from Task 6)
# After running and filling in results:
git add Notebooks/01_baseline/ results/
git commit -m "feat: baseline YOLO results — mAP50=X, full-string acc=Y"
git checkout develop
git merge feature/baseline-yolo
git branch -d feature/baseline-yolo
git checkout main
git merge develop
git tag -a v0.1-baseline -m "Baseline YOLO single-stage experiment complete"
git checkout develop
```

---

## Task 8: Notebook 02 — ROI: Faster R-CNN (Detectron2)

**Files:**
- Create: `Notebooks/02_roi_detection/faster_rcnn.ipynb`

- [ ] **Step 1: Create notebook**

**Cell 1 — Setup:**
```python
import sys
from pathlib import Path

ROOT = Path("../..")
sys.path.insert(0, str(ROOT))

# Detectron2 install (Colab)
# !pip install 'git+https://github.com/facebookresearch/detectron2.git'

import torch
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import json
import cv2
import numpy as np
```

**Cell 2 — Register COCO dataset for ROI detection:**
```python
COCO_ROOT = ROOT / "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.coco"

def get_roi_dicts(split: str):
    """Load COCO annotations, filter to 'Reading Digit' class only (ROI)."""
    json_path = COCO_ROOT / split / "_annotations.coco.json"
    with open(json_path) as f:
        coco = json.load(f)

    # Reading Digit category id = 11 in COCO
    roi_cat_id = next(c["id"] for c in coco["categories"] if c["name"] == "Reading Digit")

    img_lookup = {img["id"]: img for img in coco["images"]}
    # Group annotations by image
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
                "category_id": 0,  # single class: ROI
            })
        dataset_dicts.append(record)
    return dataset_dicts

for split in ["train", "valid", "test"]:
    DatasetCatalog.register(f"watermeter_roi_{split}", lambda s=split: get_roi_dicts(s))
    MetadataCatalog.get(f"watermeter_roi_{split}").set(thing_classes=["ROI"])
```

**Cell 3 — Configure & train:**
```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("watermeter_roi_train",)
cfg.DATASETS.TEST = ("watermeter_roi_valid",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only ROI
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = (2000, 2500)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.OUTPUT_DIR = str(ROOT / "models/checkpoints/roi_faster_rcnn")

import os
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

**Cell 4 — Evaluate IoU on test set:**
```python
from models.metrics.evaluation import compute_iou_bbox
from models.data.unified_loader import load_water_meter_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

test_dicts = get_roi_dicts("test")
ious = []
for d in test_dicts:
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    if len(pred_boxes) > 0:
        # Take highest-confidence prediction
        h, w = img.shape[:2]
        px1, py1, px2, py2 = pred_boxes[0]
        pred_cxcywh = ((px1+px2)/(2*w), (py1+py2)/(2*h), (px2-px1)/w, (py2-py1)/h)

        gt = d["annotations"][0]["bbox"]  # [x, y, w, h] absolute
        gt_cxcywh = ((gt[0]+gt[2]/2)/w, (gt[1]+gt[3]/2)/h, gt[2]/w, gt[3]/h)

        ious.append(compute_iou_bbox(pred_cxcywh, gt_cxcywh))

mean_iou = np.mean(ious) if ious else 0.0
print(f"Mean IoU on test: {mean_iou:.4f}")
print(f"Detections: {len(ious)}/{len(test_dicts)}")
```

**Cell 5 — Visualize + Conclusions** *(аналогично Task 6, Cell 7-8 — draw ROI boxes on test images)*

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/roi-faster-rcnn develop
git add Notebooks/02_roi_detection/faster_rcnn.ipynb
git commit -m "feat: ROI detection — Faster R-CNN (Detectron2) notebook"
```

---

## Task 9: Notebook 02 — ROI: YOLO

**Files:**
- Create: `Notebooks/02_roi_detection/yolo_roi.ipynb`

- [ ] **Step 1: Create notebook**

Структура аналогична Task 8, но вместо Detectron2:

**Cell 1 — Setup:** стандартный (Ultralytics YOLO).

**Cell 2 — Prepare single-class dataset:**
```python
"""
Создаём временную копию YOLO-датасета, где оставляем только класс 'Reading Digit' (class 10)
и перемаппиваем его в class 0 (единственный класс).
"""
import shutil

SRC = ROOT / "WaterMetricsDATA/utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
DST = ROOT / "WaterMetricsDATA/_roi_only_yolo"

for split in ["train", "valid", "test"]:
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    src_labels = SRC / split / "labels"
    src_images = SRC / split / "images"

    for label_file in src_labels.glob("*.txt"):
        roi_lines = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 10:  # Reading Digit
                    roi_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        if roi_lines:
            img_name = label_file.stem + ".jpg"
            src_img = src_images / img_name
            if src_img.exists():
                shutil.copy(src_img, DST / split / "images" / img_name)
                with open(DST / split / "labels" / label_file.name, "w") as f:
                    f.write("\n".join(roi_lines))

# Write data.yaml
import yaml
data_yaml = {
    "train": "../train/images",
    "val": "../valid/images",
    "test": "../test/images",
    "nc": 1,
    "names": ["ROI"],
}
with open(DST / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)
```

**Cell 3 — Train:**
```python
model = YOLO("yolo11n.pt")
results = model.train(
    data=str(DST / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    project=str(ROOT / "models/weights/roi_yolo"),
    name="run1",
    device="cuda" if torch.cuda.is_available() else "cpu",
    patience=10,
)
```

**Cell 4-5:** Evaluate IoU + visualize (аналогично Task 8 Cell 4-5, но через `best_model.predict()`).

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/roi-yolo develop
git add Notebooks/02_roi_detection/yolo_roi.ipynb
git commit -m "feat: ROI detection — YOLO for reading window notebook"
```

---

## Task 10: Notebook 02 — ROI: Segmentation (U-Net)

**Files:**
- Create: `Notebooks/02_roi_detection/segmentation_unet.ipynb`

- [ ] **Step 1: Create notebook**

**Cell 1 — Setup:** стандартный + `import segmentation_models_pytorch as smp` (add `segmentation-models-pytorch` to deps).

**Cell 2 — Prepare segmentation dataset:**
```python
"""
waterMeterDataset уже имеет masks/. Для utility-meter нужно конвертировать
bboxes 'Reading Digit' в бинарные маски.
"""
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ROISegmentationDataset(Dataset):
    def __init__(self, samples, img_size=256, transform=None):
        self.samples = [s for s in samples if s.roi_polygon or s.roi_bbox]
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(s.image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Create binary mask from polygon or bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        if s.roi_polygon:
            pts = np.array([(int(x*w), int(y*h)) for x, y in s.roi_polygon], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        elif s.roi_bbox:
            cx, cy, bw, bh = s.roi_bbox
            x1, y1 = int((cx - bw/2)*w), int((cy - bh/2)*h)
            x2, y2 = int((cx + bw/2)*w), int((cy + bh/2)*h)
            mask[y1:y2, x1:x2] = 1

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        return img, mask
```

**Cell 3 — Model + train:**
```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

# Standard training loop: BCE + Dice loss, Adam, 30 epochs
# (full loop code in notebook)
```

**Cell 4-5:** Evaluate IoU (маска → bbox → IoU) + visualize mask overlays.

- [ ] **Step 2: Add `segmentation-models-pytorch` to `pyproject.toml`**

- [ ] **Step 3: Commit**

```bash
git checkout -b feature/roi-segmentation develop
git add Notebooks/02_roi_detection/segmentation_unet.ipynb pyproject.toml
git commit -m "feat: ROI detection — U-Net segmentation notebook"
```

---

## Task 11: ROI Comparison & Decision Point

- [ ] **Step 1: After running Tasks 8-10 on GPU, compare results**

Create `results/roi_comparison.md`:

```markdown
# ROI Detection Comparison

| Method | Mean IoU | Detection Rate | Inference (ms) |
|--------|----------|----------------|----------------|
| Faster R-CNN | ... | ... | ... |
| YOLO | ... | ... | ... |
| U-Net | ... | ... | ... |

**Winner:** ...
**Reason:** ...
```

- [ ] **Step 2: Merge all ROI branches**

```bash
git checkout develop
git merge feature/roi-faster-rcnn
git merge feature/roi-yolo
git merge feature/roi-segmentation
# resolve conflicts if any
git checkout main
git merge develop
git tag -a v0.2-roi -m "ROI detection experiments complete"
git checkout develop
```

---

## Task 12: Notebook 03 — OCR: CRNN + CTC

**Files:**
- Create: `Notebooks/03_ocr/crnn_ctc.ipynb`

**Зависимость:** лучший ROI-детектор из Task 11.

- [ ] **Step 1: Create notebook**

**Cell 1 — Setup:** стандартный + torch CTC loss.

**Cell 2 — Crop ROI from images using best detector:**
```python
"""
Загружаем лучший ROI-детектор, кропаем reading window из каждого изображения.
Это входные данные для всех OCR-экспериментов.
"""
# Load best ROI model (path determined by Task 11 winner)
# roi_model = YOLO(...) or DefaultPredictor(cfg)

def crop_roi(image, roi_model):
    """Detect ROI and return cropped region."""
    # ... detector-specific inference
    # Returns: cropped_image (numpy array)
    pass

# Pre-crop all train/valid/test and save to temp directory
CROPS_DIR = ROOT / "WaterMetricsDATA/_roi_crops"
# ... (batch cropping logic)
```

**Cell 3 — CRNN model definition:**
```python
import torch
import torch.nn as nn


class CRNN(nn.Module):
    CHARS = "0123456789."

    def __init__(self, img_height=32, num_classes=12):  # 11 chars + blank
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2), nn.ReLU(),
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)                   # (B, 512, 1, W')
        conv = conv.squeeze(2).permute(0, 2, 1)  # (B, W', 512)
        rnn_out, _ = self.rnn(conv)          # (B, W', 512)
        output = self.fc(rnn_out)            # (B, W', num_classes)
        return output.permute(1, 0, 2)       # (T, B, C) for CTC
```

**Cell 4 — Train with CTC loss:**
```python
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop: resize crops to 32xW, grayscale, CTC loss
# (full loop code in notebook)
```

**Cell 5 — Evaluate:**
```python
from models.metrics.evaluation import full_string_accuracy, character_error_rate, measure_inference_time

# CTC greedy decode → string → compare with GT
# Compute full-string acc, CER, inference time
```

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/ocr-crnn-ctc develop
git add Notebooks/03_ocr/crnn_ctc.ipynb
git commit -m "feat: OCR — CRNN + CTC notebook"
```

---

## Task 13: Notebook 03 — OCR: Transformer (TrOCR/PARSeq)

**Files:**
- Create: `Notebooks/03_ocr/transformer_ocr.ipynb`

- [ ] **Step 1: Create notebook**

**Cell 1 — Setup:** стандартный + `from transformers import TrOCRProcessor, VisionEncoderDecoderModel`

**Cell 2 — Load pretrained TrOCR:**
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
```

**Cell 3 — Zero-shot evaluation on ROI crops** (без дообучения):
```python
# Inference on cropped ROIs → predictions
# Compare with GT using full-string acc, CER
```

**Cell 4 — Fine-tune on our data** (если zero-shot недостаточно):
```python
# Standard HuggingFace Trainer fine-tuning
# or PARSeq from https://github.com/baudm/parseq
```

**Cell 5-6:** Evaluate + visualize.

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/ocr-transformer develop
git add Notebooks/03_ocr/transformer_ocr.ipynb
git commit -m "feat: OCR — Transformer (TrOCR) notebook"
```

---

## Task 14: Notebook 03 — OCR: CNN + CTC (без RNN)

**Files:**
- Create: `Notebooks/03_ocr/cnn_ctc.ipynb`

- [ ] **Step 1: Create notebook**

Аналогично Task 12 (CRNN), но модель — чистый CNN без LSTM:

```python
class CNNCTC(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.cnn = nn.Sequential(
            # Deeper CNN with more conv layers to compensate for lack of RNN
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.ReLU(),
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(0, 2, 1)
        return self.fc(conv).permute(1, 0, 2)
```

Cells 2-5: train, evaluate, compare — аналогично Task 12.

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/ocr-cnn-ctc develop
git add Notebooks/03_ocr/cnn_ctc.ipynb
git commit -m "feat: OCR — CNN + CTC (no RNN) notebook"
```

---

## Task 15: Notebook 03 — OCR: Per-Digit Classifier

**Files:**
- Create: `Notebooks/03_ocr/per_digit_classifier.ipynb`

- [ ] **Step 1: Create notebook**

**Cell 2 — Slice ROI into fixed slots:**
```python
def slice_into_digits(roi_crop, num_slots=8):
    """Split ROI crop into N vertical slots for per-digit classification.

    Assumes reading format: up to 5 integer digits + separator + up to 3 decimal digits.
    """
    h, w = roi_crop.shape[:2]
    slot_width = w // num_slots
    slots = []
    for i in range(num_slots):
        x1 = i * slot_width
        x2 = (i + 1) * slot_width if i < num_slots - 1 else w
        slots.append(roi_crop[:, x1:x2])
    return slots
```

**Cell 3 — Digit classifier (simple CNN):**
```python
class DigitClassifier(nn.Module):
    """Classifies a single digit slot: 0-9, dot, empty."""
    def __init__(self, num_classes=12):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

Cells 4-6: train on sliced digits, evaluate full-string accuracy, compare.

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/ocr-per-digit develop
git add Notebooks/03_ocr/per_digit_classifier.ipynb
git commit -m "feat: OCR — per-digit classifier notebook"
```

---

## Task 16: Notebook 04 — CRAFT + CRNN/Transformer

**Files:**
- Create: `Notebooks/04_combinations/craft_crnn.ipynb`

- [ ] **Step 1: Create notebook**

**Cell 2 — CRAFT text detection:**
```python
from craft_text_detector import Craft

craft = Craft(output_dir=str(ROOT / "models/checkpoints/craft_output"), crop_type="poly", cuda=True)

# Detect text regions in meter image
result = craft.detect_text(str(sample.image_path))
# result["boxes"] — detected text region polygons
# result["crops"] — cropped text regions
```

**Cell 3 — Feed crops to CRNN or TrOCR:**
```python
# Option A: CRNN from Task 12
# Option B: TrOCR from Task 13
# Use whichever performed better in OCR experiments

# Combine detected regions → full reading
```

Cells 4-5: evaluate, compare.

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/combo-craft-crnn develop
git add Notebooks/04_combinations/craft_crnn.ipynb
git commit -m "feat: combination — CRAFT + CRNN/Transformer notebook"
```

---

## Task 17: Notebook 04 — Mask R-CNN + Sequence Decoder

**Files:**
- Create: `Notebooks/04_combinations/maskrcnn_decoder.ipynb`

- [ ] **Step 1: Create notebook**

**Cell 2 — Mask R-CNN for digit instance segmentation:**
```python
from detectron2 import model_zoo
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # digits 0-9
# ... train on utility-meter dataset digit annotations
```

**Cell 3 — Sequence decoder:** sort detected digit masks left-to-right → compose reading.

Cells 4-5: evaluate, compare.

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/combo-maskrcnn-decoder develop
git add Notebooks/04_combinations/maskrcnn_decoder.ipynb
git commit -m "feat: combination — Mask R-CNN + sequence decoder notebook"
```

---

## Task 18: Notebook 04 — Detectron2 + OCR

**Files:**
- Create: `Notebooks/04_combinations/detectron2_ocr.ipynb`

- [ ] **Step 1: Create notebook**

**Approach:** Faster R-CNN (from Task 8) для ROI → TrOCR/CRNN (лучший из Task 12-13) для OCR. End-to-end pipeline.

```python
# Cell 2: Load ROI detector (Faster R-CNN from Task 8)
# Cell 3: Load OCR model (best from Tasks 12-13)
# Cell 4: Pipeline — detect ROI → crop → OCR → reading
# Cell 5: Evaluate full-string accuracy, CER, end-to-end inference time
```

- [ ] **Step 2: Commit**

```bash
git checkout -b feature/combo-detectron2-ocr develop
git add Notebooks/04_combinations/detectron2_ocr.ipynb
git commit -m "feat: combination — Detectron2 + OCR end-to-end notebook"
```

---

## Task 19: Final Comparison

- [ ] **Step 1: Create `results/comparison.md`**

```markdown
# Experiment Comparison

## Single-Stage
| Experiment | Full-String Acc | Per-Digit Acc | CER | Inference (ms) |
|-----------|----------------|---------------|-----|----------------|
| Baseline YOLO | | | | |

## ROI Detection
| Experiment | Mean IoU | Detection Rate | Inference (ms) |
|-----------|----------|----------------|----------------|
| Faster R-CNN | | | |
| YOLO | | | |
| U-Net | | | |

## OCR (using best ROI: ...)
| Experiment | Full-String Acc | Per-Digit Acc | CER | Inference (ms) |
|-----------|----------------|---------------|-----|----------------|
| CRNN + CTC | | | | |
| TrOCR | | | | |
| CNN + CTC | | | | |
| Per-digit | | | | |

## Combinations (end-to-end)
| Experiment | Full-String Acc | CER | Total Inference (ms) |
|-----------|----------------|-----|---------------------|
| CRAFT + CRNN | | | |
| Mask R-CNN + decoder | | | |
| Detectron2 + OCR | | | |

## Recommendation
**Best pipeline:** ...
**Reason:** ...
**Next steps:** FastAPI integration, Docker, inference optimization
```

- [ ] **Step 2: Merge all branches, tag final**

```bash
git checkout develop
git merge feature/ocr-crnn-ctc
git merge feature/ocr-transformer
git merge feature/ocr-cnn-ctc
git merge feature/ocr-per-digit
git merge feature/combo-craft-crnn
git merge feature/combo-maskrcnn-decoder
git merge feature/combo-detectron2-ocr
git add results/comparison.md
git commit -m "docs: final experiment comparison table"
git checkout main
git merge develop
git tag -a v0.3-research-complete -m "All ML experiments complete"
```

---

## Task 20: CLAUDE.md

- [ ] **Step 1: Create `CLAUDE.md`**

Write after all experiments are defined — contents should reflect actual project state: commands, architecture, dataset details, conventions established in this plan.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md with project conventions and architecture"
```
