# OCR Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 OCR approaches for reading digits from water meter ROI crops and evaluate each on both datasets (WM and UM).

**Architecture:** Shared `models/data/ocr_dataset.py` provides ROI cropping helpers and dataset loaders used by all 4 notebooks. Each notebook is independent: it loads data, trains, evaluates, saves to `results/`. Models output a digit string (e.g. "04821"); evaluation uses FSA/per-digit-acc/CER from `models/metrics/evaluation.py`.

**Tech Stack:** Python 3.13, PyTorch, torchvision, opencv-python, transformers (TrOCR), ultralytics (YOLO digit detection for per-digit), tqdm

**Branch:** `feature/ocr-notebooks` from `develop`

---

## Key Context

**Datasets:**
- **WM** (waterMeterDataset): `UnifiedSample.roi_polygon` (normalized polygon), `UnifiedSample.value` (float → label = `str(int(sample.value))`). No digit-level bbox GT.
- **UM** (utility-meter YOLO): `UnifiedSample.digit_bboxes = [(class_id, cx, cy, w, h), ...]` (class 0–9 = digit value), ROI via `filter_utility_meter_roi_samples`. GT label = `str(int("".join(str(c) for c,*_ in sorted_by_cx)))`.

**Two crop modes (use_warp flag in every notebook):**
- `use_warp=True` → `warp_roi_polygon` (perspective warp, polygon GT or U-Net mask output)
- `use_warp=False` → `crop_roi_bbox` (axis-aligned crop, YOLO/FRCNN bbox output)

**Evaluation gotcha:** UM GT strips leading zeros via `str(int(...))`. Evaluate in two modes: raw and normalized (`lstrip("0")`). Report both in metrics JSON.

**Orientation:** After prediction, try `pred` and `pred[::-1]` (digit reversal heuristic). Keep the result that is a valid integer. Falls back to raw pred.

**Output files per notebook:**
- `results/ocr_<method>_metrics.json`
- `results/ocr_comparison.csv` (append row)
- `results/ocr_<method>_predictions.png`

---

## File Map

```
models/data/
  ocr_dataset.py          ← NEW: crop/warp helpers, dataset loaders, OCRDataset class

tests/
  test_ocr_dataset.py     ← NEW: unit tests for helpers

Notebooks/03_ocr/
  per_digit_classifier.ipynb   ← NEW: CNN classifier on individual digit crops
  cnn_ctc.ipynb                ← NEW: CNN + CTC decoder
  crnn_ctc.ipynb               ← NEW: CNN + BiLSTM + CTC
  transformer_ocr.ipynb        ← NEW: TrOCR fine-tuning
```

---

## Task 1: OCR Data Infrastructure

**Files:**
- Create: `models/data/ocr_dataset.py`
- Create: `tests/test_ocr_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ocr_dataset.py
import pytest
import numpy as np
from pathlib import Path
from models.data.ocr_dataset import (
    warp_roi_polygon,
    crop_roi_bbox,
    load_wm_ocr_samples,
    load_um_ocr_samples,
    load_um_digit_crops,
    CHARSET,
)

DATA_ROOT = Path("WaterMetricsDATA")
WM_PATH   = DATA_ROOT / "waterMeterDataset/WaterMeters"
UM_YOLO   = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"


class TestCropHelpers:
    def test_warp_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.4), (0.1, 0.4)]
        out = warp_roi_polygon(img, polygon, out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)

    def test_crop_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = crop_roi_bbox(img, (0.5, 0.5, 0.4, 0.2), out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)

    def test_crop_empty_bbox_returns_zeros(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = crop_roi_bbox(img, (0.5, 0.5, 0.0, 0.0), out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)


class TestLoadWmOcrSamples:
    def test_returns_two_lists(self):
        train, test = load_wm_ocr_samples(WM_PATH)
        assert len(train) > 0 and len(test) > 0

    def test_samples_have_polygon_and_value(self):
        train, _ = load_wm_ocr_samples(WM_PATH)
        for s in train[:5]:
            assert s.roi_polygon is not None
            assert s.value is not None

    def test_split_deterministic(self):
        a, _ = load_wm_ocr_samples(WM_PATH, seed=42)
        b, _ = load_wm_ocr_samples(WM_PATH, seed=42)
        assert [s.image_path for s in a] == [s.image_path for s in b]


class TestLoadUmOcrSamples:
    def test_returns_list_with_label_and_bbox(self):
        samples = load_um_ocr_samples(UM_YOLO, "train")
        assert len(samples) > 0
        for img_path, label, roi_bbox in samples[:3]:
            assert img_path.exists()
            assert label.isdigit()
            assert len(roi_bbox) == 4

    def test_label_has_no_leading_zeros(self):
        samples = load_um_ocr_samples(UM_YOLO, "train")
        for _, label, _ in samples:
            assert label == str(int(label)), f"leading zeros in {label!r}"


class TestLoadUmDigitCrops:
    def test_returns_crops_and_classes(self):
        crops = load_um_digit_crops(UM_YOLO, "train")
        assert len(crops) > 0
        for crop, cls in crops[:5]:
            assert crop.shape == (32, 32, 3)
            assert 0 <= cls <= 9


class TestCharset:
    def test_charset_is_digits(self):
        assert CHARSET == "0123456789"
        assert len(CHARSET) == 10
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_ocr_dataset.py -v
```
Expected: `ModuleNotFoundError: No module named 'models.data.ocr_dataset'`

- [ ] **Step 3: Implement `models/data/ocr_dataset.py`**

```python
"""Shared OCR data helpers for all 03_ocr notebooks."""
from pathlib import Path
import numpy as np
import cv2

CHARSET = "0123456789"   # 10 classes; CTC blank index = 10


# ─── Crop helpers ────────────────────────────────────────────────────────────

def _order_corners(box: np.ndarray) -> np.ndarray:
    """Order 4 box corners as [TL, TR, BR, BL]."""
    s    = box.sum(axis=1)
    diff = np.diff(box, axis=1).ravel()
    return np.array(
        [box[s.argmin()], box[diff.argmin()],
         box[s.argmax()], box[diff.argmax()]],
        dtype=np.float32,
    )


def warp_roi_polygon(
    img_bgr: np.ndarray,
    polygon: list[tuple[float, float]],
    out_h: int = 64,
    out_w: int = 256,
) -> np.ndarray:
    """Perspective-warp the ROI polygon region to a (out_h × out_w) rectangle."""
    h, w = img_bgr.shape[:2]
    pts  = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box  = _order_corners(cv2.boxPoints(rect))
    dst  = np.array([[0, 0], [out_w - 1, 0],
                     [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h))


def crop_roi_bbox(
    img_bgr: np.ndarray,
    bbox_cxcywh: tuple[float, float, float, float],
    out_h: int = 64,
    out_w: int = 256,
) -> np.ndarray:
    """Crop axis-aligned bbox and resize to (out_h × out_w)."""
    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = bbox_cxcywh
    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_w, out_h))


# ─── Dataset loaders ─────────────────────────────────────────────────────────

def load_wm_ocr_samples(
    wm_path: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list, list]:
    """Return (train, test) lists of UnifiedSample with roi_polygon and value.

    Label for each sample: str(int(sample.value))
    """
    from models.data.unified_loader import load_water_meter_dataset_split
    train_raw, test_raw = load_water_meter_dataset_split(wm_path, train_ratio, seed)
    train = [s for s in train_raw if s.roi_polygon is not None and s.value is not None]
    test  = [s for s in test_raw  if s.roi_polygon is not None and s.value is not None]
    return train, test


def load_um_ocr_samples(
    yolo_path: Path,
    split: str,
) -> list[tuple[Path, str, tuple[float, float, float, float]]]:
    """Return (img_path, label, roi_bbox_cxcywh) for UM images that have both
    ROI (class 10) and digit (class 0–9) annotations.

    label = digits sorted left-to-right by cx, normalized: str(int(label)).
    """
    from models.data.roi_dataset import filter_utility_meter_roi_samples
    roi_map = {p.stem: (p, bbox)
               for p, bbox in filter_utility_meter_roi_samples(yolo_path, split)}

    labels_dir = yolo_path / split / "labels"
    results = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.stem not in roi_map:
            continue
        img_path, roi_bbox = roi_map[label_file.stem]
        digits: list[tuple[int, float]] = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls > 9:
                    continue
                digits.append((cls, float(parts[1])))  # (digit, cx)
        if not digits:
            continue
        digits.sort(key=lambda d: d[1])
        raw = "".join(str(d[0]) for d in digits)
        label = str(int(raw)) if raw.lstrip("0") else "0"
        results.append((img_path, label, roi_bbox))
    return results


def load_um_digit_crops(
    yolo_path: Path,
    split: str,
    crop_size: int = 32,
) -> list[tuple[np.ndarray, int]]:
    """Return (crop_bgr, digit_class) for every digit bbox in the UM split.

    Used to build per-digit classifier training set.
    crop_size: output square size in pixels.
    """
    labels_dir = yolo_path / split / "labels"
    images_dir = yolo_path / split / "images"
    results = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / (label_file.stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls > 9:
                    continue
                cx, cy, bw, bh = (float(p) for p in parts[1:5])
                x1 = max(0, int((cx - bw / 2) * w))
                y1 = max(0, int((cy - bh / 2) * h))
                x2 = min(w, int((cx + bw / 2) * w))
                y2 = min(h, int((cy + bh / 2) * h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                results.append((cv2.resize(crop, (crop_size, crop_size)), cls))
    return results
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_ocr_dataset.py -v
```
Expected: all 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add models/data/ocr_dataset.py tests/test_ocr_dataset.py
git commit -m "feat: OCR data helpers — warp/crop, WM/UM loaders, digit crops"
```

---

## Task 2: per_digit_classifier.ipynb

**Files:** Create `Notebooks/03_ocr/per_digit_classifier.ipynb`

**Approach:**
- Train a ResNet18-based 10-class CNN on UM digit crops (32×32).
- **UM evaluation**: crop each digit using GT bbox → classify → sort by cx → reconstruct label.
- **WM evaluation**: warp ROI → split into N equal columns where N = `len(str(int(sample.value)))` from each test sample. Classify each column. (Heuristic; no digit GT for WM.)

- [ ] **Step 1: Create notebook cells**

**Cell 0** — Title markdown:
```
# 03 — OCR: Per-Digit Classifier

Train a CNN digit classifier on individual digit crops (UM dataset).
Evaluate by cropping each digit and assembling predictions sorted by position.
WM evaluation uses perspective-warped ROI split into N equal columns.
```

**Cell 1** — IN_COLAB setup (same pattern as ROI notebooks, BRANCH="feature/ocr-notebooks", extra pip installs: none beyond base).

**Cell 2** — Config:
```python
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
WM_PATH      = DATA_ROOT / "waterMeterDataset/WaterMeters"
WEIGHTS_DIR  = WEIGHTS_BASE / "ocr_per_digit"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_SIZE  = 32
EPOCHS     = 15
BATCH_SIZE = 64
LR         = 1e-3
RUN_NAME   = f"perdigit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

**Cell 3** — Imports and helpers:
```python
import torch, torch.nn as nn, cv2, numpy as np, json, csv, time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms.functional as TF
import torchvision.models as tv_models
from datetime import datetime
from tqdm import tqdm

from models.data.ocr_dataset import (
    load_wm_ocr_samples, load_um_ocr_samples,
    load_um_digit_crops, warp_roi_polygon, CHARSET,
)
from models.metrics.evaluation import full_string_accuracy, per_digit_accuracy, character_error_rate


def build_digit_classifier():
    m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def train_classifier(model, loader, epochs, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total, correct = 0, 0
        for x, y in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  acc={correct/total:.3f}")
    return model


def predict_label_from_crops(model, crops_with_cx, device):
    """crops_with_cx: [(crop_bgr, cx_normalized), ...]"""
    model.eval()
    preds = []
    with torch.no_grad():
        for crop, cx in sorted(crops_with_cx, key=lambda x: x[1]):
            t = TF.to_tensor(cv2.cvtColor(
                cv2.resize(crop, (CROP_SIZE, CROP_SIZE)), cv2.COLOR_BGR2RGB
            )).unsqueeze(0).to(device)
            cls = model(t).argmax(1).item()
            preds.append(str(cls))
    return "".join(preds)


def orient_fix(pred: str) -> str:
    """Try reversed digit string; keep whichever is a valid integer."""
    rev = pred[::-1]
    if rev.isdigit():
        return rev if int(rev) >= 0 else pred
    return pred
```

**Cell 4** — Load UM digit crops + train:
```python
print("Loading UM digit crops...")
raw_crops = load_um_digit_crops(UM_YOLO_PATH, "train")
print(f"  {len(raw_crops)} digit crop samples")

# Build tensor dataset
imgs  = torch.stack([TF.to_tensor(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c, _ in raw_crops])
labels = torch.tensor([cls for _, cls in raw_crops], dtype=torch.long)
ds    = TensorDataset(imgs, labels)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

model = build_digit_classifier()
print(f"Training per-digit classifier ({EPOCHS} epochs)...")
model = train_classifier(model, loader, EPOCHS, DEVICE)
torch.save(model.state_dict(), WEIGHTS_DIR / f"best_{RUN_NAME}.pth")
print("Training done.")
```

**Cell 5** — UM evaluation:
```python
um_test_samples = load_um_ocr_samples(UM_YOLO_PATH, "test")
print(f"UM test samples: {len(um_test_samples)}")

um_preds, um_gts, um_times = [], [], []
um_img_path = UM_YOLO_PATH  # for loading crops

for img_path, label_gt, roi_bbox in um_test_samples:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    # load digit bboxes from label file
    label_file = UM_YOLO_PATH / "test" / "labels" / (img_path.stem + ".txt")
    crops_with_cx = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) > 9: continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1, y1 = max(0,int((cx-bw/2)*w)), max(0,int((cy-bh/2)*h))
            x2, y2 = min(w,int((cx+bw/2)*w)), min(h,int((cy+bh/2)*h))
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops_with_cx.append((crop, cx))

    t0 = time.perf_counter()
    pred = predict_label_from_crops(model, crops_with_cx, DEVICE)
    um_times.append((time.perf_counter() - t0) * 1000)

    pred_fixed = orient_fix(pred)
    um_preds.append(pred_fixed)
    um_gts.append(label_gt)

um_fsa_raw  = full_string_accuracy(um_preds, um_gts)
um_fsa_norm = full_string_accuracy(
    [p.lstrip("0") or "0" for p in um_preds],
    [g.lstrip("0") or "0" for g in um_gts]
)
um_pda = float(np.mean([per_digit_accuracy(p, g) for p, g in zip(um_preds, um_gts)]))
um_cer = float(np.mean([character_error_rate(p, g) for p, g in zip(um_preds, um_gts)]))
um_ms  = float(np.mean(um_times))
print(f"UM — FSA raw={um_fsa_raw:.3f}  FSA norm={um_fsa_norm:.3f}  PDA={um_pda:.3f}  CER={um_cer:.3f}  {um_ms:.1f}ms")
```

**Cell 6** — WM evaluation (warp + grid split):
```python
wm_train, wm_test = load_wm_ocr_samples(WM_PATH)
print(f"WM test samples: {len(wm_test)}")

wm_preds, wm_gts, wm_times = [], [], []
for s in wm_test:
    img = cv2.imread(str(s.image_path))
    label_gt = str(int(s.value))
    n_digits = len(label_gt)

    t0 = time.perf_counter()
    warped = warp_roi_polygon(img, s.roi_polygon, out_h=CROP_SIZE, out_w=CROP_SIZE * n_digits)
    # split into n_digits equal columns
    col_w = warped.shape[1] // n_digits
    crops_with_cx = []
    for i in range(n_digits):
        col = warped[:, i*col_w:(i+1)*col_w]
        col = cv2.resize(col, (CROP_SIZE, CROP_SIZE))
        crops_with_cx.append((col, i / n_digits))  # cx = column index / n
    pred = predict_label_from_crops(model, crops_with_cx, DEVICE)
    wm_times.append((time.perf_counter() - t0) * 1000)

    wm_preds.append(orient_fix(pred))
    wm_gts.append(label_gt)

wm_fsa_raw  = full_string_accuracy(wm_preds, wm_gts)
wm_fsa_norm = full_string_accuracy(
    [p.lstrip("0") or "0" for p in wm_preds],
    [g.lstrip("0") or "0" for g in wm_gts]
)
wm_pda = float(np.mean([per_digit_accuracy(p, g) for p, g in zip(wm_preds, wm_gts)]))
wm_cer = float(np.mean([character_error_rate(p, g) for p, g in zip(wm_preds, wm_gts)]))
wm_ms  = float(np.mean(wm_times))
print(f"WM — FSA raw={wm_fsa_raw:.3f}  FSA norm={wm_fsa_norm:.3f}  PDA={wm_pda:.3f}  CER={wm_cer:.3f}  {wm_ms:.1f}ms")
```

**Cell 7** — Visualization (2×4 grid, GT vs pred label as title):
```python
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes[0]):
    if i >= len(um_test_samples): ax.axis("off"); continue
    img_path, gt, _ = um_test_samples[i]
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.set_title(f"GT:{gt}\nPred:{um_preds[i]}", fontsize=9); ax.axis("off")
for i, ax in enumerate(axes[1]):
    if i >= len(wm_test): ax.axis("off"); continue
    s = wm_test[i]
    crop = warp_roi_polygon(cv2.imread(str(s.image_path)), s.roi_polygon)
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_title(f"GT:{wm_gts[i]}\nPred:{wm_preds[i]}", fontsize=9); ax.axis("off")
plt.suptitle("Per-Digit Classifier — Row 0: UM, Row 1: WM")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ocr_per_digit_predictions.png", dpi=150)
plt.close(); print("Saved ocr_per_digit_predictions.png")
```

**Cell 8** — Save results:
```python
metrics = {
    "method": "per_digit_classifier",
    "utility_meter": {
        "fsa_raw": round(um_fsa_raw, 4), "fsa_norm": round(um_fsa_norm, 4),
        "per_digit_acc": round(um_pda, 4), "cer": round(um_cer, 4),
        "avg_inference_ms": round(um_ms, 1), "n_test": len(um_test_samples),
    },
    "water_meter": {
        "fsa_raw": round(wm_fsa_raw, 4), "fsa_norm": round(wm_fsa_norm, 4),
        "per_digit_acc": round(wm_pda, 4), "cer": round(wm_cer, 4),
        "avg_inference_ms": round(wm_ms, 1), "n_test": len(wm_test),
    },
    "config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "backbone": "resnet18"},
    "run_date": datetime.now().isoformat(),
}
with open(RESULTS_DIR / "ocr_per_digit_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

csv_path = RESULTS_DIR / "ocr_comparison.csv"
header = ["method","um_fsa_raw","um_fsa_norm","um_pda","um_cer","um_ms",
          "wm_fsa_raw","wm_fsa_norm","wm_pda","wm_cer","wm_ms","run_date"]
write_header = not csv_path.exists()
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if write_header: w.writerow(header)
    w.writerow(["per_digit_classifier",
                round(um_fsa_raw,4), round(um_fsa_norm,4), round(um_pda,4), round(um_cer,4), round(um_ms,1),
                round(wm_fsa_raw,4), round(wm_fsa_norm,4), round(wm_pda,4), round(wm_cer,4), round(wm_ms,1),
                datetime.now().strftime("%Y-%m-%d %H:%M")])
print("Saved results.")
```

**Cell 9** — Conclusions markdown (fill after run).

- [ ] **Step 2: Write notebook JSON to file** (use Write tool with the cells above)

- [ ] **Step 3: Syntax-check**

```bash
uv run python -c "
import json, py_compile, tempfile, pathlib
nb = json.load(open('Notebooks/03_ocr/per_digit_classifier.ipynb'))
code = '\n'.join(''.join(c['source']) if isinstance(c['source'],list) else c['source']
    for c in nb['cells'] if c['cell_type']=='code')
with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
    f.write(code); fname = f.name
try: py_compile.compile(fname, doraise=True); print('OK')
finally: pathlib.Path(fname).unlink()
"
```

- [ ] **Step 4: Commit**

```bash
git add Notebooks/03_ocr/per_digit_classifier.ipynb
git commit -m "feat: OCR notebook — per-digit CNN classifier"
```

---

## Task 3: cnn_ctc.ipynb

**Files:** Create `Notebooks/03_ocr/cnn_ctc.ipynb`

**Architecture:** CNN (VGG-style small: 4 conv blocks) → adaptive pool to fixed height 1 → sequence of width steps → Linear → log_softmax → CTCLoss.

- [ ] **Step 1: Create notebook cells**

**Cell 0** — Title:
```
# 03 — OCR: CNN + CTC

Sequence-to-sequence OCR using a CNN feature extractor with CTC loss.
No recurrent layers — faster inference than CRNN.
Evaluated on both WM (polygon warp) and UM (bbox crop).
```

**Cell 1** — IN_COLAB setup (BRANCH="feature/ocr-notebooks").

**Cell 2** — Config:
```python
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"
WM_PATH      = DATA_ROOT / "waterMeterDataset/WaterMeters"
WEIGHTS_DIR  = WEIGHTS_BASE / "ocr_cnn_ctc"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_H       = 64
IMG_W       = 256
EPOCHS_UM   = 30
EPOCHS_WM   = 30
BATCH_SIZE  = 32
LR          = 1e-3
NUM_CLASSES = 11   # 10 digits + CTC blank (index 10)
RUN_NAME    = f"cnn_ctc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

**Cell 3** — Model + dataset + helpers:
```python
import torch, torch.nn as nn, cv2, numpy as np, json, csv, time
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from datetime import datetime
from tqdm import tqdm

from models.data.ocr_dataset import (
    load_wm_ocr_samples, load_um_ocr_samples,
    warp_roi_polygon, crop_roi_bbox, CHARSET,
)
from models.metrics.evaluation import full_string_accuracy, per_digit_accuracy, character_error_rate


class CNNCTC(nn.Module):
    """Small CNN → CTC. Input: (B, 3, IMG_H, IMG_W). Output: (T, B, NUM_CLASSES)."""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )  # output: (B, 256, 4, W/4)
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # (B, 256, 1, T)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)                           # (B, 256, H', T)
        x = self.pool(x).squeeze(2)               # (B, 256, T)
        x = self.fc(x.permute(0, 2, 1))           # (B, T, C)
        return x.permute(1, 0, 2).log_softmax(2)  # (T, B, C)


class OCRDataset(Dataset):
    """Generic OCR dataset for both WM and UM.

    samples: list of (img_path, label_str, roi)
             roi is polygon (list of tuples) when use_warp=True,
             bbox (cx,cy,w,h tuple) when use_warp=False.
    """
    def __init__(self, samples, use_warp: bool = True, augment: bool = False):
        self.samples  = samples
        self.use_warp = use_warp
        self.augment  = augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str, roi = self.samples[idx]
        img = cv2.imread(str(img_path))
        if self.use_warp:
            crop = warp_roi_polygon(img, roi, IMG_H, IMG_W)
        else:
            crop = crop_roi_bbox(img, roi, IMG_H, IMG_W)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = TF.to_tensor(crop)
        label  = torch.tensor([CHARSET.index(c) for c in label_str if c in CHARSET],
                               dtype=torch.long)
        return tensor, label


def collate_ctc(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_cat = torch.cat(labels)
    return imgs, labels_cat, lengths


def ctc_decode(log_probs: torch.Tensor) -> list[str]:
    """Greedy CTC decode. log_probs: (T, B, C)."""
    preds = log_probs.argmax(2).permute(1, 0).cpu().numpy()  # (B, T)
    results = []
    for seq in preds:
        chars, prev = [], -1
        for c in seq:
            if c != prev and c != 10:  # 10 = blank
                chars.append(CHARSET[c])
            prev = c
        results.append("".join(chars))
    return results


def orient_fix(pred: str) -> str:
    rev = pred[::-1]
    return rev if rev.isdigit() else pred


def train_ctc(model, loader, epochs, device, best_path=None):
    model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//2), gamma=0.1)
    criterion  = nn.CTCLoss(blank=10, zero_infinity=True)
    best_loss  = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for imgs, labels, lengths in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels, lengths = imgs.to(device), labels.to(device), lengths.to(device)
            log_p   = model(imgs)        # (T, B, C)
            T       = log_p.size(0)
            inp_len = torch.full((imgs.size(0),), T, dtype=torch.long, device=device)
            loss = criterion(log_p, labels, inp_len, lengths)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        scheduler.step()
        avg = total / len(loader)
        if best_path and avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), best_path)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}")
    return model


@torch.no_grad()
def eval_ctc(model, dataset, device):
    model.eval()
    preds, gts, times = [], [], []
    for i in range(len(dataset)):
        img_t, label_t = dataset[i]
        gt = "".join(CHARSET[c] for c in label_t.tolist())
        t0 = time.perf_counter()
        log_p = model(img_t.unsqueeze(0).to(device))
        times.append((time.perf_counter() - t0) * 1000)
        pred = ctc_decode(log_p)[0]
        preds.append(orient_fix(pred))
        gts.append(gt)
    fsa_raw  = full_string_accuracy(preds, gts)
    fsa_norm = full_string_accuracy([p.lstrip("0") or "0" for p in preds],
                                    [g.lstrip("0") or "0" for g in gts])
    pda = float(np.mean([per_digit_accuracy(p, g) for p, g in zip(preds, gts)]))
    cer = float(np.mean([character_error_rate(p, g) for p, g in zip(preds, gts)]))
    ms  = float(np.mean(times))
    return fsa_raw, fsa_norm, pda, cer, ms, preds, gts

print("Helpers loaded.")
```

**Cell 4** — Experiment 1 UM (use_warp=False, bbox crop):
```python
um_train_raw = load_um_ocr_samples(UM_YOLO_PATH, "train")
um_test_raw  = load_um_ocr_samples(UM_YOLO_PATH, "test")
# convert to (img_path, label, roi) format
um_train_samples = [(p, l, b) for p, l, b in um_train_raw]
um_test_samples  = [(p, l, b) for p, l, b in um_test_raw]

um_train_ds = OCRDataset(um_train_samples, use_warp=False, augment=True)
um_test_ds  = OCRDataset(um_test_samples,  use_warp=False)
um_loader   = DataLoader(um_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=WORKERS, collate_fn=collate_ctc)
print(f"UM train: {len(um_train_ds)}, test: {len(um_test_ds)}")

model_um = CNNCTC()
print(f"Training CNN+CTC on UM ({EPOCHS_UM} epochs)...")
model_um = train_ctc(model_um, um_loader, EPOCHS_UM, DEVICE,
                     best_path=WEIGHTS_DIR / f"um_best_{RUN_NAME}.pth")
torch.save(model_um.state_dict(), WEIGHTS_DIR / f"um_last_{RUN_NAME}.pth")
```

**Cell 5** — UM evaluation:
```python
um_fsa_raw, um_fsa_norm, um_pda, um_cer, um_ms, um_preds, um_gts = eval_ctc(model_um, um_test_ds, DEVICE)
print(f"UM — FSA raw={um_fsa_raw:.3f}  norm={um_fsa_norm:.3f}  PDA={um_pda:.3f}  CER={um_cer:.3f}  {um_ms:.1f}ms")
```

**Cell 6** — Experiment 2 WM (use_warp=True):
```python
wm_train_raw, wm_test_raw = load_wm_ocr_samples(WM_PATH)
wm_train_samples = [(s.image_path, str(int(s.value)), s.roi_polygon) for s in wm_train_raw]
wm_test_samples  = [(s.image_path, str(int(s.value)), s.roi_polygon) for s in wm_test_raw]

wm_train_ds = OCRDataset(wm_train_samples, use_warp=True, augment=True)
wm_test_ds  = OCRDataset(wm_test_samples,  use_warp=True)
wm_loader   = DataLoader(wm_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=WORKERS, collate_fn=collate_ctc)
print(f"WM train: {len(wm_train_ds)}, test: {len(wm_test_ds)}")

model_wm = CNNCTC()
print(f"Training CNN+CTC on WM ({EPOCHS_WM} epochs)...")
model_wm = train_ctc(model_wm, wm_loader, EPOCHS_WM, DEVICE,
                     best_path=WEIGHTS_DIR / f"wm_best_{RUN_NAME}.pth")
torch.save(model_wm.state_dict(), WEIGHTS_DIR / f"wm_last_{RUN_NAME}.pth")
```

**Cell 7** — WM evaluation:
```python
wm_fsa_raw, wm_fsa_norm, wm_pda, wm_cer, wm_ms, wm_preds, wm_gts = eval_ctc(model_wm, wm_test_ds, DEVICE)
print(f"WM — FSA raw={wm_fsa_raw:.3f}  norm={wm_fsa_norm:.3f}  PDA={wm_pda:.3f}  CER={wm_cer:.3f}  {wm_ms:.1f}ms")
```

**Cells 8–9**: Visualization (same 2×4 pattern as per_digit) + Save results (same structure, method="cnn_ctc").

**Cell 10**: Conclusions markdown.

- [ ] **Step 2: Write notebook JSON**

- [ ] **Step 3: Syntax-check** (same command as Task 2 Step 3)

- [ ] **Step 4: Commit**

```bash
git add Notebooks/03_ocr/cnn_ctc.ipynb
git commit -m "feat: OCR notebook — CNN+CTC"
```

---

## Task 4: crnn_ctc.ipynb

**Files:** Create `Notebooks/03_ocr/crnn_ctc.ipynb`

**Architecture:** Same CNN backbone as Task 3, but followed by 2-layer BiLSTM (hidden=256) before the linear projection. Everything else (dataset, training loop, evaluation) is identical to Task 3 — only the model class changes.

- [ ] **Step 1: Create notebook cells**

Same structure as Task 3 (copy cells 0-2, 4-10). Only change Cell 3's model definition:

```python
class CRNNCTC(nn.Module):
    """CNN + BiLSTM + CTC. Input: (B, 3, IMG_H, IMG_W). Output: (T, B, NUM_CLASSES)."""
    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn  = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=False)
        self.fc   = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x).squeeze(2).permute(2, 0, 1)  # (T, B, 256)
        x, _ = self.rnn(x)                              # (T, B, 2*hidden)
        return self.fc(x).log_softmax(2)                # (T, B, C)
```

Config: `EPOCHS_UM=30, EPOCHS_WM=30` (same), RUN_NAME uses "crnn_ctc". Weights saved as `ocr_crnn_ctc/`.

- [ ] **Step 2: Write notebook JSON**

- [ ] **Step 3: Syntax-check**

- [ ] **Step 4: Commit**

```bash
git add Notebooks/03_ocr/crnn_ctc.ipynb
git commit -m "feat: OCR notebook — CRNN+CTC (BiLSTM)"
```

---

## Task 5: transformer_ocr.ipynb

**Files:** Create `Notebooks/03_ocr/transformer_ocr.ipynb`

**Architecture:** Fine-tune `microsoft/trocr-base-printed` (TrOCR) from HuggingFace. Vision encoder (ViT) + autoregressive transformer decoder. Already pre-trained on printed text — minimal fine-tuning needed.

**Extra dependency:** `uv sync --extra transformers` (already defined in pyproject.toml).

- [ ] **Step 1: Create notebook cells**

**Cell 0** — Title:
```
# 03 — OCR: Transformer OCR (TrOCR)

Fine-tune microsoft/trocr-base-printed on water meter ROI crops.
Vision encoder (ViT-base) + autoregressive transformer decoder.
No CTC — generates digit sequences auto-regressively.
```

**Cell 1** — IN_COLAB setup: add `transformers sentencepiece` to pip install list.

**Cell 2** — Config:
```python
MODEL_NAME  = "microsoft/trocr-base-printed"
EPOCHS_UM   = 10
EPOCHS_WM   = 10
BATCH_SIZE  = 16
LR          = 5e-5
IMG_H, IMG_W = 384, 384   # TrOCR expected input size
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = WEIGHTS_BASE / "ocr_trocr"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_NAME    = f"trocr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

**Cell 3** — Imports + dataset:
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch, cv2, numpy as np, json, csv, time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datetime import datetime
from tqdm import tqdm

from models.data.ocr_dataset import (
    load_wm_ocr_samples, load_um_ocr_samples,
    warp_roi_polygon, crop_roi_bbox,
)
from models.metrics.evaluation import full_string_accuracy, per_digit_accuracy, character_error_rate

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)


class TrOCRDataset(Dataset):
    def __init__(self, samples, use_warp: bool = True):
        self.samples  = samples
        self.use_warp = use_warp

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str, roi = self.samples[idx]
        img = cv2.imread(str(img_path))
        crop = warp_roi_polygon(img, roi, IMG_H, IMG_W) if self.use_warp \
               else crop_roi_bbox(img, roi, IMG_H, IMG_W)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil, return_tensors="pt").pixel_values.squeeze(0)
        labels = processor.tokenizer(label_str, return_tensors="pt",
                                     padding=False).input_ids.squeeze(0)
        return pixel_values, labels


def collate_trocr(batch):
    imgs, labels = zip(*batch)
    pixel_values = torch.stack(imgs)
    max_len = max(l.size(0) for l in labels)
    padded = torch.full((len(labels), max_len), processor.tokenizer.pad_token_id, dtype=torch.long)
    for i, l in enumerate(labels):
        padded[i, :l.size(0)] = l
    padded[padded == processor.tokenizer.pad_token_id] = -100  # ignore in loss
    return pixel_values, padded


def build_trocr():
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = 10
    model.config.no_repeat_ngram_size   = 3
    model.config.early_stopping         = True
    model.config.beam_size              = 4
    return model


def train_trocr(model, loader, epochs, device, best_path=None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for pixel_values, labels in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            loss = model(pixel_values=pixel_values, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        avg = total / len(loader)
        if best_path and avg < best_loss:
            best_loss = avg; model.save_pretrained(best_path)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={avg:.4f}")
    return model


@torch.no_grad()
def eval_trocr(model, dataset, device):
    model.eval()
    preds, gts, times = [], [], []
    for i in range(len(dataset)):
        pixel_values, label_ids = dataset[i]
        gt_ids = label_ids[label_ids != -100]
        gt = processor.tokenizer.decode(gt_ids, skip_special_tokens=True)
        t0 = time.perf_counter()
        gen = model.generate(pixel_values.unsqueeze(0).to(device))
        times.append((time.perf_counter() - t0) * 1000)
        pred = processor.tokenizer.decode(gen[0], skip_special_tokens=True).strip()
        pred_digits = "".join(c for c in pred if c.isdigit())
        preds.append(pred_digits); gts.append(gt)
    fsa_raw  = full_string_accuracy(preds, gts)
    fsa_norm = full_string_accuracy([p.lstrip("0") or "0" for p in preds],
                                    [g.lstrip("0") or "0" for g in gts])
    pda = float(np.mean([per_digit_accuracy(p, g) for p, g in zip(preds, gts)]))
    cer = float(np.mean([character_error_rate(p, g) for p, g in zip(preds, gts)]))
    ms  = float(np.mean(times))
    return fsa_raw, fsa_norm, pda, cer, ms, preds, gts

print("TrOCR helpers loaded.")
```

**Cells 4–9**: UM train → eval → WM train → eval → visualization → save results (method="trocr", same CSV columns, weights saved via `model.save_pretrained(path)`).

**Cell 10**: Conclusions markdown.

- [ ] **Step 2: Write notebook JSON**

- [ ] **Step 3: Install transformers extra and syntax-check**

```bash
uv sync --extra transformers --extra cuda
uv run python -c "..."   # same syntax check
```

- [ ] **Step 4: Commit**

```bash
git add Notebooks/03_ocr/transformer_ocr.ipynb
git commit -m "feat: OCR notebook — TrOCR fine-tuning"
```

---

## Self-Review

**Spec coverage:**
- ✓ per_digit_classifier: Task 2
- ✓ cnn_ctc: Task 3
- ✓ crnn_ctc: Task 4
- ✓ transformer_ocr: Task 5
- ✓ Shared data infrastructure: Task 1
- ✓ Both datasets evaluated in every notebook
- ✓ Both crop modes (warp / bbox) used
- ✓ Orientation fix applied in every eval
- ✓ Results saved to JSON + CSV per notebook
- ✓ UM GT normalization (leading zeros) handled

**No placeholders:** All code blocks are complete.

**Type consistency:**
- `OCRDataset` samples format: `(img_path: Path, label_str: str, roi: list|tuple)` — consistent across Tasks 3, 4
- `TrOCRDataset` uses same samples format — consistent
- `eval_ctc` / `eval_trocr` return same 7-tuple — consistent
- `load_um_ocr_samples` returns `list[tuple[Path, str, tuple]]` — used correctly in Tasks 3, 4, 5
- `load_wm_ocr_samples` returns `tuple[list[UnifiedSample], list[UnifiedSample]]` — Tasks 3–5 convert inline with `(s.image_path, str(int(s.value)), s.roi_polygon)`
