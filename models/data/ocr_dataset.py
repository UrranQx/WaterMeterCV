"""Shared OCR data helpers for all 03_ocr notebooks.

Three persistent crop datasets are created by prepare_ocr_crops() and loaded
by load_ocr_crops(). All notebooks use these pre-computed crops.

Crop types:
  wm_polygon — WM roi_polygon perspective-warped (best quality)
  wm_bbox    — WM polygon_to_bbox axis-aligned crop (comparison)
  um_bbox    — UM ROI class-10 bbox crop
"""
from pathlib import Path
import csv
import numpy as np
import cv2

CHARSET = "0123456789"   # 10 digits; CTC blank = index 10
OUT_H   = 64
OUT_W   = 256


# ─── Crop helpers ─────────────────────────────────────────────────────────────

def _order_corners(box: np.ndarray) -> np.ndarray:
    """Order 4 corners as TL, TR, BR, BL (sum / diff trick)."""
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
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> np.ndarray:
    """Perspective-warp the roi_polygon region to (out_h × out_w)."""
    h, w = img_bgr.shape[:2]
    pts  = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box  = _order_corners(cv2.boxPoints(rect))
    dst  = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(img_bgr, cv2.getPerspectiveTransform(box, dst), (out_w, out_h))


def crop_roi_bbox(
    img_bgr: np.ndarray,
    bbox_cxcywh: tuple[float, float, float, float],
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> np.ndarray:
    """Axis-aligned bbox crop resized to (out_h × out_w)."""
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


# ─── Dataset preparation ──────────────────────────────────────────────────────

def prepare_ocr_crops(
    wm_path: Path,
    um_yolo_path: Path,
    dst_root: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> None:
    """Create three OCR crop datasets under dst_root (idempotent).

    wm_polygon/  — WM perspective warp on roi_polygon
    wm_bbox/     — WM axis-aligned crop on polygon_to_bbox(roi_polygon)
    um_bbox/     — UM axis-aligned crop on ROI class-10 bbox

    Each split: images/<stem>.png  +  labels.csv (filename, label)
    Skips images already present (safe to re-run).
    """
    from models.data.unified_loader import load_water_meter_dataset_split
    from models.data.roi_dataset import polygon_to_bbox, filter_utility_meter_roi_samples

    wm_path, um_yolo_path, dst_root = Path(wm_path), Path(um_yolo_path), Path(dst_root)

    # ── WM crops ─────────────────────────────────────────────────────────────
    wm_train, wm_test = load_water_meter_dataset_split(wm_path, train_ratio, seed)

    for split_name, samples in [("train", wm_train), ("test", wm_test)]:
        valid = [s for s in samples if s.roi_polygon is not None and s.value is not None]

        for crop_type, crop_fn in [
            ("wm_polygon", lambda img, s: warp_roi_polygon(img, s.roi_polygon, out_h, out_w)),
            ("wm_bbox",    lambda img, s: crop_roi_bbox(img, polygon_to_bbox(s.roi_polygon), out_h, out_w)),
        ]:
            img_dir = dst_root / crop_type / split_name / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dst_root / crop_type / split_name / "labels.csv"

            existing = {r["filename"] for r in _read_csv(csv_path)}
            new_rows: list[dict] = []

            for s in valid:
                fname = s.image_path.stem + ".png"
                if fname in existing:
                    continue
                img = cv2.imread(str(s.image_path))
                if img is None:
                    continue
                crop = crop_fn(img, s)
                cv2.imwrite(str(img_dir / fname), crop)
                new_rows.append({"filename": fname, "label": str(int(s.value))})

            _append_csv(csv_path, new_rows)

    # ── UM crops ─────────────────────────────────────────────────────────────
    for split_name in ("train", "valid", "test"):
        out_split = "test" if split_name in ("valid", "test") else "train"
        img_dir  = dst_root / "um_bbox" / out_split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dst_root / "um_bbox" / out_split / "labels.csv"

        existing    = {r["filename"] for r in _read_csv(csv_path)}
        roi_samples = filter_utility_meter_roi_samples(um_yolo_path, split_name)
        labels_dir  = um_yolo_path / split_name / "labels"
        new_rows: list[dict] = []

        for img_path, roi_bbox in roi_samples:
            fname = img_path.stem + ".png"
            if fname in existing:
                continue
            # reconstruct label from digit bboxes sorted left-to-right by cx
            label_file = labels_dir / (img_path.stem + ".txt")
            digits: list[tuple[int, float]] = []
            if label_file.exists():
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(parts[0])
                        if cls > 9:
                            continue
                        digits.append((cls, float(parts[1])))
            if not digits:
                continue
            digits.sort(key=lambda d: d[1])
            raw = "".join(str(d[0]) for d in digits)
            label = str(int(raw)) if raw.lstrip("0") else "0"

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            crop = crop_roi_bbox(img, roi_bbox, out_h, out_w)
            cv2.imwrite(str(img_dir / fname), crop)
            new_rows.append({"filename": fname, "label": label})

        _append_csv(csv_path, new_rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label"])
        if write_header:
            w.writeheader()
        w.writerows(rows)


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_ocr_crops(crops_dir: Path, split: str) -> list[tuple[Path, str]]:
    """Return [(img_path, label_str)] from a pre-computed crop directory.

    crops_dir: e.g. DATA_ROOT / "ocr_crops" / "wm_polygon"
    split: "train" or "test"
    """
    csv_path = Path(crops_dir) / split / "labels.csv"
    img_dir  = Path(crops_dir) / split / "images"
    result   = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            result.append((img_dir / row["filename"], row["label"]))
    return result


# ─── Per-digit crops (for per_digit_classifier only) ─────────────────────────

def load_um_digit_crops(
    yolo_path: Path,
    split: str,
    crop_size: int = 32,
) -> list[tuple[np.ndarray, int]]:
    """Return (crop_bgr crop_size×crop_size, digit_class 0–9) for every digit bbox in split."""
    labels_dir = Path(yolo_path) / split / "labels"
    images_dir = Path(yolo_path) / split / "images"
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
                if len(parts) < 5 or int(parts[0]) > 9:
                    continue
                cls = int(parts[0])
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
