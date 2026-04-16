"""Shared OCR data helpers for all 03_ocr notebooks.

Two persistent crop datasets are created by prepare_ocr_crops() and loaded
by load_ocr_crops(). All notebooks use these pre-computed crops.

Crop types:
  wm_polygon — WM roi_polygon perspective-warped (best quality)
  wm_bbox    — WM detected bbox + projection-profile rotation crop (comparison)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import csv
import numpy as np
import cv2

CHARSET = "0123456789"   # 10 digits; CTC blank = index 10
OUT_H   = 64
OUT_W   = 256


# ─── Rotation & corner helpers ───────────────────────────────────────────────

def _estimate_rotation(pts: np.ndarray) -> float:
    """Estimate rotation angle (degrees) of the ROI's long axis from horizontal.

    Uses minAreaRect + long-side direction vector.  Returns angle in (-90, 90].
    The 0°/180° ambiguity is *not* resolved here — downstream orientation
    detection (OCR confidence on 0° vs 180°) handles that.

    Works with any point set (GT polygon, U-Net contour, etc.).
    """
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)

    # Identify the longer pair of sides
    d01 = float(np.linalg.norm(box[1] - box[0]))
    d12 = float(np.linalg.norm(box[2] - box[1]))
    long_vec = (box[1] - box[0]) if d01 >= d12 else (box[2] - box[1])

    angle = float(np.degrees(np.arctan2(long_vec[1], long_vec[0])))

    # Normalise to (-90, 90] — keeps the smaller rotation, punts 180° to
    # orientation detection.
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle


def _order_corners(box: np.ndarray, angle_deg: float) -> np.ndarray:
    """Order 4 box corners as TL, TR, BR, BL for any rotation.

    Rotates the points into a virtual "horizontal" frame (using *angle_deg*),
    then picks top-left / top-right / bottom-right / bottom-left by simple
    y-then-x sorting.  Maps the indices back to the original coordinates.
    """
    cx = box[:, 0].mean()
    cy = box[:, 1].mean()
    cos_a = np.cos(np.radians(-angle_deg))
    sin_a = np.sin(np.radians(-angle_deg))

    # Rotate to horizontal frame
    dx = box[:, 0] - cx
    dy = box[:, 1] - cy
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a

    # Top row = smaller ry, bottom row = larger ry
    idx = list(range(4))
    idx.sort(key=lambda i: (ry[i], rx[i]))
    top = sorted(idx[:2], key=lambda i: rx[i])
    bot = sorted(idx[2:], key=lambda i: rx[i])

    return np.array(
        [box[top[0]], box[top[1]], box[bot[1]], box[bot[0]]],
        dtype=np.float32,
    )


def estimate_roi_rotation(
    polygon: list[tuple[float, float]],
    img_shape: tuple[int, ...],
) -> float:
    """Public helper: rotation angle (degrees) from a normalised polygon.

    Usable in notebooks and in the inference pipeline (U-Net / YOLO outputs).
    """
    h, w = img_shape[:2]
    pts = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)
    return _estimate_rotation(pts)


# ─── Content-based rotation (for bbox path) ───────────────────────────────────

def _projection_score(gray: np.ndarray) -> float:
    """Std of per-row horizontal-edge density (Sobel Y).

    Sobel Y highlights the top/bottom borders of the digit strip.
    When the strip is horizontal those borders concentrate in a few rows →
    high std.  More robust than raw brightness on cluttered backgrounds.
    """
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    proj = np.abs(sobel).sum(axis=1)
    return float(proj.std())


def estimate_rotation_from_crop(
    crop_bgr: np.ndarray,
    angle_range: float = 90.0,
    coarse_step: float = 2.0,
    fine_step: float = 0.5,
) -> float:
    """Estimate rotation angle by maximising horizontal projection sharpness.

    Coarse search over [-angle_range, +angle_range] at *coarse_step* resolution,
    then fine search ±coarse_step around the best with *fine_step* resolution.

    Default *angle_range* is 90° to handle all real-world meter orientations.
    In ``crop_roi_from_detection`` the range is automatically restricted to ±45°
    once the crop is already landscape (prevents rotating landscape → portrait).

    Returns the CCW angle in degrees to pass to ``cv2.getRotationMatrix2D``
    so the digit line becomes horizontal.  No external models needed — works
    identically during training and inference.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = (w / 2.0, h / 2.0)

    def _score(angle: float) -> float:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return _projection_score(rot)

    # Coarse pass
    best_angle = 0.0
    best_score = _score(0.0)
    for a in np.arange(-angle_range, angle_range + coarse_step, coarse_step):
        s = _score(float(a))
        if s > best_score:
            best_score = s
            best_angle = float(a)

    # Fine pass
    for a in np.arange(
        best_angle - coarse_step,
        best_angle + coarse_step + fine_step,
        fine_step,
    ):
        s = _score(float(a))
        if s > best_score:
            best_score = s
            best_angle = float(a)

    return best_angle


# ─── Crop helpers ─────────────────────────────────────────────────────────────

def warp_roi_polygon(
    img_bgr: np.ndarray,
    polygon: list[tuple[float, float]],
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> np.ndarray:
    """Perspective-warp the roi_polygon region to (out_h × out_w).

    Uses rotation-aware corner ordering so it works correctly on rotated
    meters (any angle).  0°/180° ambiguity is not resolved here.
    """
    h, w = img_bgr.shape[:2]
    pts = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    angle = _estimate_rotation(pts)
    corners = _order_corners(box, angle)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(
        img_bgr, cv2.getPerspectiveTransform(corners, dst), (out_w, out_h),
    )


def crop_roi_bbox(
    img_bgr: np.ndarray,
    bbox_cxcywh: tuple[float, float, float, float],
    out_h: int = OUT_H,
    out_w: int = OUT_W,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Axis-aligned bbox crop resized to (out_h × out_w).

    If *rotation_deg* is supplied (e.g. from ``estimate_roi_rotation``), the
    image is first rotated around the bbox centre so the ROI becomes horizontal
    before cropping.  This works both during training (angle from GT polygon)
    and during inference (angle from U-Net polygon or other source).
    """
    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = bbox_cxcywh

    if abs(rotation_deg) > 0.5:
        cx_px, cy_px = cx * w, cy * h
        M = cv2.getRotationMatrix2D((cx_px, cy_px), rotation_deg, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h))

    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_w, out_h))


def crop_roi_from_detection(
    img_bgr: np.ndarray,
    bbox_cxcywh: tuple[float, float, float, float],
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> np.ndarray:
    """Bbox-path crop: rotate the *original image* around bbox centre, then crop.

    1. Cut a padded estimation crop to find the rotation angle.
    2. Coarse 90°: if crop is portrait, decide CW or CCW via projection score.
    3. Fine angle via projection-profile search on coarse-corrected crop.
    4. Rotate the **original image** around (cx, cy) by total angle —
       surrounding real pixels fill in, no border artifacts.
    5. Adaptive padding (more for angles near 45°) → crop → resize.

    Works identically during training and inference.
    """
    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = bbox_cxcywh
    cx_px, cy_px = cx * w, cy * h
    bw_px, bh_px = bw * w, bh * h

    if bw_px < 1 or bh_px < 1:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # ── estimation crop (generous padding to see tilted content) ──────────
    est_pad = 0.3
    ex1 = max(0, int(cx_px - bw_px * (1 + est_pad) / 2))
    ey1 = max(0, int(cy_px - bh_px * (1 + est_pad) / 2))
    ex2 = min(w, int(cx_px + bw_px * (1 + est_pad) / 2))
    ey2 = min(h, int(cy_px + bh_px * (1 + est_pad) / 2))
    est_crop = img_bgr[ey1:ey2, ex1:ex2]

    if est_crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    ech, ecw = est_crop.shape[:2]
    coarse_angle = 0.0

    # ── coarse 90° for portrait crops ────────────────────────────────────
    if ech > ecw * 1.2:
        rot_cw  = cv2.rotate(est_crop, cv2.ROTATE_90_CLOCKWISE)
        rot_ccw = cv2.rotate(est_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        g_cw    = cv2.cvtColor(rot_cw,  cv2.COLOR_BGR2GRAY)
        g_ccw   = cv2.cvtColor(rot_ccw, cv2.COLOR_BGR2GRAY)
        if _projection_score(g_cw) >= _projection_score(g_ccw):
            est_crop = rot_cw
            coarse_angle = -90.0          # CW ≡ -90° in cv2 convention
        else:
            est_crop = rot_ccw
            coarse_angle = 90.0           # CCW ≡ +90°

    # ── fine angle via projection profile ────────────────────────────────
    ech2, ecw2 = est_crop.shape[:2]
    fine_range = 45.0 if (ecw2 > ech2 * 1.1) else 90.0
    fine_angle = estimate_rotation_from_crop(est_crop, angle_range=fine_range)

    total_angle = coarse_angle + fine_angle

    # ── rotate original image around bbox centre ─────────────────────────
    M = cv2.getRotationMatrix2D((cx_px, cy_px), total_angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # ── adaptive padding (more for angles near 45°) + crop ───────────────
    fine_abs_rad = np.radians(np.abs(fine_angle))
    pad = 0.1 + 0.4 * float(np.abs(np.sin(2.0 * fine_abs_rad)))

    meter_long  = max(bw_px, bh_px)
    meter_short = min(bw_px, bh_px)
    crop_w = meter_long  * (1.0 + pad)
    crop_h = meter_short * (1.0 + pad)

    x1 = max(0, int(cx_px - crop_w / 2))
    y1 = max(0, int(cy_px - crop_h / 2))
    x2 = min(w, int(cx_px + crop_w / 2))
    y2 = min(h, int(cy_px + crop_h / 2))

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    return crop # cv2.resize(crop, (out_w, out_h))


# ─── Dataset preparation ──────────────────────────────────────────────────────

def prepare_ocr_crops(
    wm_path: Path,
    dst_root: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
    out_h: int = OUT_H,
    out_w: int = OUT_W,
    roi_detector: Callable[[np.ndarray], tuple[float, float, float, float] | None] | None = None,
) -> None:
    """Create two WM OCR crop datasets under dst_root (idempotent).

    wm_polygon/  — WM perspective warp on roi_polygon (GT annotation).
    wm_bbox/     — WM bbox crop with content-based rotation correction.
                   If *roi_detector* is provided (callable: img_bgr → (cx,cy,w,h)|None),
                   it is used to get the bbox (e.g. YOLO / Faster R-CNN inference).
                   Otherwise falls back to the GT polygon-derived bbox.
                   In both cases rotation is estimated from the crop content
                   (projection profile) — no polygon annotation needed.

    UM is excluded — ROI detection on UM is unreliable (only 45/1552 have ROI).

    Each split: images/<stem>.png  +  labels.csv (filename, label)
    Skips images already present (safe to re-run).
    Note: delete existing wm_bbox crops before re-running with a different
    roi_detector to force regeneration (idempotency is by filename).
    """
    from models.data.unified_loader import load_water_meter_dataset_split
    from models.data.roi_dataset import polygon_to_bbox

    wm_path, dst_root = Path(wm_path), Path(dst_root)

    wm_train, wm_test = load_water_meter_dataset_split(wm_path, train_ratio, seed)

    for split_name, samples in [("train", wm_train), ("test", wm_test)]:
        valid = [s for s in samples if s.roi_polygon is not None and s.value is not None]

        def _crop_bbox(img, s):
            if roi_detector is not None:
                bbox = roi_detector(img)
                if bbox is None:
                    return None          # detector missed — skip sample
            else:
                bbox = polygon_to_bbox(s.roi_polygon)
            return crop_roi_from_detection(img, bbox, out_h, out_w)

        for crop_type, crop_fn in [
            ("wm_polygon", lambda img, s: warp_roi_polygon(img, s.roi_polygon, out_h, out_w)),
            ("wm_bbox",    _crop_bbox),
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
                if crop is None:
                    continue            # roi_detector missed this image
                cv2.imwrite(str(img_dir / fname), crop)
                new_rows.append({"filename": fname, "label": str(int(s.value))})

            _append_csv(csv_path, new_rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(path: Path, rows: list[dict]) -> None:
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
    """Return (crop_bgr crop_size x crop_size, digit_class 0-9) for every digit bbox in split."""
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
