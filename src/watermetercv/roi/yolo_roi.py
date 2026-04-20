"""YOLO11n ROI detector wrapper: image → normalized (cx, cy, w, h).

Mirrors ``build_wm_roi_detector`` from Notebooks/03_ocr/00_pretrained_ocr_yolo11m.ipynb cell 6.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_roi_model(weights_path: Path | str):
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"ROI weights not found: {weights_path}")
    return YOLO(str(weights_path))


def detect_roi_bbox(image_bgr: np.ndarray, model) -> tuple[float, float, float, float] | None:
    """Detect the single strongest ROI bbox; return normalized (cx, cy, w, h)."""
    try:
        result = model.predict(image_bgr, verbose=False, conf=0.001, imgsz=640, max_det=16)[0]
    except Exception:
        return None

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    try:
        idx = int(boxes.conf.argmax().item()) if hasattr(boxes, "conf") else 0
    except Exception:
        idx = 0

    b = boxes.xywhn[idx]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])
