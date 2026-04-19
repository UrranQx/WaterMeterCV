"""YOLO11m OCR predictor: image → (digit_string, mean_confidence).

Mirrors Notebooks/03_ocr/00_pretrained_ocr_yolo11m.ipynb cells 14 and 16.
The predictor stacks the overlap, last-drum, and max-digits heuristics on top
of raw Ultralytics detections.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from watermetercv.ocr.heuristics import (
    MAX_READING_DIGITS,
    apply_max_digits_heuristic,
    apply_ultralytics_last_drum_heuristic,
    apply_ultralytics_overlap_heuristic,
    safe_mean,
)

OcrPredictor = Callable[[np.ndarray], tuple[str, float]]


def load_yolo_ocr_model(weights_path: Path | str):
    """Load YOLO11m OCR model from weights file."""
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"OCR weights not found: {weights_path}")
    return YOLO(str(weights_path))


def extract_ultralytics_digit_detections(image_bgr: np.ndarray, model) -> list[dict]:
    """Run YOLO OCR on an image and return digit detections sorted by cx.

    Non-digit classes (cls > 9) are discarded. Returned dicts match the shape
    expected by heuristics.py.
    """
    if image_bgr is None or model is None:
        return []

    try:
        result = model.predict(image_bgr, verbose=False, imgsz=320, max_det=32)[0]
    except Exception:
        return []

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    digit_mask = boxes.cls <= 9
    digit_boxes = boxes[digit_mask]
    if len(digit_boxes) == 0:
        return []

    sorted_idx = digit_boxes.xywh[:, 0].argsort()
    detections: list[dict] = []
    for i in sorted_idx:
        xyxy = digit_boxes.xyxy[i].tolist()
        xywh = digit_boxes.xywh[i].tolist()
        if len(xyxy) < 4 or len(xywh) < 4:
            continue

        x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
        cx, cy, w, h = (float(v) for v in xywh[:4])
        digit = int(digit_boxes.cls[i].item())
        score = float(digit_boxes.conf[i].item()) if hasattr(digit_boxes, "conf") else 0.0

        detections.append(
            {
                "digit": digit,
                "conf": score,
                "score": score,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy, "w": w, "h": h,
            }
        )

    detections.sort(key=lambda d: d["cx"])
    return detections


def build_yolo_predictor(model, max_digits: int = MAX_READING_DIGITS) -> OcrPredictor:
    """Wrap a loaded YOLO model into a (image -> digits, confidence) predictor."""

    # Warmup — fail fast if the model cannot infer in this environment.
    _ = model.predict(
        np.zeros((64, 256, 3), dtype=np.uint8), verbose=False, imgsz=320, max_det=32
    )

    def _predict(image_bgr: np.ndarray) -> tuple[str, float]:
        detections = extract_ultralytics_digit_detections(image_bgr, model)
        if not detections:
            return "", 0.0

        detections, _ = apply_ultralytics_overlap_heuristic(detections)
        detections, _ = apply_ultralytics_last_drum_heuristic(detections)
        if not detections:
            return "", 0.0

        pred_str = "".join(str(int(d["digit"])) for d in detections)
        confs = [float(d["conf"]) for d in detections]
        pred_str, conf = apply_max_digits_heuristic(
            pred_str, safe_mean(confs), max_digits=max_digits
        )
        return pred_str, conf

    return _predict
