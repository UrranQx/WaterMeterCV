"""End-to-end water meter OCR pipeline.

Single public class ``WaterMeterOCR`` loads ROI + OCR models once, runs the full
flow on each image:
    full image → ROI bbox → rotation-corrected crop → dual-orientation OCR →
    prior-weighted selection → (digits, confidence).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.data.ocr_dataset import crop_roi_from_detection
from models.utils.orientation import dual_read_inference
from watermetercv.config import ServiceConfig, load_config
from watermetercv.ocr.predictor import build_yolo_predictor, load_yolo_ocr_model
from watermetercv.ocr.priors import select_dual_orientation_with_priors
from watermetercv.roi.yolo_roi import detect_roi_bbox, load_roi_model


@dataclass(frozen=True)
class OcrResult:
    digits: str
    confidence: float
    selected_angle: int


class WaterMeterOCR:
    """Stateful pipeline: loads weights once, exposes ``predict(image_bgr)``."""

    def __init__(self, roi_weights: Path, ocr_weights: Path, device: str = "cpu") -> None:
        self._roi_model = load_roi_model(roi_weights)
        self._ocr_model = load_yolo_ocr_model(ocr_weights)
        self._device = device
        self._move_to_device()
        self._predictor = build_yolo_predictor(self._ocr_model)

    @classmethod
    def from_config(cls, cfg: ServiceConfig) -> "WaterMeterOCR":
        return cls(roi_weights=cfg.roi_weights, ocr_weights=cfg.ocr_weights, device=cfg.device)

    @classmethod
    def from_env(cls) -> "WaterMeterOCR":
        return cls.from_config(load_config())

    def _move_to_device(self) -> None:
        if self._device == "cpu":
            return
        try:
            self._roi_model.to(self._device)
            self._ocr_model.to(self._device)
        except Exception:
            pass

    def predict(self, image_bgr: np.ndarray) -> OcrResult:
        if image_bgr is None or image_bgr.size == 0:
            return OcrResult(digits="", confidence=0.0, selected_angle=0)

        bbox = detect_roi_bbox(image_bgr, self._roi_model)
        if bbox is None:
            return OcrResult(digits="", confidence=0.0, selected_angle=0)

        crop = crop_roi_from_detection(image_bgr, bbox)
        if crop is None or crop.size == 0:
            return OcrResult(digits="", confidence=0.0, selected_angle=0)

        dual = dual_read_inference(crop, self._predictor)
        selected = select_dual_orientation_with_priors(
            dual, image_bgr=crop, ocr_model=self._ocr_model
        )
        return OcrResult(
            digits=str(selected["selected_pred"]),
            confidence=float(selected["selected_conf"]),
            selected_angle=int(selected["selected_angle"]),
        )
