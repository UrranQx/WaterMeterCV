"""Service configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_ROI_WEIGHTS = "WATERMETERCV_ROI_WEIGHTS"
ENV_OCR_WEIGHTS = "WATERMETERCV_OCR_WEIGHTS"
ENV_DEVICE = "WATERMETERCV_DEVICE"

_DEFAULT_ROI = "models/weights/roi_yolo/wm_yolo_roi_20260412_230832/weights/best.pt"
_DEFAULT_OCR = "models/weights/baseline_yolo/yolo11m_20260414_194809/weights/best.pt"


@dataclass(frozen=True)
class ServiceConfig:
    roi_weights: Path
    ocr_weights: Path
    device: str

    @property
    def roi_model_name(self) -> str:
        return self.roi_weights.parent.parent.name

    @property
    def ocr_model_name(self) -> str:
        return self.ocr_weights.parent.parent.name


def load_config() -> ServiceConfig:
    roi = Path(os.environ.get(ENV_ROI_WEIGHTS, _DEFAULT_ROI))
    ocr = Path(os.environ.get(ENV_OCR_WEIGHTS, _DEFAULT_OCR))
    device = os.environ.get(ENV_DEVICE, "cpu")
    return ServiceConfig(roi_weights=roi, ocr_weights=ocr, device=device)
