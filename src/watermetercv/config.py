"""Service configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_ROI_WEIGHTS = "WATERMETERCV_ROI_WEIGHTS"
ENV_OCR_WEIGHTS = "WATERMETERCV_OCR_WEIGHTS"
ENV_ROI_MODEL_NAME = "WATERMETERCV_ROI_MODEL_NAME"
ENV_OCR_MODEL_NAME = "WATERMETERCV_OCR_MODEL_NAME"
ENV_DEVICE = "WATERMETERCV_DEVICE"

_DEFAULT_ROI = "models/weights/roi_yolo/wm_yolo_roi_20260412_230832/weights/best.pt"
_DEFAULT_OCR = "models/weights/baseline_yolo/yolo11m_20260414_194809/weights/best.pt"


@dataclass(frozen=True)
class ServiceConfig:
    roi_weights: Path
    ocr_weights: Path
    device: str
    roi_model_name_override: str | None = None
    ocr_model_name_override: str | None = None

    @property
    def roi_model_name(self) -> str:
        return self.roi_model_name_override or _derive_model_name(self.roi_weights)

    @property
    def ocr_model_name(self) -> str:
        return self.ocr_model_name_override or _derive_model_name(self.ocr_weights)


def _derive_model_name(weights: Path) -> str:
    # Research layout: .../<run_name>/weights/best.pt → use run_name.
    # Flat layout (e.g. Docker /app/weights/roi.pt) → fall back to file stem.
    if weights.parent.name == "weights":
        return weights.parent.parent.name
    return weights.stem


def load_config() -> ServiceConfig:
    roi = Path(os.environ.get(ENV_ROI_WEIGHTS, _DEFAULT_ROI))
    ocr = Path(os.environ.get(ENV_OCR_WEIGHTS, _DEFAULT_OCR))
    device = os.environ.get(ENV_DEVICE, "cpu")
    return ServiceConfig(
        roi_weights=roi,
        ocr_weights=ocr,
        device=device,
        roi_model_name_override=os.environ.get(ENV_ROI_MODEL_NAME),
        ocr_model_name_override=os.environ.get(ENV_OCR_MODEL_NAME),
    )
