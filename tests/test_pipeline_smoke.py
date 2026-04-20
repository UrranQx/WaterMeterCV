"""Smoke test for WaterMeterOCR against real images and weights.

Skipped automatically if weights or the WaterMeters dataset are missing.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
ROI_WEIGHTS = ROOT / "models/weights/roi_yolo/wm_yolo_roi_20260412_230832/weights/best.pt"
OCR_WEIGHTS = ROOT / "models/weights/baseline_yolo/yolo11m_20260414_194809/weights/best.pt"
IMAGE_DIR = ROOT / "WaterMetricsDATA/waterMeterDataset/WaterMeters/images"

pytestmark = pytest.mark.skipif(
    not (ROI_WEIGHTS.exists() and OCR_WEIGHTS.exists() and IMAGE_DIR.exists()),
    reason="Weights or dataset missing; smoke test skipped.",
)


@pytest.fixture(scope="module")
def pipeline():
    from watermetercv import WaterMeterOCR
    return WaterMeterOCR(roi_weights=ROI_WEIGHTS, ocr_weights=OCR_WEIGHTS, device="cpu")


def _sample_image():
    for p in sorted(IMAGE_DIR.glob("*.jpg")):
        return p
    pytest.skip("No jpg images in dataset dir")


def test_pipeline_returns_digits_and_confidence(pipeline):
    image = cv2.imread(str(_sample_image()))
    assert image is not None
    result = pipeline.predict(image)
    assert isinstance(result.digits, str)
    assert all(ch.isdigit() for ch in result.digits)
    assert 0.0 <= result.confidence <= 1.0
    assert result.selected_angle in (0, 180)


def test_pipeline_handles_empty_image(pipeline):
    import numpy as np
    result = pipeline.predict(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result.digits == ""
    assert result.confidence == 0.0
