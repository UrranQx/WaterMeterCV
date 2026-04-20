"""Water meter OCR — production inference package.

Top-level entry point:
    from watermetercv import WaterMeterOCR
    ocr = WaterMeterOCR.from_env()
    result = ocr.predict(image_bgr)  # -> OcrResult(digits, confidence)
"""
from __future__ import annotations

from watermetercv.pipeline import OcrResult, WaterMeterOCR

__all__ = ["OcrResult", "WaterMeterOCR"]
