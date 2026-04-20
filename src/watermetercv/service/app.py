"""FastAPI application exposing the water meter OCR pipeline.

Endpoints:
    POST /recognize multipart image (field `file`)  -> {value: int}
    POST /predict   multipart image (field `image`) -> {digits, confidence}
    GET  /healthz   ping
    GET  /info      model versions and device

Run locally:
    uv run watermetercv-serve
Or under any ASGI server:
    uvicorn watermetercv.service.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from watermetercv.config import load_config
from watermetercv.pipeline import WaterMeterOCR
from watermetercv.service.schemas import (
    HealthResponse,
    InfoResponse,
    PredictResponse,
    RecognizeResponse,
)

_MAX_IMAGE_BYTES = 10 * 1024 * 1024

_cfg = load_config()
_pipeline: WaterMeterOCR | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    _pipeline = WaterMeterOCR.from_config(_cfg)
    yield
    _pipeline = None


app = FastAPI(
    title="Water Meter OCR",
    version="0.1.0",
    description="ROI detection + digit OCR pipeline for water meter photos.",
    lifespan=_lifespan,
)


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    return image


async def _validate_and_decode(upload: UploadFile) -> np.ndarray:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")
    return _decode_image(data)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(
        roi_model=_cfg.roi_model_name,
        ocr_model=_cfg.ocr_model_name,
        device=_cfg.device,
    )


@app.post(
    "/recognize",
    response_model=RecognizeResponse,
    responses={
        400: {"description": "Invalid image"},
        413: {"description": "Image too large"},
        422: {"description": "No reading detected"},
    },
)
async def recognize(file: UploadFile = File(..., description="JPEG or PNG image")) -> RecognizeResponse:
    image_bgr = await _validate_and_decode(file)
    result = _pipeline.predict(image_bgr)
    if not result.digits:
        raise HTTPException(status_code=422, detail="No reading detected")
    return RecognizeResponse(value=int(result.digits))


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"description": "Invalid image"}, 413: {"description": "Image too large"}},
)
async def predict(image: UploadFile = File(..., description="JPEG or PNG image")) -> PredictResponse:
    image_bgr = await _validate_and_decode(image)
    result = _pipeline.predict(image_bgr)
    return PredictResponse(digits=result.digits, confidence=round(result.confidence, 4))


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})


def main() -> None:
    """Entry point for the ``watermetercv-serve`` console script."""
    import uvicorn

    host = os.environ.get("WATERMETERCV_HOST", "0.0.0.0")
    port = int(os.environ.get("WATERMETERCV_PORT", "8000"))
    uvicorn.run("watermetercv.service.app:app", host=host, port=port, log_level="info")
