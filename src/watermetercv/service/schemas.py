"""Pydantic schemas for the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    digits: str = Field(..., description="Recognized digit string, e.g. '123456'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence of kept digits")


class RecognizeResponse(BaseModel):
    value: int = Field(..., ge=0, description="Recognized meter reading as integer; leading zeros dropped")


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    roi_model: str
    ocr_model: str
    device: str


class ErrorResponse(BaseModel):
    detail: str
