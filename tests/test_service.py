"""Integration tests for the FastAPI service using TestClient."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROI_WEIGHTS = ROOT / "models/weights/roi_yolo/wm_yolo_roi_20260412_230832/weights/best.pt"
OCR_WEIGHTS = ROOT / "models/weights/baseline_yolo/yolo11m_20260414_194809/weights/best.pt"
IMAGE_DIR = ROOT / "WaterMetricsDATA/waterMeterDataset/WaterMeters/images"

pytestmark = pytest.mark.skipif(
    not (ROI_WEIGHTS.exists() and OCR_WEIGHTS.exists() and IMAGE_DIR.exists()),
    reason="Weights or dataset missing; service test skipped.",
)


@pytest.fixture(scope="module")
def client(monkeypatch_session):
    monkeypatch_session.setenv("WATERMETERCV_ROI_WEIGHTS", str(ROI_WEIGHTS))
    monkeypatch_session.setenv("WATERMETERCV_OCR_WEIGHTS", str(OCR_WEIGHTS))
    monkeypatch_session.setenv("WATERMETERCV_DEVICE", "cpu")

    # Import after env is set so module-level load_config() picks up the paths.
    import importlib

    import watermetercv.service.app as app_module
    importlib.reload(app_module)

    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _sample_image_path() -> Path:
    for p in sorted(IMAGE_DIR.glob("*.jpg")):
        return p
    pytest.skip("No jpg images in dataset dir")


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_info(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "roi_model" in body and "ocr_model" in body and body["device"] == "cpu"


def test_predict_happy_path(client):
    img_path = _sample_image_path()
    with open(img_path, "rb") as f:
        resp = client.post("/predict", files={"image": (img_path.name, f, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert all(ch.isdigit() for ch in body["digits"])
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_non_image(client):
    resp = client.post(
        "/predict",
        files={"image": ("hello.txt", b"not an image at all", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_requires_file(client):
    resp = client.post("/predict")
    assert resp.status_code == 422
