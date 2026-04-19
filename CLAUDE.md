# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CV pipeline for reading water meter digits from photos. Research фаза завершена, выбран и зафиксирован пайплайн; упакован в FastAPI-сервис и два Docker-образа (CPU / GPU).

**Winning pipeline:** `YOLO11n (ROI)` → `crop_roi_from_detection` → `YOLO11m (OCR, dual-orientation 0°/180°)` → `select_dual_orientation_with_priors`. Канон: `Notebooks/03_ocr/00_pretrained_ocr_yolo11m.ipynb`.

## Commands

```bash
# Install deps (research stack)
uv sync

# Install deps for running the FastAPI service locally
uv sync --extra service              # CPU
uv sync --extra service --extra cuda # GPU (CUDA 13.0)

# Start the service locally
uv run watermetercv-serve            # binds 0.0.0.0:8000

# Run tests
uv run pytest tests/ -v

# Fast OCR tests only (skip slow prepare_ocr_crops; useful for iteration)
uv run pytest tests/test_ocr_dataset.py -k "not Prepare and not LoadOcr" -v

# Run a single test
uv run pytest tests/path/test_file.py::test_name -v

# Python REPL
uv run python
```

## Environments

| Среда | Команда | Примечание |
|---|---|---|
| Ноутбук (CPU) | `uv sync` | CPU torch с PyPI |
| ПК с GPU (CUDA 13.0) | `uv sync --extra cuda` | CUDA torch с pytorch.org/whl/cu130 |
| Google Colab | `pip install -q ultralytics albumentations rapidfuzz shapely` | torch предустановлен, `uv sync` не запускать |

Полный гайд по Colab (PAT, Drive, синхронизация весов, ветки): `docs/colab-workflow.md`

## Notebooks

Jupyter format: `nbformat_minor: 4` (не 5 — требует cell id), `language_info.version: "3.13.0"`.
Source cells можно хранить как строку или список строк — оба формата валидны.

## Architecture

- `models/data/` — unified dataset loaders (both datasets → `UnifiedSample`); crop helpers (`crop_roi_from_detection`, `warp_roi_polygon`) используются и в сервисе.
- `models/metrics/` — shared evaluation (full-string acc, per-digit acc, CER, IoU, inference time)
- `models/utils/` — visualization, result logging, **`orientation.py` (`dual_read_inference`, `DualOrientationResult`)** — вызывается из сервиса
- `models/weights/`, `models/checkpoints/` — gitignored
- `Notebooks/` — experiments (00 EDA → 01 baseline → 02 ROI → 03 OCR → 04 combinations)
- `src/watermetercv/` — production inference package (ROI + OCR + priors + FastAPI). Собирается `hatchling`, entry point `watermetercv-serve`.
- `configs/` — YAML hyperparams (`configs/default.yaml`)
- `results/` — metrics CSVs, `comparison.md`
- `docker/` — `Dockerfile.cpu`, `Dockerfile.gpu`, `docker-compose.yml`

## ROI Detection (feature/roi-detection — complete)

Best model: **YOLO11n** (WM IoU=0.94, 100% detection, 23 ms) or **U-Net** (IoU=0.877, 5 ms).
Full results and decisions: `docs/notes/roi-detection-findings.md`.

**UM dataset is NOT suitable for ROI** — only 45/1552 images have ROI labels; all models fail.
Do not use utility-meter for ROI training. WaterMeterDataset only.

OCR prep: two paths — bbox crop (YOLO/Faster R-CNN) and polygon/perspective warp (U-Net mask).
Both will be implemented and compared in `Notebooks/03_*`.
Orientation fix: try 0° and 180° reads, keep valid integer result (orientation classifier is future work).

## Evaluation Gotchas

GT строки из utility-meter: `str(int(sample.value))` срезает ведущие нули (`00482` → `482`).
Считать FSA/CER в двух режимах: raw (как есть) и normalized (`lstrip("0")`).
Baseline результат: yolo11n mAP50=0.617, FSA=0.047, combined=0.504 — низкий FSA частично из-за этого.
Аннотации utility-meter могут быть несогласованы с поворотами изображений (видно на val_batch labels).

`results/`: PNG/JPG gitignored, JSON/CSV метрики коммитятся в git.

## Datasets (`WaterMetricsDATA/`, gitignored)

**waterMeterDataset/WaterMeters/**
- `data.csv`: columns `photo_name, value, location`
- `location` is a Python dict string — parse with `ast.literal_eval`; polygon coords are normalized

**utility-meter-reading-dataset-for-automatic-reading-yolo.v4i** (YOLO + COCO variants)
- COCO category IDs: 1-10 = digits 0-9, **11 = "Reading Digit" (ROI bbox)**, 12-14 = colors
- YOLO class IDs: 0-9 = digits, **10 = "Reading Digit" (ROI)**
- Ground truth value: reconstruct by sorting digit bboxes left-to-right by `cx`

## Git Workflow

GitFlow-lite: `main` ← `develop` ← `feature/<name>`
Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`
Feature branches: `feature/data-exploration`, `feature/baseline-yolo`, `feature/roi-*`, `feature/ocr-*`, `feature/combo-*`
Tags: `v0.1-baseline`, `v0.2-roi`, `v0.3-research-complete`

## Service layer (`src/watermetercv/`)

Inference-only пакет, отдельный от research-кода.

- `pipeline.py` — `WaterMeterOCR`: загружает оба YOLO-веса, `predict(image_bgr) -> OcrResult(digits, confidence, selected_angle)`.
- `ocr/predictor.py` — `build_yolo_predictor`, `extract_ultralytics_digit_detections` (imgsz=320, max_det=32).
- `ocr/heuristics.py` — max-digits, overlap, last-drum, leading-zero, long-tail zero.
- `ocr/priors.py` — `select_dual_orientation_with_priors`, red-bbox horizontal cluster.
- `roi/yolo_roi.py` — `detect_roi_bbox` (imgsz=640, conf=0.001).
- `service/app.py` — FastAPI, lifespan-loaded модели, `POST /predict`, `GET /healthz`, `GET /info`.
- `config.py` — env-based `ServiceConfig`.

Env-переменные: `WATERMETERCV_ROI_WEIGHTS`, `WATERMETERCV_OCR_WEIGHTS`, `WATERMETERCV_DEVICE` (`cpu` | `cuda:0`), `WATERMETERCV_HOST`, `WATERMETERCV_PORT`.

Контракт API и рекомендации для клиентов — `docs/service.md`.

Docker-образы собираются через `pip install` с явным списком inference-deps (не `uv sync`), чтобы не тащить research-стек (pandas, easyocr, pytesseract, SMP, ipykernel). CPU ≈ 500 MB, GPU ≈ 3 GB. Веса запекаются в `/app/weights/`. `opencv-python-headless` ставится только в Docker; локально остаётся `opencv-python`.

## OCR Crops

Two crop paths in `models/data/ocr_dataset.py`:
- `wm_polygon` — perspective warp on GT roi_polygon (`warp_roi_polygon`)
- `wm_bbox` — rotation-corrected bbox crop (`crop_roi_from_detection`)

**bbox rotation approach:** rotate the *original image* around bbox centre (`getRotationMatrix2D`), then crop with adaptive padding.
Never rotate the crop — the original fills borders with real pixels, no artifacts.
- cv2 convention: positive angle = CCW. `ROTATE_90_CLOCKWISE` ≡ angle=−90° in `getRotationMatrix2D`.
- `total_angle = coarse_angle + fine_angle`; coarse ∈ {0, −90, +90} (portrait→landscape via projection score).
- Adaptive padding: `pad = 0.1 + 0.4 * |sin(2 * fine_angle_rad)|` — more headroom near 45°.

Visual debug: `uv run python scripts/debug_bbox_crop.py --batch` → 5 PNGs by angle band in `results/`.

## Plan & Spec

- Spec: `docs/superpowers/specs/2026-04-08-ml-research-plan-design.md`
- Implementation plan (20 tasks): `docs/superpowers/plans/2026-04-08-ml-research-plan.md`
