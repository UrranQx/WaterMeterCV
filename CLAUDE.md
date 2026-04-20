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

# Docker (CPU / GPU). Веса запекаются внутрь — контекст собирается из корня репо.
docker build -f docker/Dockerfile.cpu -t watermetercv:cpu .
docker build -f docker/Dockerfile.gpu -t watermetercv:gpu .
docker run --rm -p 8000:8000 watermetercv:cpu
docker run --rm --gpus all -p 8000:8000 watermetercv:gpu

# Service regression bench (сравнение с single-stage research-пайплайном)
uv run python scripts/bench_service.py --url http://localhost:8000 --tag cpu   # или --tag gpu
# → results/service_bench_{cpu,gpu}.csv + console summary
# Ключевая метрика: "docker vs single (same pred): 374/374" = пайплайн не сломался.
# "exact match (raw): 0/374" — ожидаемо: GT без ведущих нулей, normalized = реальная точность.

# Docker Hub push (после docker build)
docker tag watermetercv:cpu urran/watermetercv:cpu && docker push urran/watermetercv:cpu
docker tag watermetercv:gpu urran/watermetercv:gpu && docker push urran/watermetercv:gpu

# ARM64 CPU-образ (для коллег на Apple Silicon). Требует docker-container buildx-builder + QEMU.
docker buildx create --name wmcv-arm --driver docker-container --use --bootstrap  # один раз
docker buildx build --builder wmcv-arm --platform linux/arm64 \
    -f docker/Dockerfile.cpu.arm64 -t urran/watermetercv:cpu-arm64 --push .
docker buildx imagetools inspect urran/watermetercv:cpu-arm64
# Локальная сборка arm64 (single-platform --load работает только на docker-container driver):
docker buildx build --builder wmcv-arm --platform linux/arm64 \
    -f docker/Dockerfile.cpu.arm64 -t watermetercv:cpu-arm64-local --load .
docker run --rm --platform linux/arm64 -p 8000:8000 watermetercv:cpu-arm64-local

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
- `results/` — metrics JSON/CSV (PNG/JPG gitignored)
- `scripts/` — bench_service, debug_bbox_crop, visualization helpers
- `tests/` — 12 test modules: unit (heuristics, pipeline, orientation, datasets) + integration (TestClient service)
- `docker/` — `Dockerfile.cpu` (amd64), `Dockerfile.cpu.arm64` (Apple Silicon), `Dockerfile.gpu`, `docker-compose.yml`

## ROI Detection (feature/roi-detection — complete)

Best model: **YOLO11n** (WM IoU=0.94, 100% detection, 23 ms) or **U-Net** (IoU=0.877, 5 ms).
Full results and decisions: `docs/notes/roi-detection-findings.md`.

**UM dataset is NOT suitable for ROI** — only 45/1552 images have ROI labels; all models fail.
Do not use utility-meter for ROI training. WaterMeterDataset only.

OCR prep: two crop paths — `wm_bbox` (rotation-corrected bbox) and `wm_polygon` (perspective warp). Both in `models/data/ocr_dataset.py`.
Orientation: solved via dual-read (0°/180°) + `select_dual_orientation_with_priors`. No separate classifier needed.

## Evaluation Gotchas

GT строки из utility-meter: `str(int(sample.value))` срезает ведущие нули (`00482` → `482`).
Считать FSA/CER в двух режимах: raw (как есть) и normalized (`lstrip("0")`). Сервис всегда возвращает полную строку (все 8 барабанов), так что для сравнения с GT используйте normalized.
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

GitFlow-lite: `master` ← `develop` ← `feature/<name>`
Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`
Feature branches: `feature/data-exploration`, `feature/baseline-yolo`, `feature/roi-*`, `feature/ocr-*`, `feature/combo-*`
Tags: `v0.2-roi` (only tag so far; next: `v0.3-service` after final stabilisation)

## Service layer (`src/watermetercv/`)

Inference-only пакет, отдельный от research-кода.

- `pipeline.py` — `WaterMeterOCR`: загружает оба YOLO-веса, `predict(image_bgr) -> OcrResult(digits, confidence, selected_angle)`.
- `ocr/predictor.py` — `build_yolo_predictor`, `extract_ultralytics_digit_detections` (imgsz=320, max_det=32).
- `ocr/heuristics.py` — max-digits, overlap, last-drum, leading-zero, long-tail zero.
- `ocr/priors.py` — `select_dual_orientation_with_priors`, red-bbox horizontal cluster.
- `roi/yolo_roi.py` — `detect_roi_bbox` (imgsz=640, conf=0.001).
- `service/app.py` — FastAPI, lifespan-loaded модели. `POST /recognize` (primary, backend-контракт: поле `file`, `{"value": int}`, 422 при no-detect). `POST /predict` (internal/debug: поле `image`, `{"digits", "confidence"}`). `GET /healthz`, `GET /info`.
- `config.py` — env-based `ServiceConfig`.

Env-переменные: `WATERMETERCV_ROI_WEIGHTS`, `WATERMETERCV_OCR_WEIGHTS`, `WATERMETERCV_ROI_MODEL_NAME`, `WATERMETERCV_OCR_MODEL_NAME`, `WATERMETERCV_DEVICE` (`cpu` | `cuda:0`), `WATERMETERCV_HOST`, `WATERMETERCV_PORT`.

Контракт API и рекомендации для клиентов — `docs/service.md`.

Docker-образы собираются через `pip install` с явным списком inference-deps (не `uv sync`), чтобы не тащить research-стек. CPU ≈ 560 MB compressed, GPU ≈ 3.1 GB compressed. Веса запекаются в `/app/weights/`. Regression bench (`scripts/bench_service.py`) показывает 374/374 совпадений с research single-stage на test split.

Docker Hub теги: `urran/watermetercv:cpu` (amd64) · `urran/watermetercv:cpu-arm64` (Apple Silicon / Linux arm64) · `urran/watermetercv:gpu` (amd64 + NVIDIA). `:cpu-arm64` — отдельный тег в phase-1 rollout; после подтверждения от коллег планируется схлопнуть `:cpu`+`:cpu-arm64` в multi-arch manifest под одним тегом `:cpu`.

**Docker gotchas** (легко наступить при правке Dockerfile'ов):
- **Windows port conflict**: на Windows два процесса могут слушать 0.0.0.0:8000 одновременно. После остановки `watermetercv-serve` убедись: `netstat -an | findstr ":8000"` — если stale python.exe → `taskkill //PID <pid> //F`.
- `pip install torch` с default PyPI на **amd64** тянет полный CUDA runtime (~8 GB) даже когда ты хотел CPU. Для CPU-образа (amd64) обязательно `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision` **до** всех остальных install.
- На **arm64** default PyPI `pip install torch` — это уже CPU wheel (NVIDIA на ARM не существует). `--index-url .../whl/cpu` на arm64 не нужен и для свежих torch может не содержать aarch64 wheels. Поэтому `Dockerfile.cpu.arm64` — отдельный файл без index-url.
- `ultralytics` транзитивно ставит `opencv-python`, который перекрывает `opencv-python-headless` (оба пишутся в `/cv2/`). Runtime падает на `ImportError: libxcb.so.1`. Fix: после основного install — `pip uninstall -y opencv-python opencv-python-headless && pip install --no-deps opencv-python-headless`.
- В Dockerfile прописаны env-переменные `WATERMETERCV_ROI_MODEL_NAME` / `WATERMETERCV_OCR_MODEL_NAME`, чтобы `/info` возвращал осмысленные имена моделей — flat-path `/app/weights/roi.pt` не даёт их авто-вывести.
- `docker buildx build --platform linux/arm64 --load` **требует** docker-container driver (default docker driver не грузит foreign-arch в локальный daemon). One-time: `docker buildx create --name wmcv-arm --driver docker-container --use --bootstrap`. Для `--push` в Docker Hub — аналогично (multi-arch manifest не хранится в docker-daemon).
- QEMU-эмуляция на Docker Desktop (Windows) обычно преднастроена. Если отсутствует: `docker run --privileged --rm tonistiigi/binfmt --install all`. Признак отсутствия — ошибки вида `exec format error` при `docker run --platform linux/arm64 <amd64-host-image>`.

## OCR Crops

Two crop paths in `models/data/ocr_dataset.py`:
- `wm_polygon` — perspective warp on GT roi_polygon (`warp_roi_polygon`).
- `wm_bbox` — rotation-corrected bbox crop (`crop_roi_from_detection`). Поворот — вокруг центра **исходного** изображения, не crop'а (чтобы борта заполнялись реальными пикселями). Детали конвенции cv2, coarse/fine-angle, adaptive padding — в docstring функции.

Visual debug: `uv run python scripts/debug_bbox_crop.py --batch`.
