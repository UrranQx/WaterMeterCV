# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CV pipeline for reading water meter digits from photos. Research phase: testing 11 ML approaches (ROI detection + OCR) to find best pipeline.

## Commands

```bash
# Install deps
uv sync

# Run tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/path/test_file.py::test_name -v

# Python REPL
uv run python
```

## Environments

| Среда | Команда | Примечание |
|---|---|---|
| Ноутбук (CPU) | `uv sync` | CPU torch с PyPI |
| ПК с GPU (CUDA 12.8) | `uv sync --extra cuda` | CUDA torch с pytorch.org/whl/cu128 |
| Google Colab | `pip install -q ultralytics albumentations rapidfuzz shapely` | torch предустановлен, `uv sync` не запускать |

Полный гайд по Colab (PAT, Drive, синхронизация весов, ветки): `docs/colab-workflow.md`

## Notebooks

Jupyter format: `nbformat_minor: 4` (не 5 — требует cell id), `language_info.version: "3.13.0"`.
Source cells можно хранить как строку или список строк — оба формата валидны.

## Architecture

- `models/data/` — unified dataset loaders (both datasets → `UnifiedSample`)
- `models/metrics/` — shared evaluation (full-string acc, per-digit acc, CER, IoU, inference time)
- `models/utils/` — visualization, result logging
- `models/weights/`, `models/checkpoints/` — gitignored
- `Notebooks/` — experiments (00 EDA → 01 baseline → 02 ROI → 03 OCR → 04 combinations)
- `src/` — FastAPI service layer (future, out of current scope)
- `configs/` — YAML hyperparams (`configs/default.yaml`)
- `results/` — metrics CSVs, `comparison.md`

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

## Plan & Spec

- Spec: `docs/superpowers/specs/2026-04-08-ml-research-plan-design.md`
- Implementation plan (20 tasks): `docs/superpowers/plans/2026-04-08-ml-research-plan.md`
