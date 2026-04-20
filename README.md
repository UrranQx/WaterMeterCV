# WaterMeterCV

CV-пайплайн для извлечения цифр с фотографии водосчётчика.

Вход — JPEG/PNG с меткой счётчика; выход — строка цифр и confidence.

## Статус

Research-фаза завершена. Зафиксированный пайплайн:

```
image ─► YOLO11n (ROI) ─► crop ─► YOLO11m (OCR, 0°+180°) ─► priors voting ─► digits
```

- ROI-детектор: `wm_yolo_roi_20260412_230832` (YOLO11n, WM IoU ≈ 0.94).
- OCR-детектор: `yolo11m_20260414_194809` (YOLO11m, single-stage digit detection, ~39 MB).
- Dual-orientation read (0°/180°) с голосованием priors: leading-zero, long-tail zero, red-bbox cluster, no-red short tail, ultra-overlap, last-drum. Канонический вариант — `Notebooks/03_ocr/00_pretrained_ocr_yolo11m.ipynb`.
- Подробности ROI-исследования: `docs/notes/roi-detection-findings.md`.

## Запуск сервиса

Три сценария. Во всех — HTTP-сервер на `:8000`, endpoint `POST /predict`.

### 1. Локально (uv)

```bash
uv sync --extra service           # CPU
# или
uv sync --extra service --extra cuda   # GPU (CUDA 13.0)

uv run watermetercv-serve
```

### 2. Docker — CPU

```bash
docker build -f docker/Dockerfile.cpu -t watermetercv:cpu .
docker run --rm -p 8000:8000 watermetercv:cpu
```

### 3. Docker — GPU

Требуется `nvidia-container-toolkit` на хосте.

```bash
docker build -f docker/Dockerfile.gpu -t watermetercv:gpu .
docker run --rm --gpus all -p 8000:8000 watermetercv:gpu
```

Удобная альтернатива через `docker compose`:

```bash
docker compose -f docker/docker-compose.yml --profile cpu up --build
docker compose -f docker/docker-compose.yml --profile gpu up --build
```

## Пример запроса

```bash
curl -F "image=@meter.jpg" http://localhost:8000/predict
```

Ответ:

```json
{"digits": "00123456", "confidence": 0.87}
```

Служебные эндпойнты:

```bash
curl http://localhost:8000/healthz   # {"status":"ok"}
curl http://localhost:8000/info      # {"roi_model":"...","ocr_model":"...","device":"..."}
```

Полный контракт, коды ошибок, рекомендации по интеграции — `docs/service.md`.

## Конфигурация

Переменные окружения (все опциональные, в Docker-образах выставлены по умолчанию):

| Переменная | Назначение | Default |
|---|---|---|
| `WATERMETERCV_ROI_WEIGHTS` | путь к `.pt` весам ROI-модели | `models/weights/roi_yolo/.../best.pt` |
| `WATERMETERCV_OCR_WEIGHTS` | путь к `.pt` весам OCR-модели | `models/weights/baseline_yolo/yolo11m_.../best.pt` |
| `WATERMETERCV_DEVICE` | `cpu` / `cuda:0` | `cpu` (в GPU-образе `cuda:0`) |
| `WATERMETERCV_HOST` | bind host | `0.0.0.0` |
| `WATERMETERCV_PORT` | bind port | `8000` |

## Разработка и research

- `Notebooks/` — эксперименты (00 EDA → 01 baseline → 02 ROI → 03 OCR → 04 combinations).
- `models/data/`, `models/utils/`, `models/metrics/` — unified dataset, orientation/crop helpers, общие метрики.
- `src/watermetercv/` — inference-only пакет сервиса (ROI + OCR + priors + FastAPI).
- `configs/default.yaml` — гиперпараметры training-пайплайнов.
- `tests/` — unit + integration (TestClient).

Запуск тестов:

```bash
uv run pytest tests/ -v
```

Остальные проектные соглашения (git-workflow, работа в Colab, структура датасетов) — см. `CLAUDE.md`.

## Лицензия

Код и веса моделей, распространяемые в этом репозитории, лицензированы под
**AGPL-3.0** (см. `LICENSE`).

![[4x.avif]]

Учебный проект. Один из датасетов, использованных для обучения
(ROI-детектор), распространяется под **CC BY-NC-ND 4.0** — коммерческое
использование сервиса и запечённых в Docker-образы весов запрещено без
отдельного лицензионного гранта. Подробности и полный список атрибуций —
`NOTICE.md`.
