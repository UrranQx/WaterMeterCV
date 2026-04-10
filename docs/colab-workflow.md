# Google Colab Workflow

Гайд по запуску экспериментов WaterMeterCV в Google Colab с GPU.

## Архитектура трёх сред

```
                          ┌─────────────────────────┐
                          │        GitHub           │
                          │  (синхронизация кода)   │
                          └──┬──────────────────┬───┘
                git push/pull│                  │git clone/push
                             │                  │
               ┌─────────────▼──┐      ┌────────▼──────────┐
               │  Ноутбук / ПК  │      │  Google Colab     │
               │  uv sync       │      │  (GPU runtime)    │
               └─────────────┬──┘      └────────┬──────────┘
                             │                  │
                             └────────┬─────────┘
                                      │
               ┌──────────────────────▼──────────────────────┐
               │              Google Drive                    │
               │  MyDrive/WaterMeterCV/                       │
               │  ├── WaterMetricsDATA/  ← датасеты           │
               │  ├── weights/           ← веса моделей       │
               │  └── results/           ← PNG-артефакты      │
               └──────────────────────────────────────────────┘
```

**CODE** → всегда через git (GitHub)
**Бинарные данные** → Google Drive (датасеты, веса, PNG)
**Метрики** → `results/*.json/csv` коммитятся в git (они маленькие)

---

## 1. Приватный репозиторий — PAT

Репо публичным делать **не нужно**. Использовать GitHub Personal Access Token.

**Создать токен:**
1. GitHub → Settings → Developer Settings → Personal access tokens → Fine-grained tokens
2. Выбрать репозиторий: `UrranQx/WaterMeterCV`
3. Права: `Contents: Read and write`

**Добавить в Colab Secrets:**
1. Левая панель Colab → 🔑 Secrets → Add new secret
2. Name: `GITHUB_TOKEN`, Value: токен из шага выше
3. Enable for this notebook: ✓

---

## 2. Стандартная setup-ячейка

Вставить в каждый ноутбук **вместо** текущей setup-ячейки при запуске в Colab:

```python
import sys, subprocess
from pathlib import Path

# Определить среду
IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB:
    try:
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        pass

if IN_COLAB:
    from google.colab import drive, userdata

    # 1. Монтировать Drive (данные и веса)
    drive.mount("/content/drive")

    # 2. Клонировать репо (приватный — через PAT)
    token = userdata.get("GITHUB_TOKEN", "")
    base = f"https://{token}@github.com" if token else "https://github.com"
    if not Path("/content/WaterMeterCV").exists():
        subprocess.run(
            ["git", "clone", f"{base}/UrranQx/WaterMeterCV.git", "/content/WaterMeterCV"],
            check=True
        )

    # 3. Переключиться на нужную ветку (менять под каждый ноутбук)
    BRANCH = "feature/baseline-yolo"
    subprocess.run(["git", "-C", "/content/WaterMeterCV", "checkout", BRANCH], check=True)

    # 4. Установить пакеты, которых нет в Colab runtime
    #    (torch/torchvision НЕ трогать — они предустановлены)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "albumentations", "rapidfuzz", "shapely"],
        check=True
    )

    # 5. Пути
    ROOT         = Path("/content/WaterMeterCV")
    DATA_ROOT    = Path("/content/drive/MyDrive/WaterMeterCV/WaterMetricsDATA")
    WEIGHTS_BASE = Path("/content/drive/MyDrive/WaterMeterCV/weights")
    RESULTS_DIR  = Path("/content/drive/MyDrive/WaterMeterCV/results")
    WORKERS = 2
else:
    # Локально (ноутбук / ПК)
    ROOT         = Path("../..").resolve()
    DATA_ROOT    = ROOT / "WaterMetricsDATA"
    WEIGHTS_BASE = ROOT / "models/weights"
    RESULTS_DIR  = ROOT / "results"
    WORKERS = 0  # Windows CUDA: 0 для стабильности

sys.path.insert(0, str(ROOT))
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

---

## 3. Политика PyTorch

| Среда | Действие | Команда |
|---|---|---|
| Ноутбук (CPU) | Установить через uv | `uv sync` |
| ПК с GPU (CUDA 12.8) | Установить CUDA-версию | `uv sync --extra cuda` |
| Google Colab | Ничего — уже предустановлен | — |

> **Важно:** никогда не запускать `uv sync` в Colab — перепишет предустановленный PyTorch и сломает совместимость с runtime.

---

## 4. Переключение веток

После клона Colab остаётся на `main`. Всегда явно переключаться:

```bash
# В ячейке ноутбука
!git -C /content/WaterMeterCV checkout feature/baseline-yolo

# Создать новую ветку из Colab
!git -C /content/WaterMeterCV checkout -b feature/roi-faster-rcnn origin/develop
```

---

## 5. Синхронизация результатов

### Веса и PNG (бинарное/большое) → Google Drive

Colab пишет веса прямо в `/content/drive/MyDrive/WaterMeterCV/weights/` — они мгновенно доступны в Drive.

**Получить на локальной машине:**
- **Drive desktop app** — авто-синкает при наличии подключения (ничего делать не нужно)
- **rclone** (без desktop app):
  ```bash
  # Установка (Windows)
  winget install Rclone.Rclone

  # Скачать веса с Drive
  rclone copy gdrive:WaterMeterCV/weights ./models/weights/

  # Загрузить локальные веса → Drive
  rclone copy ./models/weights/ gdrive:WaterMeterCV/weights/
  ```

### Метрики и код → git

```bash
# В ячейке Colab после обучения
token = userdata.get("GITHUB_TOKEN", "")
!git -C /content/WaterMeterCV config user.email "you@example.com"
!git -C /content/WaterMeterCV config user.name "Name"
!git -C /content/WaterMeterCV add results/baseline_metrics.json results/baseline_comparison.csv
!git -C /content/WaterMeterCV commit -m "feat: baseline yolo results from Colab"
!git -C /content/WaterMeterCV push \
    https://{token}@github.com/UrranQx/WaterMeterCV.git feature/baseline-yolo
```

На локальной машине:
```bash
git pull origin feature/baseline-yolo  # получить метрики
# веса уже на месте через Drive
```

### Матрица синхронизации

| Откуда → Куда | Механизм |
|---|---|
| Colab → Drive | Запись в `/content/drive/...` (мгновенно) |
| Drive → ПК/ноутбук | Drive desktop app (авто) или `rclone copy gdrive:... ./...` |
| ПК/ноутбук → Drive | Drive desktop app (авто) или `rclone copy ./... gdrive:...` |
| Colab → git | `git add / commit / push` из ячейки |
| git → ПК/ноутбук | `git pull` |

### Что куда

| Артефакт | Хранилище |
|---|---|
| Веса (`.pt`, `.ckpt`) | Google Drive `/weights/` |
| Метрики (`.json`, `.csv`) | git репо (`results/`) |
| PNG-графики (`*_predictions.png`) | Google Drive `/results/` |
| Код, ноутбуки | GitHub |

---

## 6. VS Code Colab Extension

Расширение "Google Colaboratory" (VS Code Marketplace):
- Редактировать `.ipynb` локально в VS Code, выполнять на Colab GPU
- Command Palette → "Colab: Connect to Runtime" → войти в Google аккаунт → выбрать GPU runtime
- **Не решает** проблему данных и пакетов — Drive + PAT нужны в любом случае

---

## 7. Структура Drive

```
MyDrive/WaterMeterCV/
├── WaterMetricsDATA/          ← уже загружено
│   ├── waterMeterDataset/
│   └── utility-meter-.../
├── weights/                   ← создать вручную
└── results/                   ← создать вручную
```

Создать папки один раз вручную в Google Drive или через Colab:
```python
from pathlib import Path
Path("/content/drive/MyDrive/WaterMeterCV/weights").mkdir(parents=True, exist_ok=True)
Path("/content/drive/MyDrive/WaterMeterCV/results").mkdir(parents=True, exist_ok=True)
```

---

## 8. Совместимость Python

Colab runtime: Python 3.12.x. Проект требует `>=3.13` в `pyproject.toml`.

Это **не проблема** пока мы устанавливаем пакеты через `pip install -q` напрямую (без `pip install -e .`). Если понадобится `pip install -e .` — снизить `requires-python` до `>=3.12`.
