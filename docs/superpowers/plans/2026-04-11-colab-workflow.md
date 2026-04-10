# Plan: Google Colab Workflow — Документация и конфиги

## Context

Проект работает на трёх средах: ноутбук (CPU/слабый GPU), ПК с CUDA GPU, Google Colab. Сейчас нет
задокументированного Colab-воркфлоу, неясна политика установки PyTorch для каждой среды, и нет стратегии
для работы с gitignored-данными (датасеты, веса).

**Факты о среде:**
- Colab Python: 3.12.13 (пользователь сообщает; `requires-python = ">=3.13"` в pyproject.toml не меняем — уточнить совместимость на верификации)
- Датасеты уже загружены на Google Drive
- Ноутбуки уже имеют паттерн `git clone` в setup-ячейке
- VS Code Colab Extension будет использоваться

---

## Архитектура трёх сред

```
                          ┌─────────────────────────┐
                          │        GitHub           │
                          │  (center for CODE)      │
                          └──┬──────────────────┬───┘
                git push/pull│                  │git clone/push
                             │                  │
               ┌─────────────▼──┐      ┌────────▼──────────┐
               │  Ноутбук / ПК  │      │  Google Colab     │
               │  (local dev)   │      │  (GPU runtime)    │
               │  uv sync       │      │  VS Code ext.     │
               └─────────────┬──┘      └────────┬──────────┘
                             │                  │
          Google Drive       │                  │ read data /
          desktop app        │                  │ write weights
          (auto-sync)        │                  │
               ┌─────────────▼──────────────────▼──────────┐
               │              Google Drive                  │
               │  MyDrive/WaterMeterCV/                     │
               │  ├── WaterMetricsDATA/  ← уже загружено    │
               │  ├── weights/           ← Colab сохраняет  │
               │  └── results/images/    ← PNG-графики      │
               └────────────────────────────────────────────┘

CODE      → всегда через git (GitHub центр синхронизации)
БИНАРНЫЕ  → Google Drive (датасеты, веса, большие артефакты)
МЕТРИКИ   → results/*.json/csv коммитятся в git (они маленькие)
```

Прямой git pull с GitHub на ПК работает независимо — это основной workflow для code changes. Drive — только для данных. Два механизма не конкурируют.

---

## Изменяемые файлы

| Файл | Действие |
|---|---|
| `pyproject.toml` | Добавить `colab` extra (без torch), добавить комментарии к cuda extra |
| `configs/default.yaml` | Добавить `colab_paths` секцию |
| `CLAUDE.md` | Добавить `## Environments` с командами для каждой среды |
| `docs/colab-workflow.md` | **Создать** — главный гайд |

`docs/notes/`, `docs/superpowers/` — не трогаем (продуктовые/ML-стратегические заметки, не dev env).

---

## 1. `pyproject.toml`

### Что добавить

```toml
[project.optional-dependencies]
test = ["pytest>=8.0.0"]
# Локальный GPU (CUDA 13.1). Команда: uv sync --extra cuda
cuda = ["torch>=2.11.0", "torchvision>=0.26.0"]
# Colab: torch/torchvision предустановлены в runtime, ставим только остальное
# Команда в ячейке: pip install -q ultralytics albumentations rapidfuzz shapely
colab = [
    "ultralytics>=8.3.0",
    "albumentations>=2.0.0",
    "pyyaml>=6.0",
    "opencv-python>=4.10.0",
    "Pillow>=11.0.0",
    "rapidfuzz>=3.0.0",
    "shapely>=2.0.0",
]
detectron2 = []
transformers = ["transformers>=4.40.0", "sentencepiece>=0.2.0"]
craft = []
```

`colab` extra — документальная фиксация зависимостей, фактически устанавливается через `pip install -q` в ячейке (не через `uv sync`, чтобы не трогать Colab-преустановленный PyTorch).

Остальное (torch в main deps, `[[tool.uv.index]]`, `[tool.uv.sources]`) — не меняем.

---

## 2. `configs/default.yaml`

Добавить в конец файла:

```yaml
# Colab paths (Google Drive mounted at /content/drive)
colab:
  drive_base: "/content/drive/MyDrive/WaterMeterCV"
  data_base:  "/content/drive/MyDrive/WaterMeterCV/WaterMetricsDATA"
  weights_dir: "/content/drive/MyDrive/WaterMeterCV/weights"
  results_dir: "/content/drive/MyDrive/WaterMeterCV/results"
```

---

## 3. `CLAUDE.md`

Добавить новый раздел `## Environments` между `## Commands` и `## Architecture`:

```markdown
## Environments

### Ноутбук (CPU)
```bash
uv sync
```

### ПК с GPU (CUDA 13.1)
```bash
uv sync --extra cuda
```

### Google Colab
Полный гайд: `docs/colab-workflow.md`

Пакеты, которых нет в Colab runtime (torch/torchvision предустановлены):
```bash
pip install -q ultralytics albumentations rapidfuzz shapely
```

WORKERS=2 в Colab (в отличие от 0 на Windows CUDA).
```

---

## 4. `docs/colab-workflow.md` — новый файл

### 4.1 Приватный репозиторий — PAT (не делать репо публичным)

1. GitHub → Settings → Developer Settings → Personal access tokens → Fine-grained tokens
2. Права: `Contents: Read and write` (чтобы пушить результаты)
3. В Colab: левая панель → 🔑 Secrets → `GITHUB_TOKEN` → Enable for notebook

Использование в ячейке:
```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
clone_url = f"https://{token}@github.com/UrranQx/WaterMeterCV.git"
```

### 4.2 Стандартная setup-ячейка (шаблон для всех ноутбуков)

Текущая setup-ячейка в `yolo_single_stage.ipynb` уже близка к нужному — расширить до:

```python
import sys, os, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB:
    try: from google.colab import drive; IN_COLAB = True
    except ImportError: pass

if IN_COLAB:
    from google.colab import drive, userdata

    # 1. Mount Drive (данные и веса)
    drive.mount("/content/drive")

    # 2. Clone repo (private)
    token = userdata.get("GITHUB_TOKEN", "")
    base = f"https://{token}@github.com" if token else "https://github.com"
    if not Path("/content/WaterMeterCV").exists():
        subprocess.run(["git", "clone", f"{base}/UrranQx/WaterMeterCV.git",
                        "/content/WaterMeterCV"], check=True)

    # 3. Checkout нужную ветку (МЕНЯТЬ ПОД КАЖДЫЙ НОУТБУК)
    BRANCH = "feature/baseline-yolo"
    subprocess.run(["git", "-C", "/content/WaterMeterCV", "checkout", BRANCH], check=True)

    # 4. Установить недостающие пакеты
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "ultralytics", "albumentations", "rapidfuzz", "shapely"], check=True)

    # 5. Пути
    ROOT         = Path("/content/WaterMeterCV")
    DATA_ROOT    = Path("/content/drive/MyDrive/WaterMeterCV/WaterMetricsDATA")
    WEIGHTS_BASE = Path("/content/drive/MyDrive/WaterMeterCV/weights")
    RESULTS_DIR  = Path("/content/drive/MyDrive/WaterMeterCV/results")
    WORKERS = 2
else:
    ROOT = Path("../..").resolve()
    DATA_ROOT    = ROOT / "WaterMetricsDATA"
    WEIGHTS_BASE = ROOT / "models/weights"
    RESULTS_DIR  = ROOT / "results"
    WORKERS = 0  # Windows CUDA stability

sys.path.insert(0, str(ROOT))
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
```

### 4.3 Политика PyTorch по средам

| Среда | Команда | Результат |
|---|---|---|
| Ноутбук (CPU) | `uv sync` | CPU torch с PyPI |
| ПК с GPU (CUDA 13.1) | `uv sync --extra cuda` | CUDA 13.1 torch с pytorch.org/whl/cu131 |
| Google Colab | *ничего* — предустановлен | CUDA torch в Colab runtime |

**Нельзя запускать `uv sync` в Colab** — перепишет предустановленный PyTorch, сломает совместимость с runtime. Только `pip install -q ultralytics albumentations rapidfuzz shapely`.

### 4.4 Ветки — всегда явный checkout

После клона Colab останется на `main`. Нужная ветка:
```bash
!git -C /content/WaterMeterCV checkout feature/baseline-yolo
```

Новая ветка из Colab:
```bash
!git -C /content/WaterMeterCV checkout -b feature/roi-faster-rcnn origin/develop
```

### 4.5 Синхронизация результатов

**Google Drive** (бинарное / большое):
- Colab пишет веса `.pt` прямо в `/content/drive/MyDrive/WaterMeterCV/weights/` — файлы мгновенно появляются в Drive
- Google Drive desktop app на ноутбуке/ПК авто-синкает папку → веса скачиваются на локальную машину автоматически
- Дополнительных команд для синхронизации не нужно — Drive desktop app делает всё сам

**Альтернатива: rclone** (если не хочется Drive desktop app):

```bash
# Установка (Windows)
winget install Rclone.Rclone

# Настройка (один раз): rclone config → gdrive remote

# Скачать веса с Drive на ПК
rclone copy gdrive:WaterMeterCV/weights ./models/weights/

# Загрузить локальные веса → Drive
rclone copy ./models/weights/ gdrive:WaterMeterCV/weights/
```

**Git** (метрики, ноутбуки, код):
```bash
!git -C /content/WaterMeterCV config user.email "you@example.com"
!git -C /content/WaterMeterCV config user.name "Name"
!git -C /content/WaterMeterCV add results/baseline_metrics.json results/baseline_comparison.csv
!git -C /content/WaterMeterCV commit -m "feat: baseline yolo results from Colab"
!git -C /content/WaterMeterCV push \
  https://{token}@github.com/UrranQx/WaterMeterCV.git feature/baseline-yolo
```

**Матрица синхронизации:**

| Откуда → Куда | Механизм |
|---|---|
| Colab → Drive | Запись в `/content/drive/...` (авто) |
| Drive → ПК/ноутбук | Drive desktop app (авто) или `rclone copy gdrive:... ./...` |
| ПК/ноутбук → Drive | Drive desktop app (авто) или `rclone copy ./... gdrive:...` |
| Drive → Colab | Чтение из `/content/drive/...` (авто) |
| Код/метрики | `git push` / `git pull` |

**Что куда:**

| Артефакт | Место | Механизм |
|---|---|---|
| Веса (`.pt`, `.ckpt`) | Drive `/weights/` | Drive desktop app ↔ авто или rclone |
| Метрики (`.json`, `.csv`) | `results/` в репо | `git commit && push` |
| PNG-графики | Drive `/results/` | Drive desktop app → авто на ПК |
| Код и ноутбуки | GitHub | обычный git workflow |

### 4.6 VS Code Colab Extension

Расширение "Google Colaboratory" (VS Code marketplace) подключает VS Code к Colab runtime:
- Файлы редактируются локально в VS Code, выполняются на Colab GPU
- Setup-ячейка запускается как обычно — Drive монтируется, репо клонируется
- **Не решает** проблему данных и пакетов — Drive + PAT по-прежнему нужны

Использование:
1. Extensions → установить "Google Colaboratory"
2. Открыть `.ipynb` → Command Palette → "Colab: Connect to Runtime"
3. Войти в Google аккаунт → выбрать GPU runtime

### 4.7 Структура Drive

```
MyDrive/WaterMeterCV/
├── WaterMetricsDATA/     ← уже загружено
│   ├── waterMeterDataset/
│   └── utility-meter-.../
├── weights/              ← создать, Colab будет сохранять сюда
└── results/              ← создать, для PNG-артефактов
```

---

## Порядок имплементации

- [ ] `pyproject.toml` — добавить `colab` extra с комментарием
- [ ] `configs/default.yaml` — добавить `colab:` секцию
- [ ] `CLAUDE.md` — добавить `## Environments`
- [ ] Создать `docs/colab-workflow.md`

---

## Верификация

1. **Совместимость Python**: Colab показал `Python 3.12.13`. `requires-python = ">=3.13"` будет конфликтовать при `pip install -e .` — но мы это не делаем в Colab, ставим пакеты напрямую. Если понадобится `pip install -e .` — снизить до `>=3.12`.
2. Открыть `Notebooks/01_baseline/yolo_single_stage.ipynb` через VS Code Colab Extension
3. Запустить setup-ячейку → Drive монтируется, репо клонируется на ветку `feature/baseline-yolo`
4. `import torch; torch.cuda.is_available()` → `True`
5. `DATA_ROOT.exists()` → `True` (путь на Drive)
6. Запустить train-ячейку → веса появляются в Drive `/weights/`
7. На локальном ПК: Drive desktop app авто-скачивает веса, `git pull` тянет метрики
