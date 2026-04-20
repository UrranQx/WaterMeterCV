# NOTICE

WaterMeterCV — учебный CV-пайплайн для чтения цифр с водосчётчиков.
Copyright (C) 2026 WaterMeterCV authors.

Этот файл перечисляет лицензии сторонних датасетов, моделей и библиотек, на
которые опирается репозиторий. Полный текст лицензии самого проекта — в
`LICENSE` (AGPL-3.0).

---

## Project license

Исходный код в этом репозитории и fine-tuned веса моделей, которые мы
распространяем вместе с ним, лицензированы на условиях **GNU Affero General
Public License v3.0 (AGPL-3.0)**.

Сервис `watermetercv-serve` взаимодействует с пользователями по сети
(HTTP). §13 AGPL обязывает операторов такого сервиса предоставлять
пользователям исходный код соответствующей версии. Ссылка на репозиторий в
`/info` или в README развёрнутого сервиса — минимальный способ выполнить это
требование.

---

## Third-party datasets

### Water Meters Dataset (Kaggle)

- Автор: Kucev Roman (Kaggle: `tapakah68`)
- URL: <https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset>
- Лицензия: **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
  International (CC BY-NC-ND 4.0)**
- Где используется: обучение ROI-детектора
  (`models/weights/roi_yolo/wm_yolo_roi_20260412_230832`).

**Важное следствие лицензии CC BY-NC-ND 4.0:**

- **NonCommercial.** Датасет и любые производные от него (включая модели,
  обученные на нём) не могут использоваться в коммерческих целях.
- **NoDerivatives.** Распространение модифицированных копий не разрешено.
  Интерпретация fine-tuned ML-моделей как "производных" от датасета —
  консервативная, но доминирующая.

Этот репозиторий — **учебный проект**, не предназначен для коммерческого
использования. Веса ROI-модели, запечённые в Docker-образы
(`watermetercv:cpu`, `watermetercv:gpu`), наследуют ограничения этой
лицензии. Любой, кто захочет использовать сервис коммерчески, обязан:

1. получить отдельный лицензионный грант от авторов датасета; либо
2. заменить ROI-веса на веса, обученные на данных с подходящей лицензией
   (например, переобучить на Roboflow-датасете ниже).

### Utility Meter Reading Dataset (Roboflow Universe)

- Воркспейс: `watermeter-jvlgr`
- URL: <https://universe.roboflow.com/watermeter-jvlgr/utility-meter-reading-dataset-for-automatic-reading-yolo>
- Лицензия: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Где используется: обучение OCR-детектора
  (`models/weights/baseline_yolo/yolo11m_20260414_194809`).

CC BY 4.0 разрешает коммерческое и некоммерческое использование при условии
указания авторства (указано в этом файле).

---

## Third-party models and code

### Ultralytics YOLO (YOLO11n / YOLO11m)

- URL: <https://github.com/ultralytics/ultralytics>
- Лицензия: **GNU Affero General Public License v3.0 (AGPL-3.0)**
- Где используется: training и inference обеих моделей (ROI + OCR).

AGPL-3.0 распространяется на любое производное программное обеспечение,
включающее Ultralytics YOLO как зависимость. Именно поэтому WaterMeterCV
целиком лицензирован под AGPL-3.0 (см. выше).

---

## Third-party libraries (runtime dependencies)

Рантайм зависит от следующих open-source библиотек; они распространяются
под своими лицензиями, подробности — в соответствующих upstream-репозиториях.

| Библиотека | Лицензия |
|---|---|
| PyTorch (`torch`, `torchvision`) | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Pillow | HPND |
| Shapely | BSD-3-Clause |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| Pydantic | MIT |
| python-multipart | Apache-2.0 |

Все перечисленные библиотеки имеют permissive или copyleft-compatible
лицензии; конфликтов с AGPL-3.0 нет.
