# Baseline YOLO Single-Stage — Design Spec

**Date:** 2026-04-10
**Branch:** feature/baseline-yolo
**Notebook:** `Notebooks/01_baseline/yolo_single_stage.ipynb`

---

## Goal

Establish a baseline for digit reading using YOLOv11 in a single-stage detection approach. Detect all digits directly in the full image, sort left-to-right, and reconstruct the meter reading. Serves as the reference point for all subsequent ROI + OCR experiments.

---

## Data

**Training & validation:** `utility-meter...yolov11` dataset (train/val splits, YOLO format)
- 1555 train / 194 valid / 194 test images
- 14 classes: digits `0–9`, `Reading Digit`, `black`, `red`, `white`
- `data.yaml` uses relative paths — no patching needed

**Evaluation:**
- **Primary:** utility-meter test split (194 images)
- **Secondary (cross-dataset):** waterMeterDataset (1244 images) — inference only, per-digit accuracy only (no full-string, no decimal point handling — deferred)

---

## Model

**Entry point:** `yolo11n.pt` (nano, ~2.6M params)

`MODEL_SIZE` is a single config variable at the top of the notebook. Supported values and their trade-offs:

| MODEL_SIZE | Params | Expected mAP50 | Inference (GPU) | Notes |
|---|---|---|---|---|
| `yolo11n` | 2.6M | baseline | ~15 ms/img | Start here |
| `yolo11s` | 9.6M | +5–10% | ~45 ms/img | Upgrade for accuracy |
| `yolo11m` | 20M | +10–15% | ~120 ms/img | Max quality, may need batch_size↓ |

**Iteration strategy:** Run nano first to validate pipeline. Decide next step based on combined score.

---

## Training Config

| Parameter | Value |
|---|---|
| `epochs` | 50 |
| `batch_size` | 16 (reduce to 8–12 for yolo11m if OOM) |
| `img_size` | 640 |
| `device` | `cuda` |
| `patience` | 10 (early stopping) |
| `save` | True |

Weights saved to: `models/weights/baseline_yolo/<run_name>/weights/best.pt`

---

## Metrics

### Primary evaluation (utility-meter test set)

| Metric | Source | Description |
|---|---|---|
| **mAP50** | YOLO built-in `.val()` | Detection quality, IoU threshold 0.5 |
| **Full-string accuracy** | `models/metrics/evaluation.py` | % images where all digits are perfectly reconstructed |
| **Per-digit accuracy** | `models/metrics/evaluation.py` | Per-character match rate |
| **CER** | `models/metrics/evaluation.py` | Character Error Rate (Levenshtein / length) |
| **Avg inference time** | manual timing | Total test set time / num images, ms/image |

**Combined score (primary ranking metric):**
```
Combined Score = 0.8 * mAP50 + 0.2 * Full-string accuracy
```
mAP50 chosen over mAP50-95 because the task requires "found the digit or not", not sub-pixel bbox precision.

### Secondary evaluation (waterMeterDataset)

- Metrics: **per-digit accuracy only**
- Full-string accuracy skipped: WM ground truth has decimal values (e.g. `78.677`), YOLO predicts only digit sequence (`78677`) — decimal point handling deferred
- Results logged separately, not included in combined score

---

## Notebook Structure

| Cell | Type | Content |
|---|---|---|
| 1 | Code | Setup — imports, ROOT resolution, GPU check |
| 2 | Code | **Config** — `MODEL_SIZE`, paths, training params. All tunable params here |
| 3 | Code | Dataset verification — load data.yaml, print class list, count splits |
| 4 | Code | Training — `model.train(...)`, save best weights |
| 5 | Code | Primary evaluation — YOLO `.val()` for mAP50; manual loop for FSA, per-digit, CER |
| 6 | Code | Inference time — full test set timing, ms/image |
| 7 | Code | Visualization — 8 test images with predicted bboxes + reconstructed value vs GT |
| 8 | Code | Cross-dataset eval (WM) — inference only, per-digit accuracy |
| 9 | Markdown | Conclusions — filled manually after run |

---

## Value Reconstruction

Predicted digits from YOLO bboxes are sorted left-to-right by center-x coordinate, then concatenated into a string. All digit classes (0–9) included; `Reading Digit`, `black`, `red`, `white` classes are filtered out.

---

## Output Files

| File | Content |
|---|---|
| `models/weights/baseline_yolo/.../best.pt` | Trained weights |
| `results/baseline_metrics.json` | All metrics for primary + secondary eval |
| `results/baseline_predictions.png` | Grid of 8 predictions |
| `results/baseline_comparison.csv` | Created/updated on each run — one row per model size |

`baseline_comparison.csv` schema:
```
model_size, mAP50, mAP50_95, full_string_acc, per_digit_acc, CER, combined_score, inference_ms, run_date
```

---

## Iteration Plan

After nano baseline is established:

1. **Want better accuracy?** → Set `MODEL_SIZE = "yolo11s"` (or `"yolo11m"`), re-run cells 4–8, new row added to `baseline_comparison.csv`
2. **Want faster inference?** → Stay on nano or explore quantization (separate task, out of scope here)
3. **Decision gate:** If combined score satisfies requirements, proceed to ROI experiments. If not, upgrade model size.

---

## Out of Scope

- Decimal point detection/reconstruction (deferred)
- Training on waterMeterDataset (no digit-level annotations)
- Model quantization / TensorRT optimization
- Hyperparameter tuning beyond config cell
