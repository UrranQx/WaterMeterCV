# ROI Detection Notebooks — Design Spec

## Goal

Implement 3 ROI detection notebooks (`Notebooks/02_roi_detection/`) that locate the "reading window" bounding box on water/utility meter images. Each notebook trains and evaluates on two datasets independently (no merging), producing IoU metrics for comparison.

## Datasets

### utility-meter (UM)
- **Format**: YOLO (class 10 = ROI) and COCO (category 11 = "Reading Digit")
- **ROI coverage**: 45/1552 train, 6/194 valid, 6/193 test images have ROI annotations
- **Splits**: pre-existing train/valid/test
- **Note**: Very sparse ROI annotations. Heavy augmentation recommended.

### waterMeterDataset (WM)
- **Format**: `data.csv` with normalized polygon ROI (`location` column)
- **ROI coverage**: all ~1244 images have polygon ROI
- **Splits**: none pre-existing. Create 70/30 train/test split with `seed=42` for reproducibility.
- **Polygon → bbox**: convert `[(x,y), ...]` polygon to `(cx, cy, w, h)` normalized bbox via min/max.

### No merging
Each notebook runs two independent experiments: train+eval on UM, then train+eval on WM. Results compared side-by-side.

---

## Shared Infrastructure

### `models/data/unified_loader.py` — additions

```python
def load_water_meter_dataset_split(
    root: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[UnifiedSample], list[UnifiedSample]]:
    """Deterministic train/test split of waterMeterDataset."""
```

Shuffles all samples with fixed seed, splits at `int(len * train_ratio)`.

### `models/data/roi_dataset.py` — new file

Shared helpers for all 3 notebooks:

| Function | Purpose |
|----------|---------|
| `polygon_to_bbox(polygon)` | `[(x,y), ...]` normalized → `(cx, cy, w, h)` normalized |
| `filter_utility_meter_roi_samples(dataset_path, split)` | Load YOLO labels, return only images that have class 10, with `(image_path, roi_bbox)` |
| `prepare_yolo_roi_dataset(src_path, dst_path)` | Create single-class YOLO dataset on disk: filter class 10 → class 0, copy images, write `data.yaml` |
| `prepare_coco_roi_dicts(coco_root, split)` | Return Detectron2-format dataset dicts filtered to Reading Digit only |
| `prepare_wm_coco_roi_dicts(samples)` | Convert WM samples with polygon ROI to Detectron2-format dicts |
| `prepare_wm_yolo_roi_dataset(samples, dst_path)` | Convert WM samples to YOLO single-class label files on disk |
| `ROISegmentationDataset(Dataset)` | PyTorch Dataset: image + binary mask from polygon/bbox, with resize and optional albumentations transform |

### `models/metrics/evaluation.py` — no changes

`compute_iou_bbox` and `compute_iou_polygon` already exist.

---

## Notebook Cell Structure

All 3 notebooks share the same layout (21 cells):

| # | Type | Section | Content |
|---|------|---------|---------|
| 0 | md | Title | `# 02 — ROI: <Method Name>` |
| 1 | code | Setup | IN_COLAB pattern (branch = `feature/roi-detection`), imports, paths |
| 2 | code | Config | Model params, EPOCHS, BATCH_SIZE, DEVICE, dataset paths |
| 3 | code | Data Prep | Filter ROI from UM; split WM 70/30 |
| 4 | md | Header | `## Experiment 1: utility-meter dataset` |
| 5 | code | Verify UM | Count images with ROI per split, print stats |
| 6 | md | Header | `## Training (utility-meter)` |
| 7 | code | Train UM | Model-specific training |
| 8 | md | Header | `## Evaluation (utility-meter)` |
| 9 | code | Eval UM | Predict on UM test, mean IoU, detection rate, inference time |
| 10 | md | Header | `## Experiment 2: waterMeterDataset` |
| 11 | code | Prep WM | Dataset-specific preparation (COCO dicts / YOLO files / seg masks) |
| 12 | code | Train WM | Same architecture, train on WM train (70%) |
| 13 | code | Eval WM | Predict on WM test (30%), mean IoU, detection rate, inference time |
| 14 | md | Header | `## Predictions` |
| 15 | code | Viz | 2x4 grid: 4 UM test + 4 WM test. GT green, pred red. |
| 16 | md | Header | `## Comparison` |
| 17 | code | Compare | Side-by-side table of UM vs WM metrics |
| 18 | md | Header | `## Save Results` |
| 19 | code | Save | JSON + append to `results/roi_comparison.csv` |
| 20 | md | Conclusions | Fill after running |

---

## Per-Notebook Specifics

### 1. `faster_rcnn.ipynb` — Detectron2 Faster R-CNN + FPN

**Colab deps**: `pip install 'git+https://github.com/facebookresearch/detectron2.git'`

**Model**: `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` from Detectron2 model zoo.
- `ROI_HEADS.NUM_CLASSES = 1`
- `SOLVER.IMS_PER_BATCH = 4`
- `SOLVER.BASE_LR = 0.0025`

**UM experiment**:
- Data: `prepare_coco_roi_dicts(coco_root, split)` → register with `DatasetCatalog`
- Training: `DefaultTrainer`, `MAX_ITER = 3000`, `STEPS = (2000, 2500)`
- Eval: `DefaultPredictor` → top-1 box by confidence → normalize to cxcywh → `compute_iou_bbox`

**WM experiment**:
- Data: `prepare_wm_coco_roi_dicts(train_samples)` → register as new dataset
- Training: `MAX_ITER = 5000` (more data), `STEPS = (3000, 4000)`
- Eval: same as UM, polygons converted to bbox for IoU

**Weights**: `models/weights/roi_faster_rcnn/` (local), `WEIGHTS_BASE/roi_faster_rcnn/` (Colab/Drive)

### 2. `yolo_roi.ipynb` — YOLOv11 single-class

**Model**: `yolo11n.pt` (nano). `MODEL_SIZE` configurable at top.

**UM experiment**:
- Data: `prepare_yolo_roi_dataset(src, dst)` creates `_roi_only_yolo/` with class 0 = ROI
- Training: `model.train(data=..., epochs=50, patience=10, imgsz=640)`
- Eval: `model.predict()` → top-1 box → normalize → `compute_iou_bbox`

**WM experiment**:
- Data: `prepare_wm_yolo_roi_dataset(train_samples, dst)` writes YOLO labels from polygons
- Training: same hyperparams
- Eval: same pipeline

**Weights**: `models/weights/roi_yolo/`

### 3. `segmentation_unet.ipynb` — U-Net binary segmentation

**Colab deps**: `pip install segmentation-models-pytorch`

**Model**: `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)`
- Loss: `BCEWithLogitsLoss + DiceLoss`
- Optimizer: Adam, lr=1e-4
- 30 epochs, img_size=256

**UM experiment**:
- Data: `ROISegmentationDataset` — binary masks from `roi_bbox` (filled rectangle)
- Training: custom PyTorch loop (no YOLO/Detectron2 trainer)
- Eval: sigmoid → threshold 0.5 → largest connected component → bounding box → `compute_iou_bbox`. Also report pixel-level mask IoU as secondary metric.

**WM experiment**:
- Data: `ROISegmentationDataset` — binary masks from `roi_polygon` (filled polygon). Can also use native `mask_path` from waterMeterDataset if available.
- Training & eval: same as UM

**Weights**: `models/weights/roi_unet/`

---

## Results Format

### Per-notebook JSON (`results/roi_<method>_metrics.json`)

```json
{
  "method": "faster_rcnn",
  "utility_meter": {
    "mean_iou": 0.0,
    "detection_rate": 0.0,
    "avg_inference_ms": 0.0,
    "n_train": 45,
    "n_test": 6
  },
  "water_meter": {
    "mean_iou": 0.0,
    "detection_rate": 0.0,
    "avg_inference_ms": 0.0,
    "n_train": 870,
    "n_test": 374
  },
  "config": {
    "epochs_or_iters": 0,
    "batch_size": 0,
    "img_size": 0,
    "model_variant": ""
  },
  "run_date": ""
}
```

### Shared CSV (`results/roi_comparison.csv`)

Columns: `method, um_mean_iou, um_detection_rate, um_inference_ms, wm_mean_iou, wm_detection_rate, wm_inference_ms, run_date`

Each notebook appends one row.

---

## Git & Branch

All 3 notebooks on `feature/roi-detection` branch (already created).

Commit sequence:
1. `feat: ROI data helpers — split, polygon→bbox, dataset prep`
2. `feat: ROI notebook — Faster R-CNN (Detectron2)`
3. `feat: ROI notebook — YOLO single-class`
4. `feat: ROI notebook — U-Net segmentation`

---

## Colab Compatibility

- IN_COLAB setup cell with `BRANCH = "feature/roi-detection"`
- Detectron2: installed via pip in setup cell (only in Colab)
- segmentation-models-pytorch: installed via pip in setup cell (only in Colab)
- WORKERS: 2 in Colab, 0 on Windows
- Weights written to Drive paths in Colab, local `models/weights/` otherwise
- `torch.cuda.empty_cache()` before training
