# ROI Detection — Findings & Decisions

## Results (feature/roi-detection, 2026-04-14)

| Method | WM IoU | WM det% | WM inf. | UM IoU | UM det% | Notes |
|---|---|---|---|---|---|---|
| **YOLO11n** | **0.940** | 100% | **23 ms** | 0.00 | 0% | Best IoU + fastest |
| U-Net (resnet34) | 0.877 | 100% | **5 ms** | 0.00 | 0% | Fastest inference, mask output |
| Faster R-CNN (torchvision, 15 ep) | 0.924 | 100% | 208 ms | 0.00 | 0% | Heaviest, no gain over YOLO |
| Faster R-CNN (Detectron2, 5000 it) | 0.940 | 100% | 405 ms | 0.14 | 17% | Reference run, not reproducible |

WM = waterMeterDataset (1244 imgs, 870/374 train/test, seed=42).  
UM = utility-meter (45 ROI imgs, 6 test) — see below.

## UM Dataset: NOT Suitable for ROI Training

**Do not use utility-meter for ROI detection.** Only 45 of 1552 images contain a ROI annotation (class 10). This is too few for any deep learning model — all approaches fail completely on UM. This decision is final; do not revisit it.

ROI detection experiments must use **waterMeterDataset only** (1244 images, all with `roi_polygon`).

## Winner: YOLO11n

Best overall: **IoU 0.94, 100% detection, 23 ms inference**.  
U-Net is a valid alternative when mask output is preferred over bbox (see pipeline section below).  
Faster R-CNN is not recommended — no accuracy gain over YOLO at 9× the inference cost.

## Downstream OCR Pipeline — Two Approaches to Evaluate

Both bbox and polygon outputs lead to different pre-processing for OCR:

**bbox-path** (YOLO, Faster R-CNN output):
- Axis-aligned rectangular crop → background at corners when meter is rotated
- Requires rotation correction: detect digit line via YOLO digit detections, compute angle, rotate + crop

**polygon-path** (U-Net mask → contour → quadrilateral):
- `cv2.minAreaRect` / `approxPolyDP` on segmentation mask → rotated bounding quadrilateral
- `cv2.getPerspectiveTransform` to unwarp reading window → clean output regardless of orientation
- Polygon shape encodes rotation directly; no separate heuristic needed
- Ground truth `roi_polygon` in WM dataset supports native polygon-path evaluation

**Decision**: implement and compare both paths in OCR notebooks (`Notebooks/03_*`).

## Orientation / Upside-Down Handling

**Chosen approach (implement first):** after OCR, try both 0° and 180° reads; keep the result that is a valid integer in the expected range. Simple and robust for a water meter whose value is bounded.

**Future work (approach 2):** train a lightweight binary orientation classifier ("normal" / "upside-down") before the OCR stage. Revisit if approach 1 produces too many false positives.
