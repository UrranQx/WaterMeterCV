# ROI Detection — Findings & Decisions

## Results (feature/roi-detection, 2026-04-13)

| Method | WM IoU | WM det% | WM inf. | UM IoU | UM det% |
|---|---|---|---|---|---|
| YOLO11n | **0.940** | 100% | 23 ms | 0.00 | 0% |
| U-Net (resnet34) | 0.877 | 100% | **5 ms** | 0.00 | 0% |
| Faster R-CNN (torchvision) | 0.940 | 100% | 405 ms | 0.14 | 17% |

WM = waterMeterDataset (1244 imgs, 870/374 train/test split).  
UM = utility-meter (45 ROI imgs, 6 test) — see below.

## UM Dataset: NOT Suitable for ROI Training

**Do not use utility-meter for ROI detection.** Only 45 of 1552 images contain a ROI annotation (class 10). This is too few for any deep learning model — all three approaches either fail completely (YOLO, U-Net: IoU=0.0) or give near-random results (Faster R-CNN: 17% detection, IoU=0.14). This decision is final; do not revisit it.

ROI detection experiments must use **waterMeterDataset only** (1244 images, all with `roi_polygon`).

## Downstream OCR Pipeline — Two Approaches to Evaluate

Both bbox and polygon outputs lead to different pre-processing pipelines for the OCR stage:

**bbox-path** (YOLO, Faster R-CNN output):
- Axis-aligned rectangular crop → likely contains background at corners when meter is rotated
- Requires rotation correction: find digit line via YOLO digit detections, compute angle, rotate + crop

**polygon-path** (U-Net mask → contour → quadrilateral):
- `cv2.minAreaRect` / `approxPolyDP` on the segmentation mask → rotated bounding quadrilateral
- `cv2.getPerspectiveTransform` to unwarp reading window → clean output regardless of orientation
- Polygon shape encodes rotation directly; no separate heuristic needed
- Ground truth `roi_polygon` in WM dataset supports polygon-path evaluation natively

**Decision**: implement and compare both paths in OCR notebooks (03_*).

## Orientation / Upside-Down Handling

**Chosen approach (implement first):** after OCR, try both 0° and 180° reads; keep the result that is a valid integer in the expected range. Simple and robust for a water meter whose value is bounded.

**Future work (approach 2):** train a lightweight binary orientation classifier ("normal" / "upside-down") before the OCR stage. Revisit if approach 1 produces too many false positives on real-world data.
