"""Quick visual sanity-check for warp_roi_polygon and crop_roi_from_detection.

Loads N WM samples sorted by |rotation angle| (most-rotated first) and saves:
  col 0 — original image + polygon overlay + estimated angle
  col 1 — wm_polygon crop  (perspective warp, GT polygon)
  col 2 — wm_bbox crop     (YOLO detected bbox + projection-profile rotation)

Usage:
    uv run python scripts/visualize_ocr_crops.py
    uv run python scripts/visualize_ocr_crops.py --n 16 --out results/crop_check.png
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.data.unified_loader import load_water_meter_dataset_split
from models.data.ocr_dataset import (
    warp_roi_polygon,
    crop_roi_from_detection,
    estimate_roi_rotation,
    OUT_H, OUT_W,
)


def _make_roi_detector(weights_path: Path):
    """Return callable img_bgr -> (cx,cy,w,h)|None using YOLO weights."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))

        def detect(img_bgr: np.ndarray):
            result = model.predict(img_bgr, verbose=False, conf=0.001)[0]
            if result.boxes is None or len(result.boxes) == 0:
                return None
            best = int(result.boxes.conf.argmax().item())
            box = result.boxes.xywhn[best]
            return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

        return detect
    except Exception as exc:
        print(f"Warning: could not load YOLO ({exc}). Falling back to GT bbox.")
        return None


def _gt_bbox_detector(polygon, img_shape):
    """GT-polygon based fallback bbox."""
    from models.data.roi_dataset import polygon_to_bbox
    return polygon_to_bbox(polygon)


def draw_polygon_overlay(img_bgr: np.ndarray, polygon, angle: float) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]
    pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in polygon], dtype=np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for pt in pts:
        cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)
    cv2.putText(vis, f"{angle:+.1f}deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thumb(img, max_w=400, max_h=300):
    h, w = img.shape[:2]
    scale = min(max_w/w, max_h/h)
    return cv2.resize(img, (int(w*scale), int(h*scale)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--out", type=str,
                        default=str(ROOT / "results" / "ocr_crop_check.png"))
    parser.add_argument("--wm-path", type=str,
                        default=str(ROOT / "WaterMetricsDATA" / "waterMeterDataset" / "WaterMeters"))
    parser.add_argument("--yolo-weights", type=str,
                        default=str(ROOT / "models" / "weights" / "roi_yolo"
                                    / "wm_yolo_roi_20260412_230832" / "weights" / "best.pt"))
    args = parser.parse_args()

    wm_path = Path(args.wm_path)
    if not wm_path.exists():
        print(f"ERROR: WM dataset not found at {wm_path}")
        sys.exit(1)

    yolo_weights = Path(args.yolo_weights)
    roi_detector = _make_roi_detector(yolo_weights) if yolo_weights.exists() else None
    using_yolo = roi_detector is not None
    bbox_label = "BBox (YOLO + projection)" if using_yolo else "BBox (GT + projection)"

    _, test_samples = load_water_meter_dataset_split(wm_path, train_ratio=0.7, seed=42)
    valid = [s for s in test_samples if s.roi_polygon is not None and s.value is not None]

    annotated = []
    for s in valid:
        img = cv2.imread(str(s.image_path))
        if img is None:
            continue
        angle = estimate_roi_rotation(s.roi_polygon, img.shape)
        annotated.append((abs(angle), angle, img, s))

    annotated.sort(key=lambda t: t[0], reverse=True)
    annotated = annotated[:args.n]

    if not annotated:
        print("No valid samples found.")
        sys.exit(1)

    n = len(annotated)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5*n))
    if n == 1:
        axes = axes[None, :]

    axes[0, 0].set_title("Original + polygon (GT angle)", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Polygon-warp crop (GT polygon)", fontsize=12, fontweight="bold")
    axes[0, 2].set_title(bbox_label, fontsize=12, fontweight="bold")

    for row, (abs_a, angle, img, s) in enumerate(annotated):
        # col 0: original
        overlay = draw_polygon_overlay(img, s.roi_polygon, angle)
        axes[row, 0].imshow(bgr2rgb(thumb(overlay)))
        axes[row, 0].set_ylabel(
            f"GT={int(s.value)}  |a|={abs_a:.1f}",
            fontsize=9, rotation=0, labelpad=110, va="center",
        )

        # col 1: polygon warp
        poly_crop = warp_roi_polygon(img, s.roi_polygon, out_h=OUT_H, out_w=OUT_W)
        axes[row, 1].imshow(bgr2rgb(poly_crop))

        # col 2: bbox + projection rotation
        if roi_detector is not None:
            bbox = roi_detector(img)
        else:
            from models.data.roi_dataset import polygon_to_bbox
            bbox = polygon_to_bbox(s.roi_polygon)

        if bbox is not None:
            bbox_crop = crop_roi_from_detection(img, bbox, out_h=OUT_H, out_w=OUT_W)
            axes[row, 2].imshow(bgr2rgb(bbox_crop))
        else:
            axes[row, 2].text(0.5, 0.5, "No detection", ha="center", va="center",
                              transform=axes[row, 2].transAxes, fontsize=10)

        for col in range(3):
            axes[row, col].axis("off")

    plt.suptitle(
        f"OCR crop sanity-check — {n} most-rotated test samples",
        fontsize=14, y=1.001,
    )
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
