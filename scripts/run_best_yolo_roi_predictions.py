#!/usr/bin/env python3
"""Build YOLO WM ROI predictions figure from the latest comparison CSV run."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.data.roi_dataset import polygon_to_bbox
from models.data.unified_loader import load_water_meter_dataset_split
from models.metrics.evaluation import compute_iou_bbox


RUN_DIR_RE = re.compile(r"^(um|wm)_yolo_roi_(\d{8}_\d{6})$")


@dataclass
class RoiSample:
    image_path: Path
    gt_bbox: tuple[float, float, float, float]  # normalized (cx, cy, w, h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build YOLO WM ROI prediction figure from latest yolo_roi "
            "row in results/roi_comparison.csv"
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "results" / "roi_comparison.csv",
        help="Path to roi_comparison.csv",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=ROOT / "models" / "weights" / "roi_yolo",
        help="Directory with wm_yolo_roi_* runs",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "WaterMetricsDATA",
        help="Path to WaterMetricsDATA",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "roi_yolo_predictions.png",
        help="Output image path",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.001,
        help="Confidence threshold for YOLO predict()",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for water meter split",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio for water meter split",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=12,
        help="Number of WM test images to draw",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="yolo_roi",
        help="Method name in comparison CSV",
    )
    return parser.parse_args()


def parse_run_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M")


def find_latest_method_row(csv_path: Path, method: str) -> tuple[datetime, dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    latest_dt = None
    latest_row = None
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("method") != method:
                continue
            try:
                row_dt = parse_run_date(row["run_date"])
            except Exception as exc:
                raise ValueError(
                    f"Invalid run_date '{row.get('run_date')}' in {csv_path}"
                ) from exc
            if latest_dt is None or row_dt > latest_dt:
                latest_dt = row_dt
                latest_row = row

    if latest_dt is None or latest_row is None:
        raise ValueError(f"No rows with method='{method}' in {csv_path}")

    return latest_dt, latest_row


def list_checkpoints(weights_dir: Path, prefix: str) -> list[tuple[datetime, Path]]:
    checkpoints = []
    for path in weights_dir.glob(f"{prefix}_yolo_roi_*"):
        if not path.is_dir():
            continue
        match = RUN_DIR_RE.match(path.name)
        if not match:
            continue
        best_weights = path / "weights" / "best.pt"
        if not best_weights.exists():
            continue
        ts = datetime.strptime(match.group(2), "%Y%m%d_%H%M%S")
        checkpoints.append((ts, best_weights))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def pick_checkpoint(weights_dir: Path, prefix: str, target_dt: datetime) -> tuple[datetime, Path]:
    checkpoints = list_checkpoints(weights_dir, prefix)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found for prefix='{prefix}' in {weights_dir}"
        )

    best = [item for item in checkpoints if item[0] <= target_dt]
    if best:
        return best[-1]
    return checkpoints[-1]


def norm_to_xyxy_abs(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    x2 = int((cx + bw / 2) * width)
    y2 = int((cy + bh / 2) * height)
    return x1, y1, x2, y2


def load_wm_test_samples(wm_path: Path, train_ratio: float, seed: int) -> list[RoiSample]:
    _, wm_test = load_water_meter_dataset_split(wm_path, train_ratio=train_ratio, seed=seed)
    samples = [
        RoiSample(sample.image_path, polygon_to_bbox(sample.roi_polygon))
        for sample in wm_test
        if sample.roi_polygon is not None
    ]
    samples.sort(key=lambda s: s.image_path.name)
    return samples


def predict_bbox(
    model: YOLO,
    image_path: Path,
    conf_thresh: float,
) -> tuple[float, float, float, float] | None:
    result = model.predict(str(image_path), verbose=False, conf=conf_thresh)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_idx = int(result.boxes.conf.argmax().item())
    box = result.boxes.xywhn[best_idx]
    return (
        float(box[0].item()),
        float(box[1].item()),
        float(box[2].item()),
        float(box[3].item()),
    )


def draw_predictions_grid(
    axes_flat,
    samples: list[RoiSample],
    model: YOLO,
    conf_thresh: float,
) -> None:
    for i, ax in enumerate(axes_flat):
        if i >= len(samples):
            ax.axis("off")
            continue

        sample = samples[i]
        image_bgr = cv2.imread(str(sample.image_path))
        if image_bgr is None:
            ax.axis("off")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        pred_bbox = predict_bbox(model, sample.image_path, conf_thresh)
        iou_val = compute_iou_bbox(pred_bbox, sample.gt_bbox) if pred_bbox is not None else 0.0

        gx1, gy1, gx2, gy2 = norm_to_xyxy_abs(sample.gt_bbox, width, height)
        cv2.rectangle(image_rgb, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)

        if pred_bbox is not None:
            px1, py1, px2, py2 = norm_to_xyxy_abs(pred_bbox, width, height)
            cv2.rectangle(image_rgb, (px1, py1), (px2, py2), (200, 0, 0), 2)

        ax.imshow(image_rgb)
        ax.set_title(f"WM #{i + 1} IoU={iou_val:.2f}", fontsize=10)
        ax.axis("off")


def main() -> int:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1")

    run_dt, run_row = find_latest_method_row(args.csv, args.method)
    print(f"Selected run: {run_row['run_date']} ({args.method})")

    wm_dt, wm_weights = pick_checkpoint(args.weights_dir, "wm", run_dt)
    print(f"WM checkpoint: {wm_weights} ({wm_dt.strftime('%Y-%m-%d %H:%M:%S')})")

    wm_path = args.data_root / "waterMeterDataset" / "WaterMeters"

    wm_samples_all = load_wm_test_samples(wm_path, train_ratio=args.train_ratio, seed=args.seed)
    if not wm_samples_all:
        raise ValueError("No WM test samples found with ROI")
    wm_samples = wm_samples_all[: args.num_images]
    print(f"WM samples: total={len(wm_samples_all)}, plotted={len(wm_samples)}")

    wm_model = YOLO(str(wm_weights))

    n_images = len(wm_samples)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes, dtype=object).reshape(-1)

    draw_predictions_grid(axes_flat, wm_samples, wm_model, args.conf_thresh)

    plt.suptitle("YOLO ROI (WM only) - Green=GT, Red=Pred", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()

    print(f"Saved prediction figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
