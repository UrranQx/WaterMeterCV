#!/usr/bin/env python3
"""Build Faster R-CNN WM ROI predictions figure from the latest comparison CSV run."""

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
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.data.roi_dataset import polygon_to_bbox
from models.data.unified_loader import load_water_meter_dataset_split
from models.metrics.evaluation import compute_iou_bbox


WEIGHT_RE = re.compile(r"^(um|wm)_best_frcnn_(\d{8}_\d{6})\.pth$")


@dataclass
class RoiSample:
    image_path: Path
    gt_bbox: tuple[float, float, float, float]  # normalized (cx, cy, w, h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Faster R-CNN WM ROI prediction figure from latest faster_rcnn_torchvision "
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
        default=ROOT / "models" / "weights" / "roi_faster_rcnn",
        help="Directory with wm_best_frcnn_* checkpoints",
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
        default=ROOT / "results" / "roi_faster_rcnn_predictions.png",
        help="Output image path",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Score threshold for predicted boxes",
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
        default="faster_rcnn_torchvision",
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
    for path in weights_dir.glob(f"{prefix}_best_frcnn_*.pth"):
        match = WEIGHT_RE.match(path.name)
        if not match:
            continue
        ts = datetime.strptime(match.group(2), "%Y%m%d_%H%M%S")
        checkpoints.append((ts, path))
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


def load_wm_test_samples(wm_path: Path, train_ratio: float, seed: int) -> list[RoiSample]:
    _, wm_test = load_water_meter_dataset_split(wm_path, train_ratio=train_ratio, seed=seed)
    samples = [
        RoiSample(sample.image_path, polygon_to_bbox(sample.roi_polygon))
        for sample in wm_test
        if sample.roi_polygon is not None
    ]
    samples.sort(key=lambda s: s.image_path.name)
    return samples


def build_model(num_classes: int = 2) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(weights_path: Path, device: str) -> torch.nn.Module:
    model = build_model().to(device)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def norm_to_xyxy_abs(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * width)
    y1 = int((cy - bh / 2) * height)
    x2 = int((cx + bw / 2) * width)
    y2 = int((cy + bh / 2) * height)
    return x1, y1, x2, y2


def xyxy_abs_to_norm(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


@torch.no_grad()
def predict_bbox(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: str,
    score_thresh: float,
) -> tuple[float, float, float, float] | None:
    tensor = TF.to_tensor(image_rgb).unsqueeze(0).to(device)
    output = model(tensor)[0]

    boxes = output.get("boxes")
    scores = output.get("scores")
    if boxes is None or scores is None or len(boxes) == 0:
        return None

    keep = torch.where(scores >= score_thresh)[0]
    if len(keep) == 0:
        return None

    best_idx = keep[scores[keep].argmax()].item()
    best_box = boxes[best_idx].detach().cpu().tolist()

    height, width = image_rgb.shape[:2]
    return xyxy_abs_to_norm(best_box, width, height)


def draw_predictions_grid(
    axes_flat,
    samples: list[RoiSample],
    model: torch.nn.Module,
    device: str,
    score_thresh: float,
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

        pred_bbox = predict_bbox(model, image_rgb, device, score_thresh)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run_dt, run_row = find_latest_method_row(args.csv, args.method)
    print(f"Selected run: {run_row['run_date']} ({args.method})")

    wm_dt, wm_weights = pick_checkpoint(args.weights_dir, "wm", run_dt)
    print(f"WM checkpoint: {wm_weights.name} ({wm_dt.strftime('%Y-%m-%d %H:%M:%S')})")

    wm_path = args.data_root / "waterMeterDataset" / "WaterMeters"

    wm_samples_all = load_wm_test_samples(wm_path, train_ratio=args.train_ratio, seed=args.seed)
    if not wm_samples_all:
        raise ValueError("No WM test samples found with ROI")
    wm_samples = wm_samples_all[: args.num_images]
    print(f"WM samples: total={len(wm_samples_all)}, plotted={len(wm_samples)}")

    wm_model = load_model(wm_weights, device)

    n_images = len(wm_samples)
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes, dtype=object).reshape(-1)

    draw_predictions_grid(axes_flat, wm_samples, wm_model, device, args.score_thresh)

    plt.suptitle("Faster R-CNN ROI (WM only) - Green=GT, Red=Pred", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()

    print(f"Saved prediction figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
