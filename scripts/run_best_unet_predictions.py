#!/usr/bin/env python3
"""Run U-Net WM ROI prediction visualization using the latest unet_segmentation run."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.data.unified_loader import load_water_meter_dataset_split


WEIGHT_RE = re.compile(r"^(um|wm)_unet_(\d{8}_\d{6})_best\.pt$")


class ROISegmentationDataset(Dataset):
    """Binary segmentation dataset: image + ROI mask."""

    def __init__(self, samples, img_size=512, transform=None, source="bbox"):
        self.img_size = img_size
        self.transform = transform
        self.source = source
        if source == "bbox":
            self.items = [(p, b) for p, b in samples]
        else:
            self.items = [(s.image_path, s.roi_polygon) for s in samples if s.roi_polygon is not None]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, roi = self.items[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        if self.source == "bbox":
            cx, cy, bw, bh = roi
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            mask[max(0, y1) : y2, max(0, x1) : x2] = 1
        else:
            pts = np.array([(int(x * w), int(y * h)) for x, y in roi], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        return img, mask.unsqueeze(0).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build WM overlay predictions from the latest unet_segmentation row in "
            "results/roi_comparison.csv"
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
        default=ROOT / "models" / "weights" / "roi_unet",
        help="Directory with wm_unet_* checkpoints",
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
        default=ROOT / "results" / "roi_unet_predictions.png",
        help="Output image path",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=512,
        help="Inference image size",
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
        default="unet_segmentation",
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
                raise ValueError(f"Invalid run_date '{row.get('run_date')}' in {csv_path}") from exc
            if latest_dt is None or row_dt > latest_dt:
                latest_dt = row_dt
                latest_row = row

    if latest_dt is None or latest_row is None:
        raise ValueError(f"No rows with method='{method}' in {csv_path}")

    return latest_dt, latest_row


def list_checkpoints(weights_dir: Path, prefix: str) -> list[tuple[datetime, Path]]:
    checkpoints = []
    for path in weights_dir.glob(f"{prefix}_unet_*_best.pt"):
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


def load_model(weights_path: Path, device: str) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def show_mask_predictions(model, dataset, axes_flat, device: str):
    """Draw projected red mask overlay on original image size."""

    for i, ax in enumerate(axes_flat):
        if i >= len(dataset):
            ax.axis("off")
            continue

        path, _ = dataset.items[i]
        orig_bgr = cv2.imread(str(path))
        if orig_bgr is None:
            ax.axis("off")
            continue
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_rgb.shape[:2]

        img_tensor, _ = dataset[i]

        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device))
        pred_mask_true = (torch.sigmoid(pred[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
        pred_mask_resized = cv2.resize(
            pred_mask_true,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay = orig_rgb.copy()
        mask_idx = pred_mask_resized > 0
        if mask_idx.any():
            red_tint = np.array([255, 0, 0], dtype=np.float32)
            overlay_float = overlay.astype(np.float32)
            overlay_float[mask_idx] = 0.5 * overlay_float[mask_idx] + 0.5 * red_tint
            overlay = overlay_float.astype(np.uint8)

        ax.imshow(overlay)
        ax.set_title(f"WM #{i + 1}", fontsize=10)
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

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    _, wm_test = load_water_meter_dataset_split(wm_path, train_ratio=args.train_ratio, seed=args.seed)

    wm_test_ds = ROISegmentationDataset(wm_test, args.img_size, val_transform, source="polygon")
    if len(wm_test_ds) == 0:
        raise ValueError("No WM test samples found with ROI")

    total_wm = len(wm_test_ds)
    wm_count = min(args.num_images, total_wm)
    wm_test_ds.items = wm_test_ds.items[:wm_count]
    print(f"WM samples: total={total_wm}, plotted={wm_count}")

    wm_model = load_model(wm_weights, device)

    n_cols = int(np.ceil(np.sqrt(wm_count)))
    n_rows = int(np.ceil(wm_count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes, dtype=object).reshape(-1)

    show_mask_predictions(wm_model, wm_test_ds, axes_flat, device)

    plt.suptitle("U-Net ROI Segmentation (WM only) - Red overlay = predicted mask", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved prediction figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())