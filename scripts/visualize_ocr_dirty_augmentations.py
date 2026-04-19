"""Visual preview for dirty-robust OCR augmentations.

Shows how synthetic corruption changes OCR crops for both wm_polygon and wm_bbox.

Usage:
    c:/Users/alike/WaterMeterCV/.venv/Scripts/python.exe scripts/visualize_ocr_dirty_augmentations.py
    c:/Users/alike/WaterMeterCV/.venv/Scripts/python.exe scripts/visualize_ocr_dirty_augmentations.py --n 6 --split test --out results/ocr_dirty_aug_preview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.data.ocr_dataset import load_ocr_crops
from models.data.augmentations_ocr import get_ocr_train_transforms


def _gray_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unexpected image shape: {image.shape}")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_rows(
    axes: np.ndarray,
    row_offset: int,
    title_prefix: str,
    samples: list[tuple[Path, str]],
    aug,
) -> None:
    for i, (img_path, label) in enumerate(samples):
        row = row_offset + i
        base = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if base is None:
            for c in range(4):
                axes[row, c].axis("off")
            continue

        aug1 = _to_uint8(aug(image=base)["image"])
        aug2 = _to_uint8(aug(image=base)["image"])
        aug3 = _to_uint8(aug(image=base)["image"])

        show = [base, aug1, aug2, aug3]
        for c, image in enumerate(show):
            axes[row, c].imshow(_gray_to_rgb(image))
            axes[row, c].axis("off")

        axes[row, 0].set_ylabel(
            f"{title_prefix}\n{img_path.stem}\nGT={label}",
            fontsize=8,
            rotation=0,
            labelpad=62,
            va="center",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="samples per crop path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--crops-root",
        type=str,
        default=str(ROOT / "WaterMetricsDATA" / "ocr_crops"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "ocr_dirty_aug_preview.png"),
    )
    args = parser.parse_args()

    crops_root = Path(args.crops_root)
    poly = load_ocr_crops(crops_root / "wm_polygon", args.split)
    bbox = load_ocr_crops(crops_root / "wm_bbox", args.split)

    poly = sorted(poly, key=lambda x: x[0].name)[: args.n]
    bbox = sorted(bbox, key=lambda x: x[0].name)[: args.n]

    if not poly and not bbox:
        raise RuntimeError("No OCR crops found. Run prepare_ocr_crops first.")

    aug = get_ocr_train_transforms(profile="dirty_robust", p_rotate_180=0.25, to_tensor=False)

    rows = len(poly) + len(bbox)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4))
    if rows == 1:
        axes = axes[None, :]

    col_titles = ["original", "dirty aug #1", "dirty aug #2", "dirty aug #3"]
    for i, title in enumerate(col_titles):
        axes[0, i].set_title(title, fontsize=11, fontweight="bold")

    row_offset = 0
    if poly:
        _draw_rows(axes, row_offset, "wm_polygon", poly, aug)
        row_offset += len(poly)
    if bbox:
        _draw_rows(axes, row_offset, "wm_bbox", bbox, aug)

    fig.suptitle(
        f"Dirty-robust OCR augmentation preview | split={args.split}",
        fontsize=13,
        y=1.002,
    )
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
