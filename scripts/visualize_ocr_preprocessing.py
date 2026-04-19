"""Visual preview of OCR pre-processing with Canny and contours.

Creates a comparison grid for both crop paths:
  - wm_polygon
  - wm_bbox

Each sample row shows:
  [0] original crop
  [1] grayscale + CLAHE
  [2] Canny edges
  [3] closed edges (morphology)
  [4] contours overlay

Usage:
    c:/Users/alike/WaterMeterCV/.venv/Scripts/python.exe scripts/visualize_ocr_preprocessing.py
    c:/Users/alike/WaterMeterCV/.venv/Scripts/python.exe scripts/visualize_ocr_preprocessing.py --n 8 --split test --out results/ocr_preproc_canny_contours.png
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


def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _prep_views(
    img_bgr: np.ndarray,
    canny_low: int,
    canny_high: int,
    min_contour_area: float,
) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

    edges = cv2.Canny(blur, threshold1=canny_low, threshold2=canny_high)

    kernel = np.ones((3, 3), dtype=np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]

    overlay = img_bgr.copy()
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 255, 0), thickness=1)

    gray_eq_bgr = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_closed_bgr = cv2.cvtColor(edges_closed, cv2.COLOR_GRAY2BGR)

    return {
        "orig": img_bgr,
        "gray_eq": gray_eq_bgr,
        "edges": edges_bgr,
        "edges_closed": edges_closed_bgr,
        "overlay": overlay,
        "n_contours": np.array([len(contours)], dtype=np.int32),
    }


def _draw_group(
    axes: np.ndarray,
    row_offset: int,
    group_name: str,
    samples: list[tuple[Path, str]],
    canny_low: int,
    canny_high: int,
    min_contour_area: float,
) -> None:
    for i, (img_path, label) in enumerate(samples):
        row = row_offset + i

        img = cv2.imread(str(img_path))
        if img is None:
            for col in range(5):
                axes[row, col].axis("off")
            continue

        views = _prep_views(
            img,
            canny_low=canny_low,
            canny_high=canny_high,
            min_contour_area=min_contour_area,
        )

        items = [
            ("orig", views["orig"]),
            ("gray+clahe", views["gray_eq"]),
            ("canny", views["edges"]),
            ("closed", views["edges_closed"]),
            ("contours", views["overlay"]),
        ]

        for col, (_, image) in enumerate(items):
            axes[row, col].imshow(_bgr2rgb(image))
            axes[row, col].axis("off")

        n_contours = int(views["n_contours"][0])
        axes[row, 0].set_ylabel(
            f"{group_name}\n{img_path.stem}\nGT={label}\ncont={n_contours}",
            fontsize=8,
            rotation=0,
            labelpad=70,
            va="center",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="samples per crop path")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--canny-low", type=int, default=60)
    parser.add_argument("--canny-high", type=int, default=160)
    parser.add_argument("--min-contour-area", type=float, default=12.0)
    parser.add_argument(
        "--crops-root",
        type=str,
        default=str(ROOT / "WaterMetricsDATA" / "ocr_crops"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "ocr_preproc_canny_contours.png"),
    )
    args = parser.parse_args()

    crops_root = Path(args.crops_root)
    poly = load_ocr_crops(crops_root / "wm_polygon", args.split)
    bbox = load_ocr_crops(crops_root / "wm_bbox", args.split)

    poly = sorted(poly, key=lambda x: x[0].name)[: args.n]
    bbox = sorted(bbox, key=lambda x: x[0].name)[: args.n]

    if not poly and not bbox:
        raise RuntimeError("No OCR crops found. Run prepare_ocr_crops first.")

    rows = len(poly) + len(bbox)
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.3))
    if rows == 1:
        axes = axes[None, :]

    titles = ["original", "gray+clahe", "canny", "closed", "contours"]
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=11, fontweight="bold")

    row_offset = 0
    if poly:
        _draw_group(
            axes,
            row_offset,
            "wm_polygon",
            poly,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            min_contour_area=args.min_contour_area,
        )
        row_offset += len(poly)

    if bbox:
        _draw_group(
            axes,
            row_offset,
            "wm_bbox",
            bbox,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            min_contour_area=args.min_contour_area,
        )

    fig.suptitle(
        f"OCR pre-processing preview | split={args.split} | Canny=({args.canny_low},{args.canny_high})",
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
