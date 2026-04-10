import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def draw_digit_bboxes(
    image: np.ndarray,
    bboxes: list[tuple[int, float, float, float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw digit bounding boxes on image.

    Args:
        bboxes: list of (class_id, cx, cy, w, h) in normalized coords.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for cls_id, cx, cy, bw, bh in bboxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, str(cls_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


def draw_roi_polygon(
    image: np.ndarray,
    polygon: list[tuple[float, float]],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw ROI polygon overlay on image.

    Args:
        polygon: list of (x, y) in normalized coords.
    """
    img = image.copy()
    h, w = img.shape[:2]
    pts = np.array([(int(x * w), int(y * h)) for x, y in polygon], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    return img


def plot_comparison_table(
    results: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> None:
    """Plot a comparison table of experiment results.

    Args:
        results: {experiment_name: {metric_name: value}}
    """
    experiments = list(results.keys())
    if not experiments:
        return

    metrics = list(results[experiments[0]].keys())
    data = [[results[exp].get(m, 0.0) for m in metrics] for exp in experiments]

    fig, ax = plt.subplots(figsize=(len(metrics) * 2, len(experiments) * 0.6 + 1))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{v:.4f}" for v in row] for row in data],
        rowLabels=experiments,
        colLabels=metrics,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
