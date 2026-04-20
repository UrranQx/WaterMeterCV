import time
from typing import Callable
from rapidfuzz.distance import Levenshtein


def full_string_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    """Fraction of predictions that exactly match ground truth."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
    return correct / len(predictions)


def per_digit_accuracy(pred: str, gt: str) -> float:
    """Per-character accuracy between two strings, aligned left-to-right."""
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 0.0
    pred_padded = pred.ljust(max_len)
    gt_padded = gt.ljust(max_len)
    correct = sum(1 for p, g in zip(pred_padded, gt_padded) if p == g)
    return correct / max_len


def character_error_rate(pred: str, gt: str) -> float:
    """CER = edit_distance(pred, gt) / len(gt)."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return Levenshtein.distance(pred, gt) / len(gt)


def compute_iou_bbox(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """IoU for two bboxes in (cx, cy, w, h) normalized format."""
    ax1 = a[0] - a[2] / 2
    ay1 = a[1] - a[3] / 2
    ax2 = a[0] + a[2] / 2
    ay2 = a[1] + a[3] / 2

    bx1 = b[0] - b[2] / 2
    by1 = b[1] - b[3] / 2
    bx2 = b[0] + b[2] / 2
    by2 = b[1] + b[3] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_iou_polygon(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
) -> float:
    """IoU for two polygons using shapely."""
    from shapely.geometry import Polygon

    a = Polygon(poly_a)
    b = Polygon(poly_b)

    if not a.is_valid or not b.is_valid:
        return 0.0

    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def measure_inference_time(fn: Callable, *args, n_runs: int = 10, **kwargs) -> float:
    """Average inference time in milliseconds."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000)
    return sum(times) / len(times)
