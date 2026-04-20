"""Dual-orientation selection with red-bbox + leading-zero + long-tail priors.

Mirrors Notebooks/03_ocr/00_pretrained_ocr_yolo11m.ipynb cell 15.

Main entry: ``select_dual_orientation_with_priors(dual, image_bgr, ocr_model)``
Returns the winning prediction/confidence/angle plus diagnostic vote details.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from models.utils.orientation import DualOrientationResult
from watermetercv.ocr.heuristics import (
    LEADING_ZERO_ORIENTATION_MIN,
    LEADING_ZERO_VOTE_WEIGHT,
    NO_RED_SHORT_READING_MAX_DIGITS,
    NO_RED_SHORT_READING_VOTE_WEIGHT,
    RED_BBOX_VOTE_WEIGHT,
    STAT_TAIL_VOTE_WEIGHT,
    apply_ultralytics_last_drum_heuristic,
    apply_ultralytics_overlap_heuristic,
    is_long_tail_zero_pattern,
    is_no_red_upside_down_pattern,
    leading_zero_count,
)
from watermetercv.ocr.predictor import extract_ultralytics_digit_detections

RED_CLUSTER_MIN_COVERAGE = 0.0025
RED_CLUSTER_LEFT_MAX_X = 0.43
RED_CLUSTER_RIGHT_MIN_X = 0.57
RED_BBOX_MIN_ACTIVE_PIXELS = 64
RED_BBOX_STRICTNESS = 0.08


def estimate_red_horizontal_cluster_in_bboxes(
    image_bgr: np.ndarray,
    detections: list[dict],
    min_coverage: float = RED_CLUSTER_MIN_COVERAGE,
    min_active_pixels: int = RED_BBOX_MIN_ACTIVE_PIXELS,
) -> dict | None:
    if image_bgr is None or image_bgr.size == 0 or not detections:
        return None

    b, g, r = cv2.split(image_bgr.astype(np.float32))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    red_score = (r - np.maximum(g, b)) + 0.25 * s

    h, w = image_bgr.shape[:2]
    weighted_x_sum = 0.0
    weight_total = 0.0
    active_pixels = 0

    for d in detections:
        x1 = max(0, min(w - 1, int(round(d["x1"]))))
        y1 = max(0, min(h - 1, int(round(d["y1"]))))
        x2 = max(0, min(w - 1, int(round(d["x2"]))))
        y2 = max(0, min(h - 1, int(round(d["y2"]))))
        if x2 <= x1 or y2 <= y1:
            continue

        rr = r[y1 : y2 + 1, x1 : x2 + 1]
        gg = g[y1 : y2 + 1, x1 : x2 + 1]
        bb = b[y1 : y2 + 1, x1 : x2 + 1]
        vv = v[y1 : y2 + 1, x1 : x2 + 1]
        score_patch = red_score[y1 : y2 + 1, x1 : x2 + 1]

        base_mask = (vv >= 30.0) & (rr >= 40.0) & (rr >= gg * 1.02) & (rr >= bb * 1.02)
        if not np.any(base_mask):
            continue

        vals = score_patch[base_mask]
        thr = max(float(np.percentile(vals, 85.0)), 10.0)
        strong = base_mask & (score_patch >= thr)
        if not np.any(strong):
            continue

        _, xs = np.where(strong)
        weights = np.maximum(score_patch[strong] - thr, 0.0) + 1e-3
        weighted_x_sum += float(np.sum((xs.astype(np.float32) + float(x1)) * weights))
        weight_total += float(np.sum(weights))
        active_pixels += int(np.sum(strong))

    if active_pixels < int(min_active_pixels) or weight_total <= 1e-6:
        return None

    denom = float(max(w - 1, 1))
    x_norm = float(weighted_x_sum / (weight_total * denom))
    coverage = float(active_pixels / float(max(h * w, 1)))
    if coverage < float(min_coverage):
        return None

    return {"x_norm": x_norm, "coverage": coverage, "active_pixels": int(active_pixels)}


def get_stricter_red_bbox_thresholds(
    strictness: float = RED_BBOX_STRICTNESS,
) -> tuple[float, float, float, int]:
    center = 0.5
    s = float(np.clip(strictness, 0.0, 0.30))
    left_dist = max(center - RED_CLUSTER_LEFT_MAX_X, 0.0)
    right_dist = max(RED_CLUSTER_RIGHT_MIN_X - center, 0.0)

    left_thr = RED_CLUSTER_LEFT_MAX_X - left_dist * s
    right_thr = RED_CLUSTER_RIGHT_MIN_X + right_dist * s
    min_cov = RED_CLUSTER_MIN_COVERAGE * (1.0 + s)
    min_pixels = int(np.ceil(RED_BBOX_MIN_ACTIVE_PIXELS * (1.0 + s)))
    return left_thr, right_thr, min_cov, min_pixels


def red_bbox_orientation_prior(image_bgr: np.ndarray, ocr_model) -> dict | None:
    """Run OCR detection on 0° image, then analyse red-digit horizontal position."""
    detections = extract_ultralytics_digit_detections(image_bgr, ocr_model)
    if not detections:
        return None

    detections, _ = apply_ultralytics_overlap_heuristic(detections)
    detections, _ = apply_ultralytics_last_drum_heuristic(detections)

    left_thr, right_thr, min_cov, min_pixels = get_stricter_red_bbox_thresholds()
    stats = estimate_red_horizontal_cluster_in_bboxes(
        image_bgr, detections, min_coverage=min_cov, min_active_pixels=min_pixels
    )
    if stats is None:
        return None

    x_norm = stats["x_norm"]
    base = {
        **stats,
        "left_thr": float(left_thr),
        "right_thr": float(right_thr),
        "strict_min_coverage": float(min_cov),
        "strict_min_pixels": int(min_pixels),
    }
    if x_norm >= right_thr:
        return {**base, "angle": 0, "reason": "red_bbox_right"}
    if x_norm <= left_thr:
        return {**base, "angle": 180, "reason": "red_bbox_left"}
    return None


def _register_orientation_vote(
    votes: dict[int, float],
    vote_details: dict[str, Any],
    name: str,
    angle: int,
    weight: float,
    extra: dict | None = None,
) -> None:
    if angle not in (0, 180):
        return
    weight = float(max(weight, 0.0))
    if weight <= 0.0:
        return

    a = int(angle)
    votes[a] = float(votes.get(a, 0.0) + weight)
    detail: dict[str, Any] = {"angle": a, "weight": weight}
    if isinstance(extra, dict):
        detail.update(extra)
    vote_details[name] = detail


def select_dual_orientation_with_priors(
    dual: DualOrientationResult,
    image_bgr: np.ndarray | None = None,
    ocr_model: Any = None,
    min_leading_zeros: int = LEADING_ZERO_ORIENTATION_MIN,
) -> dict:
    """Pick 0° vs 180° prediction using a weighted vote of priors.

    Priors (in order): red-bbox location, no-red short-tail pattern,
    long-tail-zero pattern, leading-zero count. Ties resolved by confidence.
    """
    z0 = leading_zero_count(dual.pred_0)
    z180 = leading_zero_count(dual.pred_180)
    conf0 = float(dual.conf_0)
    conf180 = float(dual.conf_180)

    votes: dict[int, float] = {0: 0.0, 180: 0.0}
    vote_details: dict[str, Any] = {}

    red_prior = (
        red_bbox_orientation_prior(image_bgr, ocr_model)
        if image_bgr is not None and ocr_model is not None
        else None
    )
    if red_prior is not None:
        _register_orientation_vote(
            votes, vote_details, "red_bbox",
            int(red_prior["angle"]), RED_BBOX_VOTE_WEIGHT,
            {
                "reason": red_prior.get("reason"),
                "x_norm": float(red_prior.get("x_norm", 0.0)),
                "coverage": float(red_prior.get("coverage", 0.0)),
            },
        )

    short_tail0 = bool(is_no_red_upside_down_pattern(dual.pred_0))
    short_tail180 = bool(is_no_red_upside_down_pattern(dual.pred_180))
    if red_prior is None and short_tail0 and not short_tail180:
        _register_orientation_vote(
            votes, vote_details, "no_red_short_tail_zero",
            180, NO_RED_SHORT_READING_VOTE_WEIGHT,
            {"pattern_0": True, "pattern_180": False,
             "max_digits": int(NO_RED_SHORT_READING_MAX_DIGITS)},
        )
    elif red_prior is None and short_tail180 and not short_tail0:
        _register_orientation_vote(
            votes, vote_details, "no_red_short_tail_zero",
            0, NO_RED_SHORT_READING_VOTE_WEIGHT,
            {"pattern_0": False, "pattern_180": True,
             "max_digits": int(NO_RED_SHORT_READING_MAX_DIGITS)},
        )

    tail0 = bool(is_long_tail_zero_pattern(dual.pred_0))
    tail180 = bool(is_long_tail_zero_pattern(dual.pred_180))
    if tail0 and not tail180:
        _register_orientation_vote(
            votes, vote_details, "long_tail_zero",
            180, STAT_TAIL_VOTE_WEIGHT,
            {"tail0": True, "tail180": False},
        )
    elif tail180 and not tail0:
        _register_orientation_vote(
            votes, vote_details, "long_tail_zero",
            0, STAT_TAIL_VOTE_WEIGHT,
            {"tail0": False, "tail180": True},
        )

    if z0 >= min_leading_zeros and z180 < min_leading_zeros:
        _register_orientation_vote(
            votes, vote_details, "leading_zeros",
            0, LEADING_ZERO_VOTE_WEIGHT,
            {"leading_zeros_0": int(z0), "leading_zeros_180": int(z180)},
        )
    elif z180 >= min_leading_zeros and z0 < min_leading_zeros:
        _register_orientation_vote(
            votes, vote_details, "leading_zeros",
            180, LEADING_ZERO_VOTE_WEIGHT,
            {"leading_zeros_0": int(z0), "leading_zeros_180": int(z180)},
        )

    vote_score_0 = float(votes[0])
    vote_score_180 = float(votes[180])
    if vote_score_180 > vote_score_0 + 1e-9:
        selected_angle = 180
        reason = "vote_180"
    elif vote_score_0 > vote_score_180 + 1e-9:
        selected_angle = 0
        reason = "vote_0"
    elif conf180 > conf0:
        selected_angle = 180
        reason = "confidence_180_tiebreak"
    else:
        selected_angle = 0
        reason = "confidence_0_tiebreak"

    selected_pred = dual.pred_180 if selected_angle == 180 else dual.pred_0
    selected_conf = conf180 if selected_angle == 180 else conf0

    return {
        "selected_pred": selected_pred,
        "selected_conf": float(selected_conf),
        "selected_angle": int(selected_angle),
        "reason": reason,
        "vote_score_0": vote_score_0,
        "vote_score_180": vote_score_180,
        "heuristic_votes": vote_details,
    }
