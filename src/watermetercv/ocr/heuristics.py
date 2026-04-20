"""OCR post-processing heuristics extracted from notebook 00.

Pure functions operating on detection dicts:
    {"digit", "conf", "x1","y1","x2","y2", "cx","cy","w","h"}

All constants mirror Notebooks/03_ocr/00_pretrained_ocr_yolo11m.ipynb (cells 10, 12).
"""
from __future__ import annotations

import numpy as np

MAX_READING_DIGITS = 10
LEADING_ZERO_ORIENTATION_MIN = 3

ULTRA_LAST_DRUM_X_ALIGN_FACTOR = 0.55
ULTRA_LAST_DRUM_MIN_Y_GAP_FACTOR = 0.25
ULTRA_OVERLAP_IOA_MIN = 0.65
ULTRA_OVERLAP_CENTER_FACTOR = 0.55

LONG_TAIL_ZERO_MIN_DIGITS = 8
LONG_TAIL_ZERO_MIN_SUFFIX = 2

NO_RED_SHORT_READING_MAX_DIGITS = 7
NO_RED_SHORT_READING_VOTE_WEIGHT = 0.07

RED_BBOX_VOTE_WEIGHT = 0.85
STAT_TAIL_VOTE_WEIGHT = 0.09
LEADING_ZERO_VOTE_WEIGHT = 0.06


def digits_only(text) -> str:
    return "".join(ch for ch in str(text) if ch.isdigit())


def safe_mean(values) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def apply_max_digits_heuristic(
    pred: str, conf: float, max_digits: int = MAX_READING_DIGITS
) -> tuple[str, float]:
    pred_digits = digits_only(pred)
    try:
        conf_val = float(conf)
    except (TypeError, ValueError):
        conf_val = 0.0

    if len(pred_digits) <= max_digits:
        return pred_digits, conf_val

    overflow = len(pred_digits) - max_digits
    penalty = max(0.0, 1.0 - (overflow / max_digits))
    return pred_digits[:max_digits], conf_val * penalty


def leading_zero_count(text) -> int:
    d = digits_only(text)
    n = 0
    for ch in d:
        if ch == "0":
            n += 1
        else:
            break
    return n


def normalize_digits_for_stats(text) -> str:
    d = digits_only(text).lstrip("0")
    return d if d else "0"


def trailing_zero_count(text) -> int:
    n = 0
    for ch in reversed(str(text)):
        if ch == "0":
            n += 1
        else:
            break
    return n


def is_long_tail_zero_pattern(text) -> bool:
    norm = normalize_digits_for_stats(text)
    return (
        len(norm) >= LONG_TAIL_ZERO_MIN_DIGITS
        and trailing_zero_count(norm) >= LONG_TAIL_ZERO_MIN_SUFFIX
    )


def is_no_red_upside_down_pattern(
    text, max_digits: int = NO_RED_SHORT_READING_MAX_DIGITS
) -> bool:
    d = digits_only(text)
    if not d:
        return False
    if len(d) > int(max_digits):
        return False
    return d[0] != "0" and d[-1] == "0"


def ultralytics_digit_rank(digit) -> int:
    d = int(digit)
    # Drum rollover: 0 ranks above 9.
    return 10 if d == 0 else d


def bbox_intersection_area(a: dict, b: dict) -> float:
    x_left = max(a["x1"], b["x1"])
    y_top = max(a["y1"], b["y1"])
    x_right = min(a["x2"], b["x2"])
    y_bottom = min(a["y2"], b["y2"])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    return float((x_right - x_left) * (y_bottom - y_top))


def bbox_area(d: dict) -> float:
    return float(max(d["x2"] - d["x1"], 0.0) * max(d["y2"] - d["y1"], 0.0))


def boxes_are_nested_or_almost_nested(a: dict, b: dict) -> bool:
    inter = bbox_intersection_area(a, b)
    if inter <= 0.0:
        return False

    area_a = max(bbox_area(a), 1e-6)
    area_b = max(bbox_area(b), 1e-6)
    ioa_small = inter / min(area_a, area_b)

    if ioa_small < ULTRA_OVERLAP_IOA_MIN:
        return False

    x_close = abs(a["cx"] - b["cx"]) <= ULTRA_OVERLAP_CENTER_FACTOR * max(a["w"], b["w"], 1e-6)
    y_close = abs(a["cy"] - b["cy"]) <= ULTRA_OVERLAP_CENTER_FACTOR * max(a["h"], b["h"], 1e-6)
    return bool(x_close and y_close)


def apply_ultralytics_overlap_heuristic(detections: list[dict]) -> tuple[list[dict], dict]:
    dets = sorted(detections, key=lambda d: d["cx"])
    n = len(dets)
    if n < 2:
        return dets, {"applied": False, "reason": "not_enough_boxes"}

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a_idx: int, b_idx: int) -> None:
        ra, rb = find(a_idx), find(b_idx)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_are_nested_or_almost_nested(dets[i], dets[j]):
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(dets[idx])

    filtered: list[dict] = []
    applied_count = 0
    for _, group in groups.items():
        if len(group) == 1:
            filtered.append(group[0])
            continue

        applied_count += 1
        zeros = [g for g in group if int(g["digit"]) == 0]
        if zeros:
            chosen = max(zeros, key=lambda g: (float(g["conf"]), -bbox_area(g)))
        else:
            chosen = max(group, key=lambda g: (ultralytics_digit_rank(g["digit"]), float(g["conf"])))
        filtered.append(chosen)

    filtered.sort(key=lambda d: d["cx"])
    return filtered, {
        "applied": applied_count > 0,
        "reason": "overlap_group",
        "group_count": int(applied_count),
    }


def apply_ultralytics_last_drum_heuristic(detections: list[dict]) -> tuple[list[dict], dict]:
    dets = sorted(detections, key=lambda d: d["cx"])
    if len(dets) < 2:
        return dets, {"applied": False, "reason": "not_enough_boxes"}

    a, b = dets[-2], dets[-1]
    w_ref = max(a["w"], b["w"], 1e-6)
    h_ref = max(a["h"], b["h"], 1e-6)

    x_close = abs(a["cx"] - b["cx"]) <= ULTRA_LAST_DRUM_X_ALIGN_FACTOR * w_ref
    y_split = abs(a["cy"] - b["cy"]) >= ULTRA_LAST_DRUM_MIN_Y_GAP_FACTOR * h_ref
    x_overlap = min(a["x2"], b["x2"]) > max(a["x1"], b["x1"])

    if not (x_close and y_split and x_overlap):
        return dets, {"applied": False, "reason": "not_vertical_pair"}

    rank_a = ultralytics_digit_rank(a["digit"])
    rank_b = ultralytics_digit_rank(b["digit"])
    if rank_a != rank_b:
        chosen = a if rank_a > rank_b else b
        reason = "digit_rank"
    elif a["conf"] != b["conf"]:
        chosen = a if a["conf"] >= b["conf"] else b
        reason = "confidence"
    elif a["cy"] != b["cy"]:
        chosen = a if a["cy"] < b["cy"] else b
        reason = "upper_tiebreak"
    else:
        chosen = a
        reason = "left_tiebreak"

    dropped = b if chosen is a else a
    filtered = dets[:-2] + [chosen]
    return filtered, {
        "applied": True,
        "reason": reason,
        "kept_digit": int(chosen["digit"]),
        "dropped_digit": int(dropped["digit"]),
    }
