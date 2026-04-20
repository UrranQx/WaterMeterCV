"""Shared OCR data helpers for all 03_ocr notebooks.

Two persistent crop datasets are created by prepare_ocr_crops() and loaded
by load_ocr_crops(). All notebooks use these pre-computed crops.

Crop types:
  wm_polygon — WM roi_polygon perspective-warped (best quality)
  wm_bbox    — WM detected bbox + projection-profile rotation crop (comparison)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import csv
import numpy as np
import cv2
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

CHARSET = "0123456789"   # 10 digits; CTC blank = index 10
OUT_H   = 64
OUT_W   = 256

LABEL_MODE_SOURCE = "source"
LABEL_MODE_WM_FRACTION_AWARE = "wm_fraction_aware"

DOT_ZERO_POLICY_AUTO_RED = "auto_red"
DOT_ZERO_POLICY_DROP_FRACTION = "drop_fraction"
DOT_ZERO_POLICY_EXPAND_HIDDEN = "expand_hidden"
DOT_ZERO_POLICY_PRESERVE_SOURCE = "preserve_source"
DOT_ZERO_POLICIES = {
    DOT_ZERO_POLICY_AUTO_RED,
    DOT_ZERO_POLICY_DROP_FRACTION,
    DOT_ZERO_POLICY_EXPAND_HIDDEN,
    DOT_ZERO_POLICY_PRESERVE_SOURCE,
}


def value_text_to_ocr_label(value_text: str) -> str:
    """Convert value text to OCR target label.

    Current OCR charset is digits-only, so separators are removed while keeping
    all integer and fractional digits (e.g. "595.825" -> "595825").
    """
    return "".join(ch for ch in str(value_text) if ch.isdigit())


def _split_value_text_parts(value_text: str) -> tuple[str, str]:
    """Split numeric text into integer/fraction digit parts.

    Only digits are kept in each part; separator is normalized to ".".
    """
    text = str(value_text).strip().replace(",", ".")
    if not text:
        return "", ""

    if "." in text:
        int_part, frac_part = text.split(".", 1)
    else:
        int_part, frac_part = text, ""

    int_digits = "".join(ch for ch in int_part if ch.isdigit())
    frac_digits = "".join(ch for ch in frac_part if ch.isdigit())
    if not int_digits and frac_digits:
        int_digits = "0"
    return int_digits, frac_digits


def normalize_wm_value_text_for_ocr(
    value_text: str,
    has_fractional_red: bool | None = None,
    dot_zero_policy: str = DOT_ZERO_POLICY_DROP_FRACTION,
) -> str:
    """Normalize WM value text for OCR labels with fraction-aware rules.

    Rules derived from dataset conventions:
    - 3+ fractional digits: keep as-is.
    - 2 fractional digits: assume hidden 3rd drum is 0 and append one zero.
    - 1 non-zero fractional digit: treat as abbreviated fraction and append two zeros.
    - 1 zero fractional digit (".0") follows dot_zero_policy:
      * drop_fraction (default): interpret as integer meter (drop fraction)
      * preserve_source: keep source ".0"
      * expand_hidden: force hidden fraction as "000"
      * auto_red: infer by has_fractional_red (False->drop, True->"000", None->"0")
    """
    if dot_zero_policy not in DOT_ZERO_POLICIES:
        expected = ", ".join(sorted(DOT_ZERO_POLICIES))
        raise ValueError(
            f"Unsupported dot_zero_policy: {dot_zero_policy!r}. "
            f"Expected one of: {expected}"
        )

    int_digits, frac_digits = _split_value_text_parts(value_text)
    if not int_digits and not frac_digits:
        return ""

    if len(frac_digits) >= 3:
        frac_norm = frac_digits
    elif len(frac_digits) == 2:
        frac_norm = frac_digits + "0"
    elif len(frac_digits) == 1:
        if frac_digits != "0":
            frac_norm = frac_digits + "00"
        elif dot_zero_policy == DOT_ZERO_POLICY_DROP_FRACTION:
            frac_norm = ""
        elif dot_zero_policy == DOT_ZERO_POLICY_PRESERVE_SOURCE:
            frac_norm = frac_digits
        elif dot_zero_policy == DOT_ZERO_POLICY_EXPAND_HIDDEN:
            frac_norm = "000"
        elif has_fractional_red is False:
            frac_norm = ""
        elif has_fractional_red is True:
            frac_norm = "000"
        else:
            frac_norm = frac_digits
    else:
        frac_norm = ""

    if frac_norm:
        return f"{int_digits}.{frac_norm}"
    return int_digits


def _normalize_legacy_float_value(raw_value: float) -> str | None:
    """Legacy fallback for samples that have numeric value but no value_text.

    Primary path uses sample.value_text and keeps all source digits.
    """
    if raw_value is None:
        return None
    text = str(raw_value).strip().replace(",", ".")
    if not text:
        return None
    try:
        dec = Decimal(text).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None
    norm = format(dec, "f")
    if "." in norm:
        norm = norm.rstrip("0").rstrip(".")
    return norm if norm else None


def sample_to_ocr_label(
    sample,
    label_mode: str = LABEL_MODE_SOURCE,
    has_fractional_red: bool | None = None,
    dot_zero_policy: str = DOT_ZERO_POLICY_DROP_FRACTION,
) -> str:
    """Build stable OCR label from UnifiedSample value/value_text.

    label_mode:
      - "source": keep all source digits exactly as annotated.
      - "wm_fraction_aware": apply WM-specific decimal normalization rules.
        dot_zero_policy:
            Policy for handling ".0" in wm_fraction_aware mode.
    """
    if label_mode not in {LABEL_MODE_SOURCE, LABEL_MODE_WM_FRACTION_AWARE}:
        raise ValueError(
            f"Unsupported label_mode: {label_mode!r}. "
            f"Expected one of: {LABEL_MODE_SOURCE!r}, {LABEL_MODE_WM_FRACTION_AWARE!r}"
        )

    raw_text = getattr(sample, "value_text", None)
    if raw_text is not None:
        if label_mode == LABEL_MODE_WM_FRACTION_AWARE:
            norm_text = normalize_wm_value_text_for_ocr(
                raw_text,
                has_fractional_red=has_fractional_red,
                dot_zero_policy=dot_zero_policy,
            )
            return value_text_to_ocr_label(norm_text)
        return value_text_to_ocr_label(raw_text)

    raw_value = getattr(sample, "value", None)
    norm = _normalize_legacy_float_value(raw_value)
    if norm is None:
        return ""
    return value_text_to_ocr_label(norm)


def _estimate_fractional_red_presence(
    image_bgr: np.ndarray,
    polygon: list[tuple[float, float]] | None,
    min_ratio: float = 0.0012,
    min_pixels: int = 24,
) -> bool | None:
    """Estimate whether the fractional (right) meter area contains red digits.

    Returns True/False when estimation is possible, otherwise None.
    """
    if image_bgr is None or image_bgr.size == 0 or polygon is None or len(polygon) < 3:
        return None

    h, w = image_bgr.shape[:2]
    pts = []
    for x_norm, y_norm in polygon:
        x = int(round(float(np.clip(x_norm, 0.0, 1.0)) * max(w - 1, 1)))
        y = int(round(float(np.clip(y_norm, 0.0, 1.0)) * max(h - 1, 1)))
        pts.append((x, y))

    if len(pts) < 3:
        return None

    poly = np.array(pts, dtype=np.int32)
    if cv2.contourArea(poly) <= 1.0:
        return None

    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly], 255)

    x, y, bw, bh = cv2.boundingRect(poly)
    if bw <= 0 or bh <= 0:
        return None
    right_x = x + int(round(0.55 * bw))
    right_x = max(x, min(x + bw, right_x))

    right_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask[y : y + bh, right_x : x + bw] = 255
    roi_mask = (poly_mask > 0) & (right_mask > 0)

    roi_pixels = int(np.count_nonzero(roi_mask))
    if roi_pixels < 32:
        return None

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    b_ch, g_ch, r_ch = cv2.split(image_bgr)

    red_hue = (h_ch <= 12) | (h_ch >= 168)
    sat_ok = s_ch >= 70
    val_ok = v_ch >= 40
    dominance = (r_ch.astype(np.int16) - np.maximum(g_ch, b_ch).astype(np.int16)) >= 18

    red_mask = red_hue & sat_ok & val_ok & dominance & roi_mask
    red_pixels = int(np.count_nonzero(red_mask))
    coverage = float(red_pixels) / float(max(roi_pixels, 1))
    return bool(red_pixels >= int(min_pixels) and coverage >= float(min_ratio))


# ─── Rotation & corner helpers ───────────────────────────────────────────────

def _estimate_rotation(pts: np.ndarray) -> float:
    """Estimate rotation angle (degrees) of the ROI's long axis from horizontal.

    Uses minAreaRect + long-side direction vector.  Returns angle in (-90, 90].
    The 0°/180° ambiguity is *not* resolved here — downstream orientation
    detection (OCR confidence on 0° vs 180°) handles that.

    Works with any point set (GT polygon, U-Net contour, etc.).
    """
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)

    # Identify the longer pair of sides
    d01 = float(np.linalg.norm(box[1] - box[0]))
    d12 = float(np.linalg.norm(box[2] - box[1]))
    long_vec = (box[1] - box[0]) if d01 >= d12 else (box[2] - box[1])

    angle = float(np.degrees(np.arctan2(long_vec[1], long_vec[0])))

    # Normalise to (-90, 90] — keeps the smaller rotation, punts 180° to
    # orientation detection.
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle


def _order_corners(box: np.ndarray, angle_deg: float) -> np.ndarray:
    """Order 4 box corners as TL, TR, BR, BL for any rotation.

    Rotates the points into a virtual "horizontal" frame (using *angle_deg*),
    then picks top-left / top-right / bottom-right / bottom-left by simple
    y-then-x sorting.  Maps the indices back to the original coordinates.
    """
    cx = box[:, 0].mean()
    cy = box[:, 1].mean()
    cos_a = np.cos(np.radians(-angle_deg))
    sin_a = np.sin(np.radians(-angle_deg))

    # Rotate to horizontal frame
    dx = box[:, 0] - cx
    dy = box[:, 1] - cy
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a

    # Top row = smaller ry, bottom row = larger ry
    idx = list(range(4))
    idx.sort(key=lambda i: (ry[i], rx[i]))
    top = sorted(idx[:2], key=lambda i: rx[i])
    bot = sorted(idx[2:], key=lambda i: rx[i])

    return np.array(
        [box[top[0]], box[top[1]], box[bot[1]], box[bot[0]]],
        dtype=np.float32,
    )


def estimate_roi_rotation(
    polygon: list[tuple[float, float]],
    img_shape: tuple[int, ...],
) -> float:
    """Public helper: rotation angle (degrees) from a normalised polygon.

    Usable in notebooks and in the inference pipeline (U-Net / YOLO outputs).
    """
    h, w = img_shape[:2]
    pts = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)
    return _estimate_rotation(pts)


# ─── Content-based rotation (for bbox path) ───────────────────────────────────

def _projection_score(gray: np.ndarray) -> float:
    """Std of per-row horizontal-edge density (Sobel Y).

    Sobel Y highlights the top/bottom borders of the digit strip.
    When the strip is horizontal those borders concentrate in a few rows →
    high std.  More robust than raw brightness on cluttered backgrounds.
    """
    # Suppress side clutter (e.g. objects touching bbox corners) by scoring
    # mostly on the central width where the digit strip usually sits.
    h, w = gray.shape[:2]
    margin = int(round(w * 0.15))
    if w - 2 * margin >= 16:
        gray = gray[:, margin:w - margin]

    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    proj = np.abs(sobel).sum(axis=1)
    return float(proj.std())


def _select_peak_by_sharpness(
    angles: list[float],
    scores: list[float],
    score_ratio: float = 0.85,
    sharp_ratio: float = 0.6,
) -> float:
    """Pick the highest peak among sufficiently sharp local maxima.

    The method filters to local maxima, keeps peaks that are both high enough
    in score and sharpness, then selects the highest one among those. This
    avoids broad clutter-driven maxima while not over-favoring a lower but
    extremely sharp secondary peak.
    """
    if not angles:
        return 0.0
    if len(angles) < 3:
        return float(angles[int(np.argmax(scores))])

    # Find local maxima and their sharpness.
    peaks: list[tuple[int, float, float]] = []
    for i in range(1, len(scores) - 1):
        s = scores[i]
        left = scores[i - 1]
        right = scores[i + 1]
        if s >= left and s >= right:
            sharp = s - 0.5 * (left + right)
            peaks.append((i, s, sharp))

    if not peaks:
        return float(angles[int(np.argmax(scores))])

    max_score = max(p[1] for p in peaks)
    max_sharp = max(p[2] for p in peaks)

    candidates = [
        p for p in peaks
        if p[1] >= score_ratio * max_score and p[2] >= sharp_ratio * max_sharp
    ]
    if not candidates:
        candidates = [p for p in peaks if p[1] >= score_ratio * max_score]
    if not candidates:
        candidates = peaks

    # Prefer the highest surviving peak; use sharpness and |angle| as tiebreakers.
    best = max(candidates, key=lambda p: (p[1], p[2], -abs(angles[p[0]])))
    return float(angles[best[0]])


def estimate_rotation_from_crop(
    crop_bgr: np.ndarray,
    angle_range: float = 90.0,
    coarse_step: float = 2.0,
    fine_step: float = 0.5,
) -> float:
    """Estimate rotation angle by maximising horizontal projection sharpness.

    Coarse search over [-angle_range, +angle_range] at *coarse_step* resolution,
    then fine search ±coarse_step around the best with *fine_step* resolution.

    Default *angle_range* is 90° to handle all real-world meter orientations.
    In ``crop_roi_from_detection`` the range is automatically restricted to ±45°
    once the crop is already landscape (prevents rotating landscape → portrait).

    Returns the CCW angle in degrees to pass to ``cv2.getRotationMatrix2D``
    so the digit line becomes horizontal.  No external models needed — works
    identically during training and inference.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = (w / 2.0, h / 2.0)

    def _rotated_gray(angle: float) -> np.ndarray:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _score_full(angle: float) -> float:
        return _projection_score(_rotated_gray(angle))

    def _score_center(angle: float) -> float:
        rot = _rotated_gray(angle)
        y1 = int(round(0.20 * h))
        y2 = int(round(0.80 * h))
        x1 = int(round(0.20 * w))
        x2 = int(round(0.80 * w))
        roi = rot[y1:y2, x1:x2]
        if roi.size == 0:
            return _projection_score(rot)
        return _projection_score(roi)

    def _search_best(score_fn) -> float:
        coarse_angles = [float(a) for a in np.arange(
            -angle_range,
            angle_range + coarse_step,
            coarse_step,
        )]
        coarse_scores = [score_fn(a) for a in coarse_angles]
        coarse_best = _select_peak_by_sharpness(
            coarse_angles,
            coarse_scores,
            score_ratio=0.85,
            sharp_ratio=0.6,
        )

        fine_lo = max(-angle_range, coarse_best - coarse_step)
        fine_hi = min(angle_range, coarse_best + coarse_step)
        fine_angles = [float(a) for a in np.arange(
            fine_lo,
            fine_hi + fine_step,
            fine_step,
        )]
        fine_scores = [score_fn(a) for a in fine_angles]
        return _select_peak_by_sharpness(
            fine_angles,
            fine_scores,
            score_ratio=0.85,
            sharp_ratio=0.6,
        )

    best_full = _search_best(_score_full)
    best_center = _search_best(_score_center)

    # Rare fallback: if full-field and center-field scorers pick opposite
    # strong tilts, prefer center-field (digit strip is usually central).
    if (
        np.sign(best_full) != np.sign(best_center)
        and abs(best_full) >= 15.0
        and abs(best_center) >= 15.0
    ):
        return float(best_center)

    return float(best_full)


# ─── Crop helpers ─────────────────────────────────────────────────────────────

def warp_roi_polygon(
    img_bgr: np.ndarray,
    polygon: list[tuple[float, float]],
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> np.ndarray:
    """Perspective-warp the roi_polygon region to (out_h × out_w).

    Uses rotation-aware corner ordering so it works correctly on rotated
    meters (any angle).  0°/180° ambiguity is not resolved here.
    """
    h, w = img_bgr.shape[:2]
    pts = np.array([[p[0] * w, p[1] * h] for p in polygon], dtype=np.float32)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    angle = _estimate_rotation(pts)
    corners = _order_corners(box, angle)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    return cv2.warpPerspective(
        img_bgr, cv2.getPerspectiveTransform(corners, dst), (out_w, out_h),
    )


def crop_roi_bbox(
    img_bgr: np.ndarray,
    bbox_cxcywh: tuple[float, float, float, float],
    out_h: int = OUT_H,
    out_w: int = OUT_W,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Axis-aligned bbox crop resized to (out_h × out_w).

    If *rotation_deg* is supplied (e.g. from ``estimate_roi_rotation``), the
    image is first rotated around the bbox centre so the ROI becomes horizontal
    before cropping.  This works both during training (angle from GT polygon)
    and during inference (angle from U-Net polygon or other source).
    """
    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = bbox_cxcywh

    if abs(rotation_deg) > 0.5:
        cx_px, cy_px = cx * w, cy * h
        M = cv2.getRotationMatrix2D((cx_px, cy_px), rotation_deg, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h))

    x1 = max(0, int((cx - bw / 2) * w))
    y1 = max(0, int((cy - bh / 2) * h))
    x2 = min(w, int((cx + bw / 2) * w))
    y2 = min(h, int((cy + bh / 2) * h))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_w, out_h))


def crop_roi_from_detection(
    img_bgr: np.ndarray,
    bbox_cxcywh: tuple[float, float, float, float],
    out_h: int = OUT_H,
    out_w: int = OUT_W,
) -> np.ndarray:
    """Bbox-path crop: rotate original image, crop, optional vertical recrop.

    1. Cut a padded estimation crop to find the rotation angle.
    2. Coarse 90°: if crop is portrait, decide CW or CCW via projection score.
    3. Fine angle via projection-profile search on coarse-corrected crop.
    4. Rotate the **original image** around (cx, cy) by total angle —
       surrounding real pixels fill in, no border artifacts.
    5. Adaptive padding (more for angles near 45°) → first crop.
    6. Optional second crop from ``rotated`` with same centre/same width and
       ``H_c2 = W_c * out_h / out_w`` (vertical crop only).
       If ``H_c2 < out_h``, skip this second crop.
    7. Resize final crop to (out_w, out_h) for fixed output shape.

    Works identically during training and inference.
    """
    h, w = img_bgr.shape[:2]
    cx, cy, bw, bh = bbox_cxcywh
    cx_px, cy_px = cx * w, cy * h
    bw_px, bh_px = bw * w, bh * h

    if bw_px < 1 or bh_px < 1:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # ── estimation crop (generous padding to see tilted content) ──────────
    est_pad = 0.3
    ex1 = max(0, int(cx_px - bw_px * (1 + est_pad) / 2))
    ey1 = max(0, int(cy_px - bh_px * (1 + est_pad) / 2))
    ex2 = min(w, int(cx_px + bw_px * (1 + est_pad) / 2))
    ey2 = min(h, int(cy_px + bh_px * (1 + est_pad) / 2))
    est_crop = img_bgr[ey1:ey2, ex1:ex2]

    if est_crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    ech, ecw = est_crop.shape[:2]
    coarse_angle = 0.0

    # ── coarse 90° for portrait crops ────────────────────────────────────
    if ech > ecw * 1.2:
        rot_cw  = cv2.rotate(est_crop, cv2.ROTATE_90_CLOCKWISE)
        rot_ccw = cv2.rotate(est_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        g_cw    = cv2.cvtColor(rot_cw,  cv2.COLOR_BGR2GRAY)
        g_ccw   = cv2.cvtColor(rot_ccw, cv2.COLOR_BGR2GRAY)
        if _projection_score(g_cw) >= _projection_score(g_ccw):
            est_crop = rot_cw
            coarse_angle = -90.0          # CW ≡ -90° in cv2 convention
        else:
            est_crop = rot_ccw
            coarse_angle = 90.0           # CCW ≡ +90°

    # ── fine angle via projection profile ────────────────────────────────
    ech2, ecw2 = est_crop.shape[:2]
    fine_range = 45.0 if (ecw2 > ech2 * 1.1) else 90.0
    fine_angle = estimate_rotation_from_crop(est_crop, angle_range=fine_range)

    total_angle = coarse_angle + fine_angle

    # ── rotate original image around bbox centre ─────────────────────────
    M = cv2.getRotationMatrix2D((cx_px, cy_px), total_angle, 1.0)
    rotated = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # ── adaptive padding (more for angles near 45°) + crop ───────────────
    fine_abs_rad = np.radians(np.abs(fine_angle))
    pad = 0.1 + 0.4 * float(np.abs(np.sin(2.0 * fine_abs_rad)))

    meter_long  = max(bw_px, bh_px)
    meter_short = min(bw_px, bh_px)
    crop_w = meter_long  * (1.0 + pad)
    crop_h = meter_short * (1.0 + pad)

    x1 = max(0, int(cx_px - crop_w / 2))
    y1 = max(0, int(cy_px - crop_h / 2))
    x2 = min(w, int(cx_px + crop_w / 2))
    y2 = min(h, int(cy_px + crop_h / 2))

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    crop_h_px, crop_w_px = crop.shape[:2]
    target_h2 = int(round(crop_w_px * out_h / out_w))

    # Vertical recrop around the first-crop centre, but sampled from rotated.
    # Using rotated (not crop) lets us expand/shrink vertically with real pixels.
    if target_h2 >= out_h:
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        x1_2 = max(0, int(round(center_x - crop_w_px / 2.0)))
        x2_2 = min(w, int(round(center_x + crop_w_px / 2.0)))
        y1_2 = max(0, int(round(center_y - target_h2 / 2.0)))
        y2_2 = min(h, int(round(center_y + target_h2 / 2.0)))

        crop2 = rotated[y1_2:y2_2, x1_2:x2_2]
        if crop2.size != 0:
            crop = crop2

    return cv2.resize(crop, (out_w, out_h))


# ─── Dataset preparation ──────────────────────────────────────────────────────

def prepare_ocr_crops(
    wm_path: Path,
    dst_root: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
    out_h: int = OUT_H,
    out_w: int = OUT_W,
    roi_polygon_detector: Callable[[np.ndarray], list[tuple[float, float]] | None] | None = None,
    fallback_to_gt_polygon_on_miss: bool = False,
    roi_detector: Callable[[np.ndarray], tuple[float, float, float, float] | None] | None = None,
    fallback_to_gt_on_miss: bool = False,
    label_mode: str = LABEL_MODE_SOURCE,
    dot_zero_policy: str = DOT_ZERO_POLICY_DROP_FRACTION,
) -> None:
    """Create two WM OCR crop datasets under dst_root (idempotent).

    wm_polygon/  — WM perspective warp on ROI polygon.
                   If *roi_polygon_detector* is provided (callable: img_bgr →
                   polygon|None), it is used to get the polygon (e.g. ROI U-Net
                   mask-to-polygon inference). Otherwise falls back to GT
                   roi_polygon annotation. If detector misses and
                   *fallback_to_gt_polygon_on_miss* is True, GT roi_polygon is
                   used instead of skipping.
    wm_bbox/     — WM bbox crop with content-based rotation correction.
                   If *roi_detector* is provided (callable: img_bgr → (cx,cy,w,h)|None),
                   it is used to get the bbox (e.g. YOLO / Faster R-CNN inference).
                   Otherwise falls back to the GT polygon-derived bbox.
                   If detector misses and *fallback_to_gt_on_miss* is True,
                   GT polygon-derived bbox is used instead of skipping.
                   In both cases rotation is estimated from the crop content
                   (projection profile) — no polygon annotation needed.

        Label modes:
            - source: keep all source digits as annotated.
            - wm_fraction_aware: WM-specific decimal normalization that can infer
                hidden trailing zeros and handle '.0' using red-fraction presence.
        dot_zero_policy:
            - drop_fraction (default): map '.0' -> integer-only label.
            - preserve_source: keep '.0' as one trailing zero.
            - expand_hidden: map '.0' -> three trailing zeros.
            - auto_red: infer '.0' using red-fraction presence heuristic.

    UM is excluded — ROI detection on UM is unreliable (only 45/1552 have ROI).

    Each split: images/<stem>.png  +  labels.csv (filename, label)
    Skips images already present (safe to re-run).
    Note: delete existing wm_polygon/wm_bbox crops before re-running with a
    different detector source to force regeneration (idempotency is by filename).
    """
    from models.data.unified_loader import load_water_meter_dataset_split
    from models.data.roi_dataset import polygon_to_bbox

    if label_mode not in {LABEL_MODE_SOURCE, LABEL_MODE_WM_FRACTION_AWARE}:
        raise ValueError(
            f"Unsupported label_mode: {label_mode!r}. "
            f"Expected one of: {LABEL_MODE_SOURCE!r}, {LABEL_MODE_WM_FRACTION_AWARE!r}"
        )
    if dot_zero_policy not in DOT_ZERO_POLICIES:
        expected = ", ".join(sorted(DOT_ZERO_POLICIES))
        raise ValueError(
            f"Unsupported dot_zero_policy: {dot_zero_policy!r}. "
            f"Expected one of: {expected}"
        )

    needs_red_for_label = (
        label_mode == LABEL_MODE_WM_FRACTION_AWARE
        and dot_zero_policy == DOT_ZERO_POLICY_AUTO_RED
    )

    wm_path, dst_root = Path(wm_path), Path(dst_root)

    wm_train, wm_test = load_water_meter_dataset_split(wm_path, train_ratio, seed)

    for split_name, samples in [("train", wm_train), ("test", wm_test)]:
        valid = [
            s
            for s in samples
            if s.value is not None and (s.roi_polygon is not None or roi_polygon_detector is not None)
        ]

        def _resolve_polygon(img, s):
            if roi_polygon_detector is None:
                return s.roi_polygon

            pred_polygon = roi_polygon_detector(img)
            if pred_polygon is not None and len(pred_polygon) >= 3:
                return pred_polygon

            if fallback_to_gt_polygon_on_miss:
                return s.roi_polygon

            return None

        def _crop_polygon(img, s):
            polygon = _resolve_polygon(img, s)
            if polygon is None:
                return None
            return warp_roi_polygon(img, polygon, out_h, out_w)

        def _crop_bbox(img, s):
            gt_bbox = polygon_to_bbox(s.roi_polygon) if s.roi_polygon is not None else None
            if roi_detector is not None:
                det_bbox = roi_detector(img)
                if det_bbox is None:
                    if not fallback_to_gt_on_miss or gt_bbox is None:
                        return None      # detector missed — skip sample
                    bbox = gt_bbox
                else:
                    bbox = det_bbox
            else:
                if gt_bbox is None:
                    return None
                bbox = gt_bbox
            return crop_roi_from_detection(img, bbox, out_h, out_w)

        for crop_type, crop_fn in [
            ("wm_polygon", _crop_polygon),
            ("wm_bbox",    _crop_bbox),
        ]:
            img_dir = dst_root / crop_type / split_name / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            csv_path = dst_root / crop_type / split_name / "labels.csv"

            rows: list[dict] = []

            for s in valid:
                fname = s.image_path.stem + ".png"
                out_path = img_dir / fname

                # For fraction-aware labels we need image context (red presence);
                # for crop generation we need image only when file is missing.
                img = None
                if needs_red_for_label or not out_path.exists():
                    img = cv2.imread(str(s.image_path))
                    if img is None:
                        continue

                has_fractional_red = None
                if needs_red_for_label:
                    has_fractional_red = _estimate_fractional_red_presence(img, s.roi_polygon)

                label = sample_to_ocr_label(
                    s,
                    label_mode=label_mode,
                    has_fractional_red=has_fractional_red,
                    dot_zero_policy=dot_zero_policy,
                )
                if not label:
                    continue

                if not out_path.exists():
                    crop = crop_fn(img, s)
                    if crop is None:
                        continue        # roi_detector missed this image
                    cv2.imwrite(str(out_path), crop)

                rows.append({"filename": fname, "label": label})

            rows.sort(key=lambda r: r["filename"])
            _write_csv(csv_path, rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label"])
        w.writeheader()
        w.writerows(rows)


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_ocr_crops(crops_dir: Path, split: str) -> list[tuple[Path, str]]:
    """Return [(img_path, label_str)] from a pre-computed crop directory.

    crops_dir: e.g. DATA_ROOT / "ocr_crops" / "wm_polygon"
    split: "train" or "test"
    """
    csv_path = Path(crops_dir) / split / "labels.csv"
    img_dir  = Path(crops_dir) / split / "images"
    result   = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            result.append((img_dir / row["filename"], row["label"]))
    return result


# ─── Per-digit crops (for per_digit_classifier only) ─────────────────────────

def load_um_digit_crops(
    yolo_path: Path,
    split: str,
    crop_size: int = 32,
) -> list[tuple[np.ndarray, int]]:
    """Return (crop_bgr crop_size x crop_size, digit_class 0-9) for every digit bbox in split."""
    labels_dir = Path(yolo_path) / split / "labels"
    images_dir = Path(yolo_path) / split / "images"
    results = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / (label_file.stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5 or int(parts[0]) > 9:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = (float(p) for p in parts[1:5])
                x1 = max(0, int((cx - bw / 2) * w))
                y1 = max(0, int((cy - bh / 2) * h))
                x2 = min(w, int((cx + bw / 2) * w))
                y2 = min(h, int((cy + bh / 2) * h))
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                results.append((cv2.resize(crop, (crop_size, crop_size)), cls))
    return results
