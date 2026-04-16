"""Step-by-step debug visualization of the bbox crop pipeline.

Columns per row:
  [0] Original + YOLO bbox (red) + GT polygon (green)
  [1] Estimation crop (padded, for angle detection)
  [2] After 90° coarse correction (if portrait)
  [3] Projection score curve -90..+90° (used range shown in green shading)
  [4] Rotated original crop (real pixels, no artifacts)
  [5] Final resize (bbox result)
  [6] Polygon warp reference

Usage — single range:
    uv run python scripts/debug_bbox_crop.py --min-angle 10 --max-angle 50

Usage — batch (generates one image per range band):
    uv run python scripts/debug_bbox_crop.py --batch
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.data.unified_loader import load_water_meter_dataset_split
from models.data.ocr_dataset import (
    warp_roi_polygon,
    crop_roi_from_detection,
    estimate_roi_rotation,
    estimate_rotation_from_crop,
    _projection_score,
    OUT_H, OUT_W,
)

BATCH_RANGES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 90)]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_yolo(weights_path: Path):
    try:
        from ultralytics import YOLO
        m = YOLO(str(weights_path))
        def detect(img):
            r = m.predict(img, verbose=False, conf=0.001)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return None
            i = int(r.boxes.conf.argmax().item())
            b = r.boxes.xywhn[i]
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return detect
    except Exception as e:
        print(f"Warning: YOLO load failed ({e}). Using GT bbox.")
        return None


def _est_crop(img, bbox, pad=0.3):
    """Padded crop for angle estimation (mirrors crop_roi_from_detection step 1)."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    cx_px, cy_px = cx * w, cy * h
    bw_px, bh_px = bw * w, bh * h
    x1 = max(0, int(cx_px - bw_px * (1 + pad) / 2))
    y1 = max(0, int(cy_px - bh_px * (1 + pad) / 2))
    x2 = min(w, int(cx_px + bw_px * (1 + pad) / 2))
    y2 = min(h, int(cy_px + bh_px * (1 + pad) / 2))
    c = img[y1:y2, x1:x2]
    return c if c.size > 0 else np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)


def _coarse90(crop):
    ch, cw = crop.shape[:2]
    if ch <= cw * 1.2:
        return crop, 0.0, "landscape"
    cw_r  = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    ccw_r = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    g_cw  = cv2.cvtColor(cw_r,  cv2.COLOR_BGR2GRAY)
    g_ccw = cv2.cvtColor(ccw_r, cv2.COLOR_BGR2GRAY)
    if _projection_score(g_cw) >= _projection_score(g_ccw):
        return cw_r, -90.0, "portrait: 90 CW"
    return ccw_r, 90.0, "portrait: 90 CCW"


def _score_curve_full(crop, step=2.0):
    """Full +/-90 deg score curve."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    c = (w / 2.0, h / 2.0)
    angles, scores = [], []
    for a in np.arange(-90.0, 90.0 + step, step):
        M = cv2.getRotationMatrix2D(c, float(a), 1.0)
        rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        angles.append(float(a))
        scores.append(_projection_score(rot))
    return angles, scores


def _rotated_crop(img, bbox, total_angle, fine_angle):
    """Rotate original image around bbox centre and crop with adaptive padding."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    cx_px, cy_px = cx * w, cy * h
    bw_px, bh_px = bw * w, bh * h

    M = cv2.getRotationMatrix2D((cx_px, cy_px), total_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

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

    c = rotated[y1:y2, x1:x2]
    return c if c.size > 0 else np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)


def _draw_bbox(img, bbox, color, thick=3):
    v = img.copy(); h, w = v.shape[:2]
    cx, cy, bw, bh = bbox
    cv2.rectangle(v, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                     (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), color, thick)
    return v


def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def thumb(img, mw=260, mh=200):
    if img is None or img.size == 0:
        return np.zeros((64, 256, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    s = min(mw / max(w, 1), mh / max(h, 1))
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))


# ─── render one batch ─────────────────────────────────────────────────────────

def render_batch(annotated, src_label, out_path, angle_min, angle_max):
    if not annotated:
        print(f"  No samples for [{angle_min}, {angle_max}] -- skipping.")
        return

    N_COLS = 7
    n = len(annotated)
    fig, axes = plt.subplots(n, N_COLS, figsize=(N_COLS * 3.8, n * 3.8))
    if n == 1:
        axes = axes[None, :]

    titles = [
        f"0. Original\n+{src_label} bbox + poly",
        "1. Estimation crop\n(pad=0.3)",
        "2. After coarse 90\n(if portrait)",
        "3. Score curve\n-90..+90 | used range shaded",
        "4. Rotated original crop\n(real pixels, adaptive pad)",
        "5. Final resize\n(bbox result)",
        "6. Poly warp\n(reference)",
    ]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=8, fontweight="bold")

    for row, (abs_a, poly_a, img, s) in enumerate(annotated):
        from models.data.roi_dataset import polygon_to_bbox
        if _yolo is not None:
            bbox = _yolo(img)
            if bbox is None:
                bbox = polygon_to_bbox(s.roi_polygon)
        else:
            bbox = polygon_to_bbox(s.roi_polygon)

        # col 0: original + overlays
        vis0 = _draw_bbox(img, bbox, (0, 0, 255))
        h0, w0 = vis0.shape[:2]
        pts = np.array([[int(p[0] * w0), int(p[1] * h0)]
                        for p in s.roi_polygon], dtype=np.int32)
        cv2.polylines(vis0, [pts], True, (0, 255, 0), 2)
        axes[row, 0].imshow(bgr2rgb(thumb(vis0)))
        axes[row, 0].set_ylabel(f"GT={int(s.value)}\npoly_a={poly_a:+.1f}",
                                fontsize=8, rotation=0, labelpad=100, va="center")

        # col 1: estimation crop
        c1 = _est_crop(img, bbox)
        axes[row, 1].imshow(bgr2rgb(thumb(c1)))

        # col 2: coarse 90 deg
        c2, coarse_a, coarse_lbl = _coarse90(c1)
        axes[row, 2].imshow(bgr2rgb(thumb(c2)))
        axes[row, 2].set_xlabel(coarse_lbl, fontsize=7)

        # col 3: score curve (full +/-90, shaded used range)
        angles, scores = _score_curve_full(c2)
        ch2, cw2 = c2.shape[:2]
        used_range = 45.0 if (cw2 > ch2 * 1.1) else 90.0
        fine_a = estimate_rotation_from_crop(c2, angle_range=used_range)
        total_a = coarse_a + fine_a

        ax = axes[row, 3]
        ax.plot(angles, scores, "b-", lw=1.5)
        ax.axvspan(-used_range, used_range, alpha=0.12, color="green",
                   label=f"used +/-{used_range:.0f}")
        ax.axvline(fine_a, color="r", ls="--", lw=1.5,
                   label=f"fine={fine_a:.1f}")
        ax.set_xlabel("angle", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.axis("on")

        # col 4: rotated original crop (real pixels)
        c4 = _rotated_crop(img, bbox, total_a, fine_a)
        axes[row, 4].imshow(bgr2rgb(thumb(c4)))
        axes[row, 4].set_xlabel(f"total={total_a:+.1f}", fontsize=7)

        # col 5: final resize
        final = crop_roi_from_detection(img, bbox, out_h=OUT_H, out_w=OUT_W)
        axes[row, 5].imshow(bgr2rgb(final))

        # col 6: polygon warp reference
        pw = warp_roi_polygon(img, s.roi_polygon, OUT_H, OUT_W)
        axes[row, 6].imshow(bgr2rgb(pw))

        for c in [0, 1, 2, 4, 5, 6]:
            axes[row, c].axis("off")

    plt.suptitle(
        f"BBox pipeline debug | {src_label} bbox | rotate original | "
        f"|angle| [{angle_min}, {angle_max}] | n={n}",
        fontsize=12, y=1.002,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

_yolo = None   # module-level so render_batch can access it

def main():
    global _yolo
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   default=8)
    parser.add_argument("--min-angle",  type=float, default=10.0)
    parser.add_argument("--max-angle",  type=float, default=70.0)
    parser.add_argument("--batch",      action="store_true",
                        help="Generate one image per BATCH_RANGES band")
    parser.add_argument("--out", type=str,
                        default=str(ROOT / "results" / "debug_bbox_crop.png"))
    parser.add_argument("--wm-path", type=str,
                        default=str(ROOT / "WaterMetricsDATA"
                                    / "waterMeterDataset" / "WaterMeters"))
    parser.add_argument("--yolo-weights", type=str,
                        default=str(ROOT / "models" / "weights" / "roi_yolo"
                                    / "wm_yolo_roi_20260412_230832"
                                    / "weights" / "best.pt"))
    args = parser.parse_args()

    wm_path = Path(args.wm_path)
    if not wm_path.exists():
        print(f"ERROR: {wm_path} not found"); sys.exit(1)

    yolo_w = Path(args.yolo_weights)
    _yolo = _make_yolo(yolo_w) if yolo_w.exists() else None
    src = "YOLO" if _yolo else "GT"

    _, test = load_water_meter_dataset_split(wm_path, 0.7, 42)
    valid = [s for s in test if s.roi_polygon is not None and s.value is not None]

    all_ann = []
    for s in valid:
        img = cv2.imread(str(s.image_path))
        if img is None: continue
        pa = estimate_roi_rotation(s.roi_polygon, img.shape)
        all_ann.append((abs(pa), pa, img, s))
    all_ann.sort(key=lambda t: t[0], reverse=True)

    if args.batch:
        out_dir = Path(args.out).parent
        for lo, hi in BATCH_RANGES:
            band = [(a, pa, i, s) for a, pa, i, s in all_ann if lo <= a <= hi][:args.n]
            out_p = out_dir / f"debug_bbox_{lo:02d}_{hi:02d}.png"
            print(f"[{lo}--{hi}] {len(band)} samples")
            render_batch(band, src, out_p, lo, hi)
    else:
        band = [(a, pa, i, s) for a, pa, i, s in all_ann
                if args.min_angle <= a <= args.max_angle][:args.n]
        print(f"|angle| [{args.min_angle:.0f}, {args.max_angle:.0f}]: {len(band)} samples")
        render_batch(band, src, Path(args.out), args.min_angle, args.max_angle)


if __name__ == "__main__":
    main()
