import pytest
import cv2
import numpy as np
from pathlib import Path
import csv
import re
from types import SimpleNamespace
from models.data.ocr_dataset import (
    warp_roi_polygon,
    crop_roi_bbox,
    crop_roi_from_detection,
    estimate_rotation_from_crop,
    prepare_ocr_crops,
    load_ocr_crops,
    estimate_roi_rotation,
    sample_to_ocr_label,
    CHARSET,
)

DATA_ROOT  = Path("WaterMetricsDATA")
WM_PATH    = DATA_ROOT / "waterMeterDataset/WaterMeters"
CROPS_ROOT = DATA_ROOT / "ocr_crops"


_VALUE_IN_NAME_RE = re.compile(r"_value_(\d+)(?:_(\d+))?$")


def _expected_label_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    m = _VALUE_IN_NAME_RE.search(stem)
    if not m:
        return None
    int_part = m.group(1)
    frac_part = m.group(2) or ""
    return int_part + frac_part


class TestCropHelpers:
    def test_warp_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.4), (0.1, 0.4)]
        out = warp_roi_polygon(img, polygon, out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)

    def test_crop_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = crop_roi_bbox(img, (0.5, 0.5, 0.4, 0.2), out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)

    def test_crop_empty_bbox_returns_zeros(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = crop_roi_bbox(img, (0.5, 0.5, 0.0, 0.0), out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)
        assert out.sum() == 0


class TestPrepareOcrCrops:
    def test_creates_wm_datasets(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        for key in ("wm_polygon", "wm_bbox"):
            for split in ("train", "test"):
                assert (tmp_path / key / split / "labels.csv").exists()
                assert (tmp_path / key / split / "images").is_dir()

    def test_labels_csv_has_header(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        with open(tmp_path / "wm_polygon" / "train" / "labels.csv") as f:
            header = next(csv.reader(f))
        assert header == ["filename", "label"]

    def test_wm_labels_are_digit_strings(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        samples = load_ocr_crops(tmp_path / "wm_polygon", "train")
        for _, label in samples[:10]:
            assert label.isdigit()

    def test_wm_labels_keep_fractional_digits(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        samples = (
            load_ocr_crops(tmp_path / "wm_polygon", "train")
            + load_ocr_crops(tmp_path / "wm_polygon", "test")
        )

        matched = 0
        for img_path, label in samples:
            expected = _expected_label_from_filename(img_path.name)
            if expected is None:
                continue
            matched += 1
            assert label == expected

        assert matched > 0

    def test_rerun_rewrites_stale_labels(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)

        csv_path = tmp_path / "wm_polygon" / "train" / "labels.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows
        rows[0]["label"] = "1"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(rows)

        prepare_ocr_crops(WM_PATH, tmp_path)

        with open(csv_path, newline="", encoding="utf-8") as f:
            fixed_rows = list(csv.DictReader(f))

        expected = _expected_label_from_filename(fixed_rows[0]["filename"])
        assert expected is not None
        assert fixed_rows[0]["label"] == expected

    def test_sample_to_ocr_label_ignores_separator_for_value_text(self):
        s = SimpleNamespace(value_text="305.162", value=None)
        assert sample_to_ocr_label(s) == "305162"

        s = SimpleNamespace(value_text="00567.0", value=None)
        assert sample_to_ocr_label(s) == "005670"

        s = SimpleNamespace(value_text="12,340", value=None)
        assert sample_to_ocr_label(s) == "12340"

    def test_sample_to_ocr_label_rounds_legacy_float_fallback(self):
        s = SimpleNamespace(value_text=None, value=90.21600000000001)
        assert sample_to_ocr_label(s) == "90216"

    def test_sample_to_ocr_label_fraction_aware_pads_hidden_zero_for_two_decimals(self):
        s = SimpleNamespace(value_text="269.85", value=None)
        assert sample_to_ocr_label(s, label_mode="wm_fraction_aware") == "269850"

    def test_sample_to_ocr_label_fraction_aware_handles_single_decimal_ambiguity(self):
        s_no_fraction = SimpleNamespace(value_text="40.0", value=None)
        assert (
            sample_to_ocr_label(
                s_no_fraction,
                label_mode="wm_fraction_aware",
                has_fractional_red=False,
            )
            == "40"
        )

        s_fraction = SimpleNamespace(value_text="40.7", value=None)
        assert sample_to_ocr_label(s_fraction, label_mode="wm_fraction_aware") == "40700"

    def test_idempotent(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        n1 = len(load_ocr_crops(tmp_path / "wm_polygon", "train"))
        prepare_ocr_crops(WM_PATH, tmp_path)  # second call
        n2 = len(load_ocr_crops(tmp_path / "wm_polygon", "train"))
        assert n1 == n2


class TestLoadOcrCrops:
    def test_returns_existing_paths(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        samples = load_ocr_crops(tmp_path / "wm_polygon", "train")
        assert len(samples) > 0
        for img_path, label in samples[:5]:
            assert img_path.exists()
            assert label.isdigit()

    def test_wm_polygon_same_count_as_wm_bbox(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        n_poly = len(load_ocr_crops(tmp_path / "wm_polygon", "train"))
        n_bbox = len(load_ocr_crops(tmp_path / "wm_bbox", "train"))
        assert n_poly == n_bbox


class TestRotation:
    """Tests for _estimate_rotation and rotation-aware corner ordering."""

    def test_upright_polygon_near_zero(self):
        """Horizontal rectangle → angle ≈ 0."""
        polygon = [(0.1, 0.4), (0.9, 0.4), (0.9, 0.6), (0.1, 0.6)]
        angle = estimate_roi_rotation(polygon, (100, 200))
        assert abs(angle) < 5.0

    def test_rotated_45_degrees(self):
        """Rectangle tilted ~45° → angle ≈ 45."""
        import math
        cx, cy, hw, hh = 0.5, 0.5, 0.3, 0.05
        a = math.radians(45)
        cos_a, sin_a = math.cos(a), math.sin(a)
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            corners.append((rx, ry))
        angle = estimate_roi_rotation(corners, (100, 100))
        assert abs(angle - 45) < 5.0

    def test_rotated_minus30(self):
        """Rectangle tilted ~-30° → angle ≈ -30."""
        import math
        cx, cy, hw, hh = 0.5, 0.5, 0.3, 0.05
        a = math.radians(-30)
        cos_a, sin_a = math.cos(a), math.sin(a)
        corners = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            corners.append((rx, ry))
        angle = estimate_roi_rotation(corners, (100, 100))
        assert abs(angle - (-30)) < 5.0

    def test_vertical_rectangle_gives_large_angle(self):
        """Tall rectangle (90° rotated) → |angle| ≈ 90."""
        polygon = [(0.45, 0.1), (0.55, 0.1), (0.55, 0.9), (0.45, 0.9)]
        angle = estimate_roi_rotation(polygon, (100, 100))
        assert abs(abs(angle) - 90) < 5.0

    def test_warp_rotated_polygon_output_shape(self):
        """warp_roi_polygon returns correct shape even for rotated polygons."""
        import math
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cx, cy, hw, hh = 0.5, 0.5, 0.3, 0.05
        a = math.radians(30)
        cos_a, sin_a = math.cos(a), math.sin(a)
        polygon = []
        for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            polygon.append((cx + dx * cos_a - dy * sin_a,
                            cy + dx * sin_a + dy * cos_a))
        out = warp_roi_polygon(img, polygon, out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)

    def test_bbox_with_rotation_output_shape(self):
        """crop_roi_bbox with rotation_deg returns correct shape."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out = crop_roi_bbox(img, (0.5, 0.5, 0.4, 0.2), out_h=64, out_w=256,
                            rotation_deg=30.0)
        assert out.shape == (64, 256, 3)

    def test_bbox_zero_rotation_matches_no_rotation(self):
        """rotation_deg=0 produces same result as default."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (0.5, 0.5, 0.4, 0.2)
        out_default = crop_roi_bbox(img, bbox, out_h=64, out_w=256)
        out_zero = crop_roi_bbox(img, bbox, out_h=64, out_w=256, rotation_deg=0.0)
        np.testing.assert_array_equal(out_default, out_zero)


class TestProjectionRotation:
    """Tests for estimate_rotation_from_crop and crop_roi_from_detection."""

    def test_horizontal_band_gives_near_zero(self):
        """Synthetic image with a bright horizontal stripe → angle ≈ 0."""
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        img[40:60, :] = 255          # bright horizontal band (digits row)
        angle = estimate_rotation_from_crop(img)
        assert abs(angle) < 5.0

    def test_tilted_band_detected(self):
        """Bright band tilted ~20° → estimated angle close to ±20°."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        # Draw a tilted white rectangle
        pts = np.array([[0, 80], [400, 80+int(400*np.tan(np.radians(20)))],
                        [400, 120+int(400*np.tan(np.radians(20)))], [0, 120]],
                       dtype=np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))
        angle = estimate_rotation_from_crop(img)
        # Should detect ~20° (or -160° normalised to -20°, within some tolerance)
        assert abs(abs(angle) - 20) < 8.0

    def test_crop_roi_from_detection_crop2_then_resize_fixed_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a white horizontal band so projection profile has signal
        img[200:280, :] = 200
        out = crop_roi_from_detection(img, (0.5, 0.5, 0.6, 0.4), out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)

    def test_crop_roi_from_detection_resizes_if_too_short(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[80:120, :] = 200
        out_h, out_w = 128, 1024
        out = crop_roi_from_detection(
            img, (0.5, 0.5, 0.4, 0.2), out_h=out_h, out_w=out_w,
        )
        assert out.shape == (out_h, out_w, 3)

    def test_crop_roi_from_detection_empty_bbox_returns_zeros(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = crop_roi_from_detection(img, (0.5, 0.5, 0.0, 0.0), out_h=64, out_w=256)
        assert out.shape == (64, 256, 3)
        assert out.sum() == 0

    def test_prepare_ocr_crops_with_gt_fallback(self, tmp_path):
        """prepare_ocr_crops without roi_detector uses GT bbox + projection."""
        prepare_ocr_crops(WM_PATH, tmp_path)
        n_poly = len(load_ocr_crops(tmp_path / "wm_polygon", "train"))
        n_bbox = len(load_ocr_crops(tmp_path / "wm_bbox", "train"))
        assert n_poly > 0
        assert n_bbox > 0

    def test_prepare_ocr_crops_with_custom_detector(self, tmp_path):
        """prepare_ocr_crops with roi_detector uses it for bbox path."""
        call_count = []

        def fake_detector(img_bgr):
            call_count.append(1)
            h, w = img_bgr.shape[:2]
            return (0.5, 0.5, 0.4, 0.3)   # always return centre bbox

        prepare_ocr_crops(WM_PATH, tmp_path, roi_detector=fake_detector)
        assert len(call_count) > 0   # detector was called
        n_bbox = len(load_ocr_crops(tmp_path / "wm_bbox", "train"))
        assert n_bbox > 0

    def test_prepare_ocr_crops_detector_miss_skips_sample(self, tmp_path):
        """If roi_detector returns None, sample is skipped in wm_bbox."""
        def never_detect(img_bgr):
            return None

        prepare_ocr_crops(WM_PATH, tmp_path, roi_detector=never_detect)
        n_poly = len(load_ocr_crops(tmp_path / "wm_polygon", "train"))
        n_bbox = len(load_ocr_crops(tmp_path / "wm_bbox", "train"))
        assert n_poly > 0
        assert n_bbox == 0


class TestCharset:
    def test_charset(self):
        assert CHARSET == "0123456789"
        assert len(CHARSET) == 10
