import pytest
import numpy as np
from pathlib import Path
from models.data.ocr_dataset import (
    warp_roi_polygon,
    crop_roi_bbox,
    prepare_ocr_crops,
    load_ocr_crops,
    CHARSET,
)

DATA_ROOT  = Path("WaterMetricsDATA")
WM_PATH    = DATA_ROOT / "waterMeterDataset/WaterMeters"
CROPS_ROOT = DATA_ROOT / "ocr_crops"


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
        import csv
        with open(tmp_path / "wm_polygon" / "train" / "labels.csv") as f:
            header = next(csv.reader(f))
        assert header == ["filename", "label"]

    def test_wm_labels_are_digit_strings(self, tmp_path):
        prepare_ocr_crops(WM_PATH, tmp_path)
        samples = load_ocr_crops(tmp_path / "wm_polygon", "train")
        for _, label in samples[:10]:
            assert label.isdigit()

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


class TestCharset:
    def test_charset(self):
        assert CHARSET == "0123456789"
        assert len(CHARSET) == 10
