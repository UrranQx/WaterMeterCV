import pytest
from pathlib import Path
from models.data.roi_dataset import (
    polygon_to_bbox,
    filter_utility_meter_roi_samples,
    prepare_yolo_roi_dataset,
)

DATA_ROOT = Path("WaterMetricsDATA")
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"


class TestPolygonToBbox:
    def test_square_polygon(self):
        polygon = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
        cx, cy, w, h = polygon_to_bbox(polygon)
        assert abs(cx - 0.2) < 1e-6
        assert abs(cy - 0.2) < 1e-6
        assert abs(w - 0.2) < 1e-6
        assert abs(h - 0.2) < 1e-6

    def test_triangle_polygon(self):
        polygon = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        cx, cy, w, h = polygon_to_bbox(polygon)
        assert abs(cx - 0.5) < 1e-6
        assert abs(cy - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6


class TestFilterUtilityMeterRoi:
    def test_returns_only_images_with_roi(self):
        samples = filter_utility_meter_roi_samples(UM_YOLO_PATH, "train")
        assert len(samples) > 0
        for img_path, bbox in samples:
            assert img_path.exists()
            assert len(bbox) == 4
            cx, cy, w, h = bbox
            assert 0 <= cx <= 1 and 0 <= cy <= 1

    def test_train_has_45_roi_images(self):
        samples = filter_utility_meter_roi_samples(UM_YOLO_PATH, "train")
        assert len(samples) == 45


class TestPrepareYoloRoiDataset:
    def test_creates_single_class_dataset(self, tmp_path):
        dst = tmp_path / "roi_yolo"
        prepare_yolo_roi_dataset(UM_YOLO_PATH, dst)
        assert (dst / "data.yaml").exists()
        label_files = list((dst / "train" / "labels").glob("*.txt"))
        assert len(label_files) > 0
        with open(label_files[0]) as f:
            for line in f:
                parts = line.strip().split()
                assert parts[0] == "0"
