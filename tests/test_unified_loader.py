import pytest
from pathlib import Path
from models.data.unified_loader import (
    load_water_meter_dataset,
    load_utility_meter_dataset,
    load_water_meter_dataset_split,
    UnifiedSample,
)

DATA_ROOT = Path("WaterMetricsDATA")
WM_PATH = DATA_ROOT / "waterMeterDataset" / "WaterMeters"
UM_YOLO_PATH = DATA_ROOT / "utility-meter-reading-dataset-for-automatic-reading-yolo.v4i.yolov11"


@pytest.fixture
def wm_samples():
    return load_water_meter_dataset(WM_PATH)


@pytest.fixture
def um_samples():
    return load_utility_meter_dataset(UM_YOLO_PATH, split="train")


class TestUnifiedSample:
    def test_wm_returns_list_of_unified_samples(self, wm_samples):
        assert len(wm_samples) > 0
        assert isinstance(wm_samples[0], UnifiedSample)

    def test_wm_sample_has_required_fields(self, wm_samples):
        s = wm_samples[0]
        assert s.image_path.exists()
        assert isinstance(s.value, float)
        assert s.roi_polygon is not None
        assert len(s.roi_polygon) >= 3

    def test_um_returns_list_of_unified_samples(self, um_samples):
        assert len(um_samples) > 0
        assert isinstance(um_samples[0], UnifiedSample)

    def test_um_sample_has_digit_bboxes(self, um_samples):
        s = um_samples[0]
        assert s.image_path.exists()
        assert s.digit_bboxes is not None
        assert len(s.digit_bboxes) > 0

    def test_um_value_is_none_when_not_available(self, um_samples):
        s = um_samples[0]
        assert s.value is None or isinstance(s.value, float)


class TestWaterMeterSplit:
    def test_split_returns_two_lists(self):
        train, test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        assert isinstance(train, list)
        assert isinstance(test, list)
        assert len(train) > 0
        assert len(test) > 0

    def test_split_ratio_approximately_correct(self):
        train, test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        total = len(train) + len(test)
        assert abs(len(train) / total - 0.7) < 0.01

    def test_split_is_deterministic(self):
        t1, _ = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        t2, _ = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        assert [s.image_path for s in t1] == [s.image_path for s in t2]

    def test_split_no_overlap(self):
        train, test = load_water_meter_dataset_split(WM_PATH, train_ratio=0.7, seed=42)
        train_paths = {s.image_path for s in train}
        test_paths = {s.image_path for s in test}
        assert train_paths.isdisjoint(test_paths)
