import pytest
from pathlib import Path
from models.data.unified_loader import (
    load_water_meter_dataset,
    load_utility_meter_dataset,
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
