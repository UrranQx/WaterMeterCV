import pytest
from models.metrics.evaluation import (
    full_string_accuracy,
    per_digit_accuracy,
    character_error_rate,
    compute_iou_polygon,
    compute_iou_bbox,
)


class TestFullStringAccuracy:
    def test_perfect(self):
        preds = ["123.456", "789.012"]
        gts = ["123.456", "789.012"]
        assert full_string_accuracy(preds, gts) == 1.0

    def test_half_correct(self):
        preds = ["123.456", "999.999"]
        gts = ["123.456", "789.012"]
        assert full_string_accuracy(preds, gts) == 0.5

    def test_empty(self):
        assert full_string_accuracy([], []) == 0.0


class TestPerDigitAccuracy:
    def test_perfect(self):
        assert per_digit_accuracy("12345", "12345") == 1.0

    def test_one_wrong(self):
        assert per_digit_accuracy("12345", "12346") == 4 / 5

    def test_different_lengths(self):
        acc = per_digit_accuracy("123", "1234")
        assert 0.0 <= acc <= 1.0


class TestCER:
    def test_perfect(self):
        assert character_error_rate("12345", "12345") == 0.0

    def test_one_substitution(self):
        cer = character_error_rate("12345", "12346")
        assert 0.0 < cer < 1.0


class TestIoU:
    def test_identical_bbox(self):
        bbox = (0.5, 0.5, 0.2, 0.2)
        assert compute_iou_bbox(bbox, bbox) == pytest.approx(1.0)

    def test_no_overlap_bbox(self):
        a = (0.1, 0.1, 0.1, 0.1)
        b = (0.9, 0.9, 0.1, 0.1)
        assert compute_iou_bbox(a, b) == pytest.approx(0.0)

    def test_identical_polygon(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert compute_iou_polygon(poly, poly) == pytest.approx(1.0)
