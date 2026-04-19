"""Unit tests for OCR heuristics — no YOLO / model dependencies."""
from __future__ import annotations

import pytest

from watermetercv.ocr.heuristics import (
    apply_max_digits_heuristic,
    apply_ultralytics_last_drum_heuristic,
    apply_ultralytics_overlap_heuristic,
    digits_only,
    is_long_tail_zero_pattern,
    is_no_red_upside_down_pattern,
    leading_zero_count,
    safe_mean,
    trailing_zero_count,
    ultralytics_digit_rank,
)


def _box(digit: int, cx: float, cy: float, w: float = 20.0, h: float = 30.0, conf: float = 0.9) -> dict:
    return {
        "digit": int(digit),
        "conf": float(conf),
        "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2,
        "cx": cx, "cy": cy, "w": w, "h": h,
    }


class TestScalarHeuristics:
    def test_digits_only_strips_non_digits(self):
        assert digits_only("12.34a") == "1234"

    def test_digit_rank_zero_ranks_highest(self):
        assert ultralytics_digit_rank(0) > ultralytics_digit_rank(9)

    def test_leading_and_trailing_zeros(self):
        assert leading_zero_count("00482") == 2
        assert trailing_zero_count("12300") == 2

    def test_long_tail_zero_pattern(self):
        assert is_long_tail_zero_pattern("12345600")
        assert not is_long_tail_zero_pattern("1234560")  # only 7 digits

    def test_no_red_upside_down_pattern(self):
        assert is_no_red_upside_down_pattern("12340")
        assert not is_no_red_upside_down_pattern("01234")
        assert not is_no_red_upside_down_pattern("12345678")  # too many digits

    def test_safe_mean_empty(self):
        assert safe_mean([]) == 0.0
        assert safe_mean([0.5, 0.3]) == pytest.approx(0.4)


class TestMaxDigitsHeuristic:
    def test_pass_through_short(self):
        assert apply_max_digits_heuristic("1234", 0.9) == ("1234", 0.9)

    def test_truncate_and_penalize_overflow(self):
        pred, conf = apply_max_digits_heuristic("1234567890123", 1.0)
        assert pred == "1234567890"  # MAX_READING_DIGITS = 10
        assert 0.0 < conf < 1.0

    def test_strips_non_digits(self):
        pred, _ = apply_max_digits_heuristic("1.2a3", 0.9)
        assert pred == "123"


class TestOverlapHeuristic:
    def test_noop_below_two_boxes(self):
        dets, info = apply_ultralytics_overlap_heuristic([])
        assert info["applied"] is False
        assert dets == []

    def test_nested_boxes_collapse_to_one(self):
        inner = _box(5, cx=100, cy=50, w=10, h=20, conf=0.95)
        outer = _box(3, cx=100, cy=50, w=12, h=22, conf=0.5)
        dets, info = apply_ultralytics_overlap_heuristic([inner, outer])
        assert info["applied"] is True
        assert len(dets) == 1

    def test_zero_prefers_over_nonzero_in_nested_group(self):
        five = _box(5, cx=100, cy=50, w=10, h=20, conf=0.95)
        zero = _box(0, cx=100, cy=50, w=11, h=21, conf=0.5)
        dets, _ = apply_ultralytics_overlap_heuristic([five, zero])
        assert len(dets) == 1
        assert dets[0]["digit"] == 0


class TestLastDrumHeuristic:
    def test_noop_when_not_vertical_pair(self):
        dets = [_box(1, cx=10, cy=50), _box(2, cx=30, cy=50), _box(3, cx=50, cy=50)]
        out, info = apply_ultralytics_last_drum_heuristic(dets)
        assert info["applied"] is False
        assert len(out) == 3

    def test_collapses_vertical_pair_at_end(self):
        # last two boxes at same cx, split vertically
        dets = [
            _box(1, cx=10, cy=50),
            _box(9, cx=50, cy=40, w=10, h=20),
            _box(0, cx=50, cy=65, w=10, h=20),  # drum rollover: 0 > 9
        ]
        out, info = apply_ultralytics_last_drum_heuristic(dets)
        assert info["applied"] is True
        assert len(out) == 2
        assert out[-1]["digit"] == 0
