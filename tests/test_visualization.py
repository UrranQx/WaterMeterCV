import pytest
import numpy as np
from models.utils.visualization import draw_digit_bboxes, draw_roi_polygon


class TestVisualization:
    @pytest.fixture
    def dummy_image(self):
        return np.zeros((640, 640, 3), dtype=np.uint8)

    def test_draw_digit_bboxes_returns_image(self, dummy_image):
        bboxes = [(5, 0.3, 0.3, 0.05, 0.08), (3, 0.4, 0.3, 0.05, 0.08)]
        result = draw_digit_bboxes(dummy_image, bboxes)
        assert result.shape == dummy_image.shape

    def test_draw_roi_polygon_returns_image(self, dummy_image):
        polygon = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        result = draw_roi_polygon(dummy_image, polygon)
        assert result.shape == dummy_image.shape
