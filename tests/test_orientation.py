import numpy as np
import torch

from models.utils.orientation import (
    dual_read_inference,
    orientation_aware_loss,
    orientation_aware_min_loss,
    rotate_batch_180,
    rotate_image_180,
    select_best_orientation,
)


def test_select_best_orientation_prefers_higher_confidence():
    pred, conf, angle = select_best_orientation("123", 0.6, "321", 0.9)
    assert pred == "321"
    assert conf == 0.9
    assert angle == 180


def test_select_best_orientation_tie_prefers_zero_degree():
    pred, conf, angle = select_best_orientation("123", 0.8, "321", 0.8)
    assert pred == "123"
    assert conf == 0.8
    assert angle == 0


def test_dual_read_inference_calls_predictor_for_both_orientations():
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    calls: list[np.ndarray] = []

    def predictor(img: np.ndarray):
        calls.append(img.copy())
        if len(calls) == 1:
            return "123", 0.4
        return "321", 0.95

    result = dual_read_inference(image, predictor)

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1], rotate_image_180(image))
    assert result.selected_pred == "321"
    assert result.selected_angle == 180


def test_rotate_batch_180():
    images = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    rotated = rotate_batch_180(images)
    assert rotated.shape == images.shape
    assert torch.equal(rotated[0, 0, 0, 0], images[0, 0, -1, -1])


def test_orientation_aware_min_loss():
    loss_0 = torch.tensor([1.0, 3.0], dtype=torch.float32)
    loss_180 = torch.tensor([2.0, 0.5], dtype=torch.float32)
    final_loss = orientation_aware_min_loss(loss_0, loss_180)
    assert torch.isclose(final_loss, torch.tensor(0.75))


def test_orientation_aware_loss_wrapper():
    images = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ]
    )

    def per_sample_loss(batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(1, 2, 3))

    final_loss = orientation_aware_loss(images, per_sample_loss)
    assert final_loss.ndim == 0
