"""Orientation helpers for OCR.

The ROI crop stage intentionally leaves the 0°/180° ambiguity unresolved.
This module provides utilities to handle that ambiguity during training and
inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class DualOrientationResult:
    """Inference results for 0° and 180° passes and selected output."""

    pred_0: str
    conf_0: float
    pred_180: str
    conf_180: float
    selected_pred: str
    selected_conf: float
    selected_angle: int


def rotate_image_180(image: np.ndarray) -> np.ndarray:
    """Return a 180° rotated image."""
    if image.ndim < 2:
        raise ValueError("Image must have at least 2 dimensions")
    return cv2.rotate(image, cv2.ROTATE_180)


def select_best_orientation(
    pred_0: str,
    conf_0: float,
    pred_180: str,
    conf_180: float,
) -> tuple[str, float, int]:
    """Pick orientation with higher confidence.

    On equal confidence, 0° is preferred for deterministic behavior.
    """
    if conf_180 > conf_0:
        return pred_180, float(conf_180), 180
    return pred_0, float(conf_0), 0


def dual_read_inference(
    image: np.ndarray,
    predictor: Callable[[np.ndarray], tuple[str, float]],
) -> DualOrientationResult:
    """Run OCR predictor on 0° and 180° versions and choose best output."""
    pred_0, conf_0 = predictor(image)
    pred_180, conf_180 = predictor(rotate_image_180(image))
    selected_pred, selected_conf, selected_angle = select_best_orientation(
        pred_0,
        conf_0,
        pred_180,
        conf_180,
    )
    return DualOrientationResult(
        pred_0=pred_0,
        conf_0=float(conf_0),
        pred_180=pred_180,
        conf_180=float(conf_180),
        selected_pred=selected_pred,
        selected_conf=float(selected_conf),
        selected_angle=selected_angle,
    )


def rotate_batch_180(images: torch.Tensor) -> torch.Tensor:
    """Rotate a BCHW/CHW tensor batch by 180° on spatial dims."""
    if images.ndim < 3:
        raise ValueError("Expected CHW or BCHW tensor")
    return torch.rot90(images, k=2, dims=(-2, -1))


def orientation_aware_min_loss(loss_0: torch.Tensor, loss_180: torch.Tensor) -> torch.Tensor:
    """Compute scalar min-orientation loss.

    Expects per-sample losses with equal shape. Returns mean(min(loss_0, loss_180)).
    """
    if loss_0.shape != loss_180.shape:
        raise ValueError("loss_0 and loss_180 must have the same shape")
    return torch.minimum(loss_0, loss_180).mean()


def orientation_aware_loss(
    images: torch.Tensor,
    per_sample_loss_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply 0°/180° training strategy and return min-loss objective.

    ``per_sample_loss_fn`` must return per-sample losses for the given batch.
    """
    loss_0 = per_sample_loss_fn(images)
    loss_180 = per_sample_loss_fn(rotate_batch_180(images))
    return orientation_aware_min_loss(loss_0, loss_180)


__all__ = [
    "DualOrientationResult",
    "rotate_image_180",
    "select_best_orientation",
    "dual_read_inference",
    "rotate_batch_180",
    "orientation_aware_min_loss",
    "orientation_aware_loss",
]
