"""OCR-specific albumentations presets.

Transforms here are intended for cropped meter-reading regions.
"""

from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.data.ocr_dataset import OUT_H, OUT_W


OCR_AUGMENTATION_PROFILES = ("default", "dirty_robust")


def available_ocr_augmentation_profiles() -> tuple[str, ...]:
    """Return supported OCR augmentation profile names."""
    return OCR_AUGMENTATION_PROFILES


def _default_train_transforms(
    out_h: int,
    out_w: int,
    p_rotate_180: float,
) -> list[A.BasicTransform]:
    return [
        A.Resize(height=out_h, width=out_w),
        A.Rotate(
            limit=(180, 180),
            border_mode=cv2.BORDER_REPLICATE,
            interpolation=cv2.INTER_LINEAR,
            p=p_rotate_180,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            ],
            p=0.25,
        ),
        A.RandomBrightnessContrast(p=0.25),
        A.Perspective(scale=(0.02, 0.06), keep_size=True, p=0.15),
    ]


def _dirty_robust_train_transforms(
    out_h: int,
    out_w: int,
    p_rotate_180: float,
) -> list[A.BasicTransform]:
    # Stronger synthetic corruption profile for dirty/aged meter surfaces.
    return [
        A.Resize(height=out_h, width=out_w),
        A.Rotate(
            limit=(180, 180),
            border_mode=cv2.BORDER_REPLICATE,
            interpolation=cv2.INTER_LINEAR,
            p=p_rotate_180,
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.6, 2.5), p=1.0),
                A.GaussNoise(std_range=(0.04, 0.16), p=1.0),
            ],
            p=0.45,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.35, 0.25),
                    contrast_limit=(-0.35, 0.35),
                    p=1.0,
                ),
                A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(70, 150), p=1.0),
            ],
            p=0.55,
        ),
        A.ImageCompression(
            compression_type="jpeg",
            quality_range=(35, 95),
            p=0.30,
        ),
        A.Downscale(
            scale_range=(0.35, 0.75),
            interpolation_pair={
                "downscale": cv2.INTER_AREA,
                "upscale": cv2.INTER_LINEAR,
            },
            p=0.25,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.04, 0.22),
            hole_width_range=(0.02, 0.12),
            fill="random_uniform",
            p=0.30,
        ),
        A.Perspective(
            scale=(0.02, 0.08),
            keep_size=True,
            p=0.20,
        ),
    ]


def get_ocr_train_transforms(
    out_h: int = OUT_H,
    out_w: int = OUT_W,
    p_rotate_180: float = 0.5,
    profile: str = "default",
    to_tensor: bool = False,
) -> A.Compose:
    """Train-time transforms for OCR crops.

    Profiles:
      - default: mild augmentations for regular conditions.
      - dirty_robust: stronger corruption profile for dirty/noisy meters.
    """
    if profile not in OCR_AUGMENTATION_PROFILES:
        raise ValueError(
            f"Unsupported OCR augmentation profile: {profile!r}. "
            f"Use one of {OCR_AUGMENTATION_PROFILES}."
        )

    if profile == "dirty_robust":
        transforms = _dirty_robust_train_transforms(out_h, out_w, p_rotate_180)
    else:
        transforms = _default_train_transforms(out_h, out_w, p_rotate_180)

    if to_tensor:
        transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_ocr_val_transforms(
    out_h: int = OUT_H,
    out_w: int = OUT_W,
    to_tensor: bool = False,
) -> A.Compose:
    """Validation/inference transforms for OCR crops."""
    transforms: list[A.BasicTransform] = [A.Resize(height=out_h, width=out_w)]
    if to_tensor:
        transforms.append(ToTensorV2())
    return A.Compose(transforms)


__all__ = [
    "OCR_AUGMENTATION_PROFILES",
    "available_ocr_augmentation_profiles",
    "get_ocr_train_transforms",
    "get_ocr_val_transforms",
]
