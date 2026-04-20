import pytest
import numpy as np

from models.data.augmentations_ocr import (
    available_ocr_augmentation_profiles,
    get_ocr_train_transforms,
    get_ocr_val_transforms,
)


def test_ocr_transforms_resize_shape():
    img = np.zeros((80, 300), dtype=np.uint8)
    train_t = get_ocr_train_transforms(out_h=64, out_w=256, p_rotate_180=0.0)
    val_t = get_ocr_val_transforms(out_h=64, out_w=256)

    train_out = train_t(image=img)["image"]
    val_out = val_t(image=img)["image"]

    assert train_out.shape == (64, 256)
    assert val_out.shape == (64, 256)


def test_dirty_robust_profile_resize_shape():
    img = np.zeros((96, 320), dtype=np.uint8)
    dirty_t = get_ocr_train_transforms(
        out_h=64,
        out_w=256,
        p_rotate_180=0.0,
        profile="dirty_robust",
    )
    dirty_out = dirty_t(image=img)["image"]
    assert dirty_out.shape == (64, 256)


def test_invalid_profile_raises():
    with pytest.raises(ValueError):
        get_ocr_train_transforms(profile="unknown_profile")


def test_profiles_registry_contains_dirty_robust():
    profiles = available_ocr_augmentation_profiles()
    assert "default" in profiles
    assert "dirty_robust" in profiles
