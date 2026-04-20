import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from models.data.ocr_dataset_pytorch import (
    OCRCropDataset,
    ctc_collate_fn,
    decode_ocr_indices,
    encode_ocr_text,
)


def _create_tiny_ocr_crops(root: Path) -> Path:
    split_dir = root / "train"
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True)

    img_a = np.zeros((64, 256), dtype=np.uint8)
    img_b = np.full((64, 256), 200, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "a.png"), img_a)
    cv2.imwrite(str(img_dir / "b.png"), img_b)

    with open(split_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerow({"filename": "a.png", "label": "123"})
        writer.writerow({"filename": "b.png", "label": "90"})

    return root


def test_encode_decode_roundtrip():
    text = "0123456789"
    encoded = encode_ocr_text(text)
    decoded = decode_ocr_indices(encoded)
    assert decoded == text


def test_dataset_item_shape_and_types(tmp_path):
    crops_dir = _create_tiny_ocr_crops(tmp_path / "wm_polygon")
    ds = OCRCropDataset(crops_dir=crops_dir, split="train")

    image, target, text, path = ds[0]

    assert image.shape == (1, 64, 256)
    assert image.dtype == torch.float32
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert target.dtype == torch.long
    assert text == "123"
    assert path.name == "a.png"


def test_ctc_collate_fn(tmp_path):
    crops_dir = _create_tiny_ocr_crops(tmp_path / "wm_polygon")
    ds = OCRCropDataset(crops_dir=crops_dir, split="train")

    batch = [ds[0], ds[1]]
    collated = ctc_collate_fn(batch)

    assert collated["images"].shape == (2, 1, 64, 256)
    assert collated["targets"].dtype == torch.long
    assert torch.equal(collated["target_lengths"], torch.tensor([3, 2], dtype=torch.long))
    assert collated["texts"] == ["123", "90"]
