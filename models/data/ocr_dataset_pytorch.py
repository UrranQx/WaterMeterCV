"""PyTorch dataset helpers for OCR crops.

This module wraps precomputed OCR crops produced by
``models.data.ocr_dataset.prepare_ocr_crops`` and provides utilities for
CTC-style training/evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.data.ocr_dataset import CHARSET, OUT_H, OUT_W, load_ocr_crops


def encode_ocr_text(text: str, charset: str = CHARSET) -> torch.Tensor:
    """Encode a digit string into class indices tensor."""
    char_to_idx = {ch: i for i, ch in enumerate(charset)}
    try:
        indices = [char_to_idx[ch] for ch in text]
    except KeyError as exc:
        raise ValueError(f"Unsupported character in OCR text: {exc.args[0]!r}") from exc
    return torch.tensor(indices, dtype=torch.long)


def decode_ocr_indices(
    indices: Sequence[int] | torch.Tensor,
    charset: str = CHARSET,
    blank_index: int | None = None,
) -> str:
    """Decode class indices into text.

    If ``blank_index`` is provided, indices equal to it are skipped.
    """
    chars: list[str] = []
    for raw_idx in indices:
        idx = int(raw_idx)
        if blank_index is not None and idx == blank_index:
            continue
        if idx < 0 or idx >= len(charset):
            raise ValueError(f"Index {idx} is out of charset bounds")
        chars.append(charset[idx])
    return "".join(chars)


def _to_tensor(image: np.ndarray | torch.Tensor, grayscale: bool) -> torch.Tensor:
    """Convert image output from cv2/albumentations to float tensor CHW."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            tensor = torch.from_numpy(image).unsqueeze(0)
        elif image.ndim == 3:
            tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        else:
            raise ValueError(f"Unsupported image ndim: {image.ndim}")
        tensor = tensor.float()
    elif torch.is_tensor(image):
        tensor = image.float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if grayscale and tensor.ndim == 3 and tensor.shape[0] != 1:
        tensor = tensor.mean(dim=0, keepdim=True)

    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    return tensor


class OCRCropDataset(Dataset):
    """Dataset over persistent OCR crops for model training/inference."""

    def __init__(
        self,
        crops_dir: Path,
        split: str,
        transform=None,
        out_h: int = OUT_H,
        out_w: int = OUT_W,
        grayscale: bool = True,
    ) -> None:
        self.samples = load_ocr_crops(crops_dir, split)
        self.transform = transform
        self.out_h = out_h
        self.out_w = out_w
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, Path]:
        img_path, text = self.samples[idx]

        if self.grayscale:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Failed to read OCR crop: {img_path}")

        if not self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        image_tensor = _to_tensor(image, grayscale=self.grayscale)
        target_tensor = encode_ocr_text(text)
        return image_tensor, target_tensor, text, img_path


def ctc_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, Path]],
) -> dict[str, torch.Tensor | list[str] | list[Path]]:
    """Collate batch for CTC training.

    Returns:
      - images: Tensor[B, C, H, W]
      - targets: concatenated target indices Tensor[sum(target_lengths)]
      - target_lengths: Tensor[B]
      - texts: list[str]
      - paths: list[Path]
    """
    images, targets, texts, paths = zip(*batch)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    total_targets = int(target_lengths.sum().item())
    if total_targets == 0:
        concat_targets = torch.empty((0,), dtype=torch.long)
    else:
        concat_targets = torch.cat(list(targets), dim=0)

    return {
        "images": torch.stack(list(images), dim=0),
        "targets": concat_targets,
        "target_lengths": target_lengths,
        "texts": list(texts),
        "paths": list(paths),
    }


__all__ = [
    "OCRCropDataset",
    "ctc_collate_fn",
    "encode_ocr_text",
    "decode_ocr_indices",
]
