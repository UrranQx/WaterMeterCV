from dataclasses import dataclass
from pathlib import Path
import csv
import ast
import random
import re
from decimal import Decimal, InvalidOperation


@dataclass
class UnifiedSample:
    """Unified representation of a single dataset sample."""
    image_path: Path
    value: float | None = None
    value_text: str | None = None
    roi_polygon: list[tuple[float, float]] | None = None  # [(x, y), ...] normalized
    roi_bbox: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    digit_bboxes: list[tuple[int, float, float, float, float]] | None = None  # [(class_id, cx, cy, w, h), ...]
    mask_path: Path | None = None
    dataset_source: str = ""


_PHOTO_VALUE_RE = re.compile(r"_value_(\d+(?:_\d+)?)$")


def _extract_value_text_from_photo_name(photo_name: str) -> str | None:
    """Extract canonical value text from WM filename (e.g. ..._value_595_825.jpg)."""
    stem = Path(photo_name).stem
    match = _PHOTO_VALUE_RE.search(stem)
    if not match:
        return None
    return _normalize_value_text(match.group(1).replace("_", "."))


def _normalize_value_text(raw_value: str) -> str | None:
    """Normalize numeric text while preserving all source digits.

    Main OCR labels are compared as digits-only strings, so we keep fractional
    digits exactly as provided by source annotations and only normalize
    separators / scientific notation formatting.
    """
    if raw_value is None:
        return None
    text = str(raw_value).strip().replace(",", ".")
    if not text:
        return None

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return text

    try:
        dec = Decimal(text)
    except InvalidOperation:
        return None
    return format(dec, "f")


def load_water_meter_dataset(root: Path) -> list[UnifiedSample]:
    """Load waterMeterDataset from data.csv.

    CSV columns: photo_name, value, location (polygon as Python dict string).
    """
    root = Path(root)
    csv_path = root / "data.csv"
    images_dir = root / "images"
    masks_dir = root / "masks"
    samples = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            photo_name = row["photo_name"]
            value_text = _extract_value_text_from_photo_name(photo_name)
            if value_text is None:
                value_text = _normalize_value_text(row.get("value", ""))

            value = None
            if value_text is not None:
                try:
                    value = float(value_text)
                except ValueError:
                    value = None

            location = ast.literal_eval(row["location"])
            polygon = [(pt["x"], pt["y"]) for pt in location["data"]]

            image_path = images_dir / photo_name
            mask_path = masks_dir / photo_name
            if not mask_path.exists():
                mask_path = None

            samples.append(UnifiedSample(
                image_path=image_path,
                value=value,
                value_text=value_text,
                roi_polygon=polygon,
                mask_path=mask_path,
                dataset_source="water_meter_dataset",
            ))

    return samples


def load_water_meter_dataset_split(
    root: Path,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[UnifiedSample], list[UnifiedSample]]:
    """Deterministic train/test split of waterMeterDataset.

    Returns (train_samples, test_samples).
    """
    all_samples = load_water_meter_dataset(root)
    shuffled = all_samples.copy()
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def load_utility_meter_dataset(root: Path, split: str = "train") -> list[UnifiedSample]:
    """Load utility-meter dataset from YOLO format labels.

    YOLO class IDs: 0-9 = digits, 10 = Reading Digit (ROI), 11-13 = colors
    """
    root = Path(root)
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    samples = []

    if not images_dir.exists():
        return samples

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        digit_bboxes = []
        roi_bbox = None

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    if cls_id == 10:
                        roi_bbox = (cx, cy, w, h)
                    elif cls_id <= 9:
                        digit_bboxes.append((cls_id, cx, cy, w, h))

        value = None
        value_text = None
        if digit_bboxes:
            sorted_digits = sorted(digit_bboxes, key=lambda d: d[1])
            value_str = "".join(str(d[0]) for d in sorted_digits)
            value_text = value_str
            try:
                value = float(value_str)
            except ValueError:
                value = None

        samples.append(UnifiedSample(
            image_path=img_path,
            value=value,
            value_text=value_text,
            roi_bbox=roi_bbox,
            digit_bboxes=digit_bboxes if digit_bboxes else None,
            dataset_source="utility_meter",
        ))

    return samples
