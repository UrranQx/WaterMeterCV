from dataclasses import dataclass
from pathlib import Path
import csv
import ast
import random


@dataclass
class UnifiedSample:
    """Unified representation of a single dataset sample."""
    image_path: Path
    value: float | None = None
    roi_polygon: list[tuple[float, float]] | None = None  # [(x, y), ...] normalized
    roi_bbox: tuple[float, float, float, float] | None = None  # (cx, cy, w, h) normalized
    digit_bboxes: list[tuple[int, float, float, float, float]] | None = None  # [(class_id, cx, cy, w, h), ...]
    mask_path: Path | None = None
    dataset_source: str = ""


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
            value = round(float(row["value"]), 3)

            location = ast.literal_eval(row["location"])
            polygon = [(pt["x"], pt["y"]) for pt in location["data"]]

            image_path = images_dir / photo_name
            mask_path = masks_dir / photo_name
            if not mask_path.exists():
                mask_path = None

            samples.append(UnifiedSample(
                image_path=image_path,
                value=value,
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
        if digit_bboxes:
            sorted_digits = sorted(digit_bboxes, key=lambda d: d[1])
            value_str = "".join(str(d[0]) for d in sorted_digits)
            try:
                value = float(value_str)
            except ValueError:
                value = None

        samples.append(UnifiedSample(
            image_path=img_path,
            value=value,
            roi_bbox=roi_bbox,
            digit_bboxes=digit_bboxes if digit_bboxes else None,
            dataset_source="utility_meter",
        ))

    return samples
