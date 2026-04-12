"""Shared ROI detection helpers for all 02_roi_detection notebooks."""
from pathlib import Path
import shutil
import yaml


def polygon_to_bbox(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Convert normalized polygon [(x, y), ...] to (cx, cy, w, h) normalized bbox."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2
    cy = y_min + h / 2
    return (cx, cy, w, h)


def filter_utility_meter_roi_samples(
    dataset_path: Path,
    split: str,
) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """Return (image_path, roi_bbox) for images that have class 10 (ROI).

    roi_bbox is (cx, cy, w, h) normalized.
    """
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / split / "labels"
    images_dir = dataset_path / split / "images"
    results = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 10:
                    bbox = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
                    for ext in (".jpg", ".jpeg", ".png"):
                        img_path = images_dir / (label_file.stem + ext)
                        if img_path.exists():
                            results.append((img_path, bbox))
                            break
                    break  # one ROI per image
    return results


def prepare_yolo_roi_dataset(src_path: Path, dst_path: Path) -> None:
    """Create single-class YOLO dataset: filter class 10 -> class 0, copy images.

    Writes data.yaml with nc=1, names=["ROI"].
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    for split in ["train", "valid", "test"]:
        (dst_path / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / split / "labels").mkdir(parents=True, exist_ok=True)

        src_labels = src_path / split / "labels"
        src_images = src_path / split / "images"

        for label_file in src_labels.glob("*.txt"):
            roi_lines = []
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == 10:
                        roi_lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

            if roi_lines:
                for ext in (".jpg", ".jpeg", ".png"):
                    src_img = src_images / (label_file.stem + ext)
                    if src_img.exists():
                        shutil.copy(src_img, dst_path / split / "images" / src_img.name)
                        break
                with open(dst_path / split / "labels" / label_file.name, "w") as f:
                    f.write("\n".join(roi_lines))

    data_yaml = {
        "path": str(dst_path),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["ROI"],
    }
    with open(dst_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)


def prepare_wm_yolo_roi_dataset(
    train_samples: list,
    test_samples: list,
    dst_path: Path,
) -> None:
    """Convert waterMeterDataset samples to YOLO single-class ROI dataset.

    Each sample must have roi_polygon. Polygon is converted to bbox.
    Creates train/ and test/ splits (test is also used as val during training).
    """
    dst_path = Path(dst_path)

    for split_name, samples in [("train", train_samples), ("test", test_samples)]:
        (dst_path / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dst_path / split_name / "labels").mkdir(parents=True, exist_ok=True)

        for sample in samples:
            if sample.roi_polygon is None:
                continue
            cx, cy, w, h = polygon_to_bbox(sample.roi_polygon)
            img_name = sample.image_path.name
            stem = sample.image_path.stem

            dst_img = dst_path / split_name / "images" / img_name
            if not dst_img.exists():
                shutil.copy(sample.image_path, dst_img)

            with open(dst_path / split_name / "labels" / f"{stem}.txt", "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = {
        "path": str(dst_path),
        "train": "train/images",
        "val": "test/images",
        "test": "test/images",
        "nc": 1,
        "names": ["ROI"],
    }
    with open(dst_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
