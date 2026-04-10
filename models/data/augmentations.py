import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.HorizontalFlip(p=0.0),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))


def get_val_transforms(img_size: int = 640) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
