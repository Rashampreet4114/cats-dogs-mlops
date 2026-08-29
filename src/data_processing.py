"""Data preprocessing, augmentation, and train/val/test splitting for the
Cats vs Dogs classifier.

Expected raw layout (produced by scripts/download_data.py):
    data/raw/cat/*.jpg
    data/raw/dog/*.jpg

Produces:
    data/processed/{train,val,test}/{cat,dog}/*.jpg
"""

import random
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

IMG_SIZE = 224
CLASSES = ["cat", "dog"]

# Standard ImageNet normalization stats — fine for a from-scratch CNN too,
# keeps preprocessing consistent with widely-used pretrained backbones.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize/normalize a single PIL image into a model-ready tensor.

    Shared by training (eval branch), the inference API, and unit tests —
    this is the one place that defines "what the model expects as input".
    """
    image = image.convert("RGB")
    tensor = get_eval_transforms()(image)
    return tensor


def create_splits(
    raw_dir: str,
    processed_dir: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split data/raw/{cat,dog}/* into processed train/val/test folders (80/10/10 by default)."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    rng = random.Random(seed)

    counts = {"train": 0, "val": 0, "test": 0}
    for cls in CLASSES:
        files = sorted((raw_dir / cls).glob("*"))
        files = [f for f in files if f.is_file()]
        rng.shuffle(files)

        n = len(files)
        n_val = int(n * val_size)
        n_test = int(n * test_size)
        n_train = n - n_val - n_test

        split_files = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }

        for split, split_list in split_files.items():
            out_dir = processed_dir / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_list:
                shutil.copy2(f, out_dir / f.name)
            counts[split] += len(split_list)

    return counts


def load_split_dataset(processed_dir: str, split: str, augment: bool = False):
    """Return a torchvision ImageFolder dataset for one split ('train'/'val'/'test')."""
    transform = get_train_transforms() if augment else get_eval_transforms()
    split_path = Path(processed_dir) / split
    return datasets.ImageFolder(root=str(split_path), transform=transform)
