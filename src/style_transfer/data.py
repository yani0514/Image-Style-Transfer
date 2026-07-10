from __future__ import annotations

from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from PIL import Image

from .io import list_images, pil_to_tensor


class ImageFolderDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.paths = list_images(self.root, limit=limit)
        self.image_size = image_size
        if not self.paths:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_training_image(self.paths[index], self.image_size)


class ImagePathDataset(Dataset[torch.Tensor]):
    """Training dataset backed by image paths supplied by an external catalog."""

    def __init__(self, paths: list[str | Path], image_size: int = 256) -> None:
        self.paths = [Path(path) for path in paths]
        self.image_size = image_size
        if not self.paths:
            raise ValueError("The image path dataset is empty")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_training_image(self.paths[index], self.image_size)


def load_training_image(path: str | Path, image_size: int) -> torch.Tensor:
    """Resize then randomly crop/flip, avoiding the same center crop every epoch."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize(
        (max(image_size, round(width * scale)), max(image_size, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = random.randint(0, resized.width - image_size)
    top = random.randint(0, resized.height - image_size)
    image = resized.crop((left, top, left + image_size, top + image_size))
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return pil_to_tensor(image)[0]


def load_coco_paths(
    split: str = "train",
    max_samples: int | None = None,
    dataset_dir: str | Path | None = None,
) -> list[str]:
    """Download COCO with FiftyOne without requiring its MongoDB catalog."""
    try:
        from fiftyone.utils.coco import download_coco_dataset_split
    except ImportError as exc:
        raise RuntimeError(
            "COCO training requires FiftyOne. Install project dependencies first."
        ) from exc

    requested_root = Path(dataset_dir).expanduser() if dataset_dir else None
    default_root = Path.home() / "fiftyone" / "coco-2017"

    # load_zoo_dataset() used by earlier versions downloaded here before trying
    # (and potentially failing) to start MongoDB. Reuse that completed download.
    roots = [root for root in (requested_root, default_root) if root is not None]
    base_root = next(
        (root for root in roots if (root / split / "data").is_dir()),
        requested_root or default_root,
    )
    split_root = base_root / split
    images_dir = split_root / "data"
    existing = list_images(images_dir)

    if max_samples is None or len(existing) < max_samples:
        download_coco_dataset_split(
            dataset_dir=str(split_root),
            split=split,
            year="2017",
            shuffle=True,
            seed=42,
            max_samples=max_samples,
            raw_dir=str(base_root / "raw"),
            scratch_dir=str(base_root / "tmp-download"),
        )
        existing = list_images(images_dir)

    paths = existing[:max_samples] if max_samples is not None else existing
    if not paths:
        raise RuntimeError(f"FiftyOne downloaded no COCO images to {images_dir}")
    return [str(path) for path in paths]


def cycling_batches(loader):
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError(
                "DataLoader produced no batches. Try reducing --batch-size or "
                "checking that the image folders are not empty."
            )
