from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .io import list_images, load_image


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
        return load_image(self.paths[index], image_size=self.image_size)[0]


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
