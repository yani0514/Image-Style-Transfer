from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(path)
    images = sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:limit] if limit is not None else images


def pil_to_tensor(image: Image.Image, device: torch.device | str | None = None) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(height, width, 3).permute(2, 0, 1).float().div(255.0)
    tensor = data.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    image = image.mul(255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode="RGB")


def resize_image(image: Image.Image, image_size: int | None, keep_aspect_ratio: bool = True) -> Image.Image:
    if image_size is None:
        return image.convert("RGB")
    width, height = image.size
    if keep_aspect_ratio:
        if width <= height:
            new_width = image_size
            new_height = round(height * image_size / width)
        else:
            new_height = image_size
            new_width = round(width * image_size / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left = max(0, (new_width - image_size) // 2)
        top = max(0, (new_height - image_size) // 2)
        return image.crop((left, top, left + image_size, top + image_size)).convert("RGB")
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS).convert("RGB")


def load_image(
    path: str | Path,
    image_size: int | None = None,
    device: torch.device | str | None = None,
    keep_aspect_ratio: bool = True,
) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = resize_image(image, image_size=image_size, keep_aspect_ratio=keep_aspect_ratio)
    return pil_to_tensor(image, device=device)


def save_image(tensor: torch.Tensor, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(output_path)
    return output_path


def match_spatial_size(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if source.shape[-2:] == reference.shape[-2:]:
        return source
    return torch.nn.functional.interpolate(
        source,
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def find_pair_output(
    output_dir: str | Path,
    content_stem: str,
    style_stem: str,
) -> Path | None:
    root = Path(output_dir)
    candidates = sorted(root.glob(f"{content_stem}__{style_stem}*.png"))
    candidates += sorted(root.glob(f"{content_stem}__{style_stem}*.jpg"))
    candidates += sorted(root.glob(f"{content_stem}__{style_stem}*.jpeg"))
    return candidates[0] if candidates else None


def make_labeled_grid(
    rows: list[list[Image.Image | None]],
    row_labels: Iterable[str],
    col_labels: Iterable[str],
    cell_size: int = 220,
    label_height: int = 28,
    label_width: int = 120,
    background: tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    row_labels = list(row_labels)
    col_labels = list(col_labels)
    width = label_width + len(col_labels) * cell_size
    height = label_height + len(row_labels) * cell_size
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col_idx, label in enumerate(col_labels):
        x = label_width + col_idx * cell_size + 6
        draw.text((x, 8), label[:28], fill=(25, 25, 25), font=font)

    for row_idx, label in enumerate(row_labels):
        y = label_height + row_idx * cell_size + 8
        draw.text((6, y), label[:22], fill=(25, 25, 25), font=font)

    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            x = label_width + col_idx * cell_size
            y = label_height + row_idx * cell_size
            if image is None:
                tile = Image.new("RGB", (cell_size, cell_size), (230, 230, 230))
                tile_draw = ImageDraw.Draw(tile)
                tile_draw.text((cell_size // 2 - 20, cell_size // 2 - 6), "missing", fill=(90, 90, 90), font=font)
            else:
                tile = resize_image(image.convert("RGB"), cell_size, keep_aspect_ratio=True)
            canvas.paste(tile, (x, y))
    return canvas

