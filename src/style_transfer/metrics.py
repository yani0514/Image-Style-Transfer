from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .config import DEFAULT_CONTENT_LAYER, DEFAULT_STYLE_LAYERS
from .io import find_pair_output, list_images, load_image
from .losses import feature_mean_std, gram_matrix
from .vgg import VGGFeatureExtractor


@torch.no_grad()
def evaluate_tensors(
    output: torch.Tensor,
    content: torch.Tensor,
    style: torch.Tensor,
    extractor: VGGFeatureExtractor,
    content_layer: str = DEFAULT_CONTENT_LAYER,
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS,
) -> dict[str, float]:
    all_layers = tuple(dict.fromkeys((content_layer, *style_layers)))
    output_features = extractor(output, all_layers)
    content_features = extractor(content, (content_layer,))
    style_features = extractor(style, style_layers)

    content_vgg_mse = F.mse_loss(
        output_features[content_layer],
        content_features[content_layer],
    )

    gram_losses = []
    stats_losses = []
    for layer in style_layers:
        output_layer = output_features[layer]
        style_layer = style_features[layer]
        gram_losses.append(F.mse_loss(gram_matrix(output_layer), gram_matrix(style_layer)))

        output_mean, output_std = feature_mean_std(output_layer)
        style_mean, style_std = feature_mean_std(style_layer)
        stats_losses.append(F.mse_loss(output_mean, style_mean))
        stats_losses.append(F.mse_loss(output_std, style_std))

    return {
        "content_vgg_mse": float(content_vgg_mse.detach().cpu()),
        "style_gram_mse": float(torch.stack(gram_losses).mean().detach().cpu()),
        "style_stats_mse": float(torch.stack(stats_losses).mean().detach().cpu()),
    }


def evaluate_output_directory(
    content_dir: str | Path,
    style_dir: str | Path,
    output_dir: str | Path,
    extractor: VGGFeatureExtractor,
    image_size: int,
    device: torch.device | str,
    content_layer: str = DEFAULT_CONTENT_LAYER,
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS,
    content_limit: int | None = None,
    style_limit: int | None = None,
) -> pd.DataFrame:
    content_paths = list_images(content_dir, limit=content_limit)
    style_paths = list_images(style_dir, limit=style_limit)
    if not content_paths:
        raise ValueError(f"No content images found in {content_dir}")
    if not style_paths:
        raise ValueError(f"No style images found in {style_dir}")
    rows: list[dict[str, object]] = []

    for content_path in content_paths:
        content = load_image(content_path, image_size=image_size, device=device)
        for style_path in style_paths:
            output_path = find_pair_output(output_dir, content_path.stem, style_path.stem)
            if output_path is None:
                rows.append({
                    "content": content_path.name,
                    "style": style_path.name,
                    "output": "",
                    "status": "missing",
                })
                continue

            style = load_image(style_path, image_size=image_size, device=device)
            output = load_image(output_path, image_size=image_size, device=device)
            metrics = evaluate_tensors(
                output=output,
                content=content,
                style=style,
                extractor=extractor,
                content_layer=content_layer,
                style_layers=style_layers,
            )
            rows.append({
                "content": content_path.name,
                "style": style_path.name,
                "output": str(output_path),
                "status": "ok",
                **metrics,
            })

    return pd.DataFrame(rows)
