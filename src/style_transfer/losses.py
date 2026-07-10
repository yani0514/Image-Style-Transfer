from __future__ import annotations

import torch
import torch.nn.functional as F


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / float(channels * height * width)


def feature_mean_std(features: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels = features.shape[:2]
    flattened = features.reshape(batch, channels, -1)
    mean = flattened.mean(dim=2).reshape(batch, channels, 1, 1)
    std = flattened.var(dim=2, unbiased=False).add(eps).sqrt().reshape(batch, channels, 1, 1)
    return mean, std


def content_loss(output_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output_features, target_features)


def style_loss(output_features: dict[str, torch.Tensor], target_grams: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = [
        F.mse_loss(gram_matrix(output_features[layer]), target_grams[layer])
        for layer in target_grams
    ]
    return torch.stack(losses).mean()


def feature_statistics_loss(
    output_features: dict[str, torch.Tensor],
    style_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for layer, output in output_features.items():
        style = style_features[layer]
        output_mean, output_std = feature_mean_std(output)
        style_mean, style_std = feature_mean_std(style)
        losses.append(F.mse_loss(output_mean, style_mean))
        losses.append(F.mse_loss(output_std, style_std))
    return torch.stack(losses).mean()


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return vertical + horizontal
