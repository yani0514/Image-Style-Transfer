from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import feature_mean_std


@dataclass(frozen=True)
class TransformerConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    alpha: float = 0.8


class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 4, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class ConvDecoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TransformerStylizer(nn.Module):
    """A compact arbitrary-style transformer for trainable experiments.

    The model encodes content and style images, matches content feature
    statistics to style statistics, then refines the token sequence with a
    transformer encoder before decoding to RGB.
    """

    def __init__(self, config: TransformerConfig = TransformerConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.decoder = ConvDecoder(config.hidden_dim)

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        alpha: float | None = None,
    ) -> torch.Tensor:
        alpha = self.config.alpha if alpha is None else alpha
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        content_mean, content_std = feature_mean_std(content_features)
        style_mean, style_std = feature_mean_std(style_features)
        normalized = (content_features - content_mean) / content_std
        stylized = normalized * style_std + style_mean
        mixed = alpha * stylized + (1.0 - alpha) * content_features

        batch, channels, height, width = mixed.shape
        tokens = mixed.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        refined = tokens.transpose(1, 2).view(batch, channels, height, width)
        return self.decoder(refined).clamp(0.0, 1.0)


def load_transformer_stylizer(
    checkpoint_path: str | Path,
    device: torch.device | str,
    config: TransformerConfig = TransformerConfig(),
) -> TransformerStylizer:
    model = TransformerStylizer(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    cleaned = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    return model.eval()


@torch.no_grad()
def run_transformer(
    content: torch.Tensor,
    style: torch.Tensor,
    model: TransformerStylizer,
    alpha: float = 0.8,
) -> tuple[torch.Tensor, dict[str, object]]:
    started_at = time.perf_counter()
    output = model(content, style, alpha=alpha)
    metadata = {
        "method": "transformer",
        "runtime_seconds": time.perf_counter() - started_at,
        "config": asdict(model.config) | {"alpha": alpha},
    }
    return output, metadata

