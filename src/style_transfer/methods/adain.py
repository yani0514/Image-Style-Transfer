from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from ..config import AdaINConfig
from ..losses import feature_mean_std, total_variation_loss
from ..vgg import VGGFeatureExtractor


def adaptive_instance_normalization(
    content_features: torch.Tensor,
    style_features: torch.Tensor,
) -> torch.Tensor:
    # Preserve normalized content structure while adopting style statistics.
    content_mean, content_std = feature_mean_std(content_features)
    style_mean, style_std = feature_mean_std(style_features)
    normalized = (content_features - content_mean) / content_std
    return normalized * style_std + style_mean


class AdaINDecoder(nn.Module):
    """Decoder architecture commonly used with AdaIN relu4_1 features."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def load_adain_decoder(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> AdaINDecoder:
    decoder = AdaINDecoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "decoder" in checkpoint:
        state_dict = checkpoint["decoder"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
        raise ValueError(f"Invalid AdaIN decoder checkpoint: {checkpoint_path}")
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("decoder."):
            key = key[len("decoder."):]
        cleaned[key] = value
    decoder.load_state_dict(cleaned, strict=True)
    return decoder.eval()


@torch.no_grad()
def run_adain_decoder(
    content: torch.Tensor,
    style: torch.Tensor,
    extractor: VGGFeatureExtractor,
    decoder: AdaINDecoder,
    config: AdaINConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    if content.is_cuda:
        torch.cuda.synchronize(content.device)
    started_at = time.perf_counter()
    content_features = extractor(content, (config.layer,))[config.layer]
    style_features = extractor(style, (config.layer,))[config.layer]
    target_features = adaptive_instance_normalization(content_features, style_features)
    # Alpha interpolates between content features and the fully stylized target.
    target_features = config.alpha * target_features + (1.0 - config.alpha) * content_features
    output = decoder(target_features).clamp(0.0, 1.0)
    if output.is_cuda:
        torch.cuda.synchronize(output.device)
    metadata = {
        "method": "adain_decoder",
        "runtime_seconds": time.perf_counter() - started_at,
        "config": asdict(config),
    }
    return output, metadata


def run_adain_inversion(
    content: torch.Tensor,
    style: torch.Tensor,
    extractor: VGGFeatureExtractor,
    config: AdaINConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    with torch.no_grad():
        content_features = extractor(content, (config.layer,))[config.layer].detach()
        style_features = extractor(style, (config.layer,))[config.layer].detach()
        adain_target = adaptive_instance_normalization(content_features, style_features)
        target_features = config.alpha * adain_target + (1.0 - config.alpha) * content_features

    generated = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=config.inversion_learning_rate)
    history: list[dict[str, float]] = []
    if content.is_cuda:
        torch.cuda.synchronize(content.device)
    started_at = time.perf_counter()

    iterator = trange(config.inversion_steps, desc="AdaIN inversion", leave=False)
    for step in iterator:
        optimizer.zero_grad()
        clamped = generated.clamp(0.0, 1.0)
        output_features = extractor(clamped, (config.layer,))[config.layer]
        reconstruction_loss = F.mse_loss(output_features, target_features)
        tv_loss = total_variation_loss(clamped)
        loss = reconstruction_loss + config.tv_weight * tv_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(0.0, 1.0)

        if step % config.log_every == 0 or step == config.inversion_steps - 1:
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "feature_loss": float(reconstruction_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
            }
            history.append(row)
            iterator.set_postfix(loss=f"{row['loss']:.3f}")

    if generated.is_cuda:
        torch.cuda.synchronize(generated.device)
    metadata = {
        "method": "adain_inversion",
        "runtime_seconds": time.perf_counter() - started_at,
        "config": asdict(config),
        "history": history,
    }
    return generated.detach().clamp(0.0, 1.0), metadata
