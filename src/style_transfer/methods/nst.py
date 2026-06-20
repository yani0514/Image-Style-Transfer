from __future__ import annotations

import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from tqdm import trange

from ..config import NSTConfig
from ..losses import gram_matrix, total_variation_loss
from ..vgg import VGGFeatureExtractor


def run_nst(
    content: torch.Tensor,
    style: torch.Tensor,
    extractor: VGGFeatureExtractor,
    config: NSTConfig,
    init: str = "content",
) -> tuple[torch.Tensor, dict[str, object]]:
    if init == "noise":
        generated = torch.rand_like(content)
    elif init == "content":
        generated = content.clone()
    else:
        raise ValueError("init must be either 'content' or 'noise'")

    generated.requires_grad_(True)
    all_layers = tuple(dict.fromkeys((config.content_layer, *config.style_layers)))

    with torch.no_grad():
        content_target = extractor(content, (config.content_layer,))[config.content_layer].detach()
        style_features = extractor(style, config.style_layers)
        style_targets = {
            layer: gram_matrix(features).detach()
            for layer, features in style_features.items()
        }

    if config.optimizer.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS([generated], lr=config.learning_rate, max_iter=1)
    elif config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam([generated], lr=config.learning_rate)
    else:
        raise ValueError("optimizer must be 'adam' or 'lbfgs'")

    history: list[dict[str, float]] = []
    started_at = time.perf_counter()

    iterator = trange(config.steps, desc="NST", leave=False)
    for step in iterator:
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            clamped = generated.clamp(0.0, 1.0)
            features = extractor(clamped, all_layers)

            c_loss = F.mse_loss(features[config.content_layer], content_target)
            s_losses = [
                F.mse_loss(gram_matrix(features[layer]), style_targets[layer])
                for layer in config.style_layers
            ]
            s_loss = torch.stack(s_losses).mean()
            tv_loss = total_variation_loss(clamped)

            total_loss = (
                config.content_weight * c_loss
                + config.style_weight * s_loss
                + config.tv_weight * tv_loss
            )
            total_loss.backward()

            if step % config.log_every == 0 or step == config.steps - 1:
                row = {
                    "step": float(step),
                    "loss": float(total_loss.detach().cpu()),
                    "content_loss": float(c_loss.detach().cpu()),
                    "style_loss": float(s_loss.detach().cpu()),
                    "tv_loss": float(tv_loss.detach().cpu()),
                }
                history.append(row)
                iterator.set_postfix(loss=f"{row['loss']:.3f}")
            return total_loss

        optimizer.step(closure)
        with torch.no_grad():
            generated.clamp_(0.0, 1.0)

    metadata = {
        "method": "nst",
        "runtime_seconds": time.perf_counter() - started_at,
        "config": asdict(config),
        "history": history,
    }
    return generated.detach().clamp(0.0, 1.0), metadata

