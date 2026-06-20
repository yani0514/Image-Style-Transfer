from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from .config import DEFAULT_CONTENT_LAYER, DEFAULT_STYLE_LAYERS
from .data import ImageFolderDataset, cycling_batches
from .losses import feature_statistics_loss, total_variation_loss
from .methods.adain import (
    AdaINDecoder,
    adaptive_instance_normalization,
    load_adain_decoder,
)
from .methods.transformer import TransformerConfig, TransformerStylizer
from .vgg import VGGFeatureExtractor


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 4
    image_size: int = 256
    learning_rate: float = 1e-4
    content_weight: float = 1.0
    style_weight: float = 10.0
    tv_weight: float = 1e-6
    num_workers: int = 0
    save_every: int = 1_000
    log_every: int = 50


def make_loaders(
    content_dir: str | Path,
    style_dir: str | Path,
    config: TrainConfig,
    content_limit: int | None = None,
    style_limit: int | None = None,
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    content_dataset = ImageFolderDataset(content_dir, image_size=config.image_size, limit=content_limit)
    style_dataset = ImageFolderDataset(style_dir, image_size=config.image_size, limit=style_limit)
    content_loader = DataLoader(
        content_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
    )
    style_loader = DataLoader(
        style_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
    )
    return content_loader, style_loader


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, object],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        output_path,
    )
    return output_path


def train_adain_decoder(
    content_dir: str | Path,
    style_dir: str | Path,
    extractor: VGGFeatureExtractor,
    output_checkpoint: str | Path,
    config: TrainConfig,
    device: torch.device | str,
    resume_checkpoint: str | Path | None = None,
    content_limit: int | None = None,
    style_limit: int | None = None,
) -> dict[str, object]:
    content_loader, style_loader = make_loaders(
        content_dir,
        style_dir,
        config,
        content_limit=content_limit,
        style_limit=style_limit,
    )
    content_batches = cycling_batches(content_loader)
    style_batches = cycling_batches(style_loader)
    decoder = (
        load_adain_decoder(resume_checkpoint, device=device)
        if resume_checkpoint
        else AdaINDecoder().to(device)
    )
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate)
    all_layers = tuple(dict.fromkeys(("relu4_1", *DEFAULT_STYLE_LAYERS)))
    history: list[dict[str, float]] = []
    started_at = time.perf_counter()

    iterator = trange(1, config.steps + 1, desc="Train AdaIN decoder", leave=False)
    for step in iterator:
        content = next(content_batches).to(device)
        style = next(style_batches).to(device)

        with torch.no_grad():
            content_features = extractor(content, ("relu4_1",))["relu4_1"]
            style_features = extractor(style, ("relu4_1",))["relu4_1"]
            target_features = adaptive_instance_normalization(content_features, style_features)
            style_targets = extractor(style, DEFAULT_STYLE_LAYERS)

        output = decoder(target_features).clamp(0.0, 1.0)
        output_features = extractor(output, all_layers)
        c_loss = F.mse_loss(output_features["relu4_1"], target_features)
        s_loss = feature_statistics_loss(
            {layer: output_features[layer] for layer in DEFAULT_STYLE_LAYERS},
            style_targets,
        )
        tv_loss = total_variation_loss(output)
        loss = (
            config.content_weight * c_loss
            + config.style_weight * s_loss
            + config.tv_weight * tv_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0 or step == 1:
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "content_loss": float(c_loss.detach().cpu()),
                "style_loss": float(s_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
            }
            history.append(row)
            iterator.set_postfix(loss=f"{row['loss']:.3f}")

        if step % config.save_every == 0:
            save_checkpoint(
                output_checkpoint,
                decoder,
                optimizer,
                step,
                metadata={"method": "adain_decoder", "config": asdict(config)},
            )

    save_checkpoint(
        output_checkpoint,
        decoder,
        optimizer,
        config.steps,
        metadata={"method": "adain_decoder", "config": asdict(config)},
    )
    return {
        "method": "train_adain_decoder",
        "runtime_seconds": time.perf_counter() - started_at,
        "checkpoint": str(output_checkpoint),
        "config": asdict(config),
        "history": history,
    }


def train_transformer_stylizer(
    content_dir: str | Path,
    style_dir: str | Path,
    extractor: VGGFeatureExtractor,
    output_checkpoint: str | Path,
    train_config: TrainConfig,
    transformer_config: TransformerConfig,
    device: torch.device | str,
    content_limit: int | None = None,
    style_limit: int | None = None,
) -> dict[str, object]:
    content_loader, style_loader = make_loaders(
        content_dir,
        style_dir,
        train_config,
        content_limit=content_limit,
        style_limit=style_limit,
    )
    content_batches = cycling_batches(content_loader)
    style_batches = cycling_batches(style_loader)
    model = TransformerStylizer(transformer_config).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    all_layers = tuple(dict.fromkeys((DEFAULT_CONTENT_LAYER, *DEFAULT_STYLE_LAYERS)))
    history: list[dict[str, float]] = []
    started_at = time.perf_counter()

    iterator = trange(1, train_config.steps + 1, desc="Train transformer", leave=False)
    for step in iterator:
        content = next(content_batches).to(device)
        style = next(style_batches).to(device)

        with torch.no_grad():
            content_target = extractor(content, (DEFAULT_CONTENT_LAYER,))[DEFAULT_CONTENT_LAYER]
            style_targets = extractor(style, DEFAULT_STYLE_LAYERS)

        output = model(content, style, alpha=transformer_config.alpha)
        output_features = extractor(output, all_layers)
        c_loss = F.mse_loss(output_features[DEFAULT_CONTENT_LAYER], content_target)
        s_loss = feature_statistics_loss(
            {layer: output_features[layer] for layer in DEFAULT_STYLE_LAYERS},
            style_targets,
        )
        tv_loss = total_variation_loss(output)
        loss = (
            train_config.content_weight * c_loss
            + train_config.style_weight * s_loss
            + train_config.tv_weight * tv_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % train_config.log_every == 0 or step == 1:
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "content_loss": float(c_loss.detach().cpu()),
                "style_loss": float(s_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
            }
            history.append(row)
            iterator.set_postfix(loss=f"{row['loss']:.3f}")

        if step % train_config.save_every == 0:
            save_checkpoint(
                output_checkpoint,
                model,
                optimizer,
                step,
                metadata={
                    "method": "transformer",
                    "train_config": asdict(train_config),
                    "transformer_config": asdict(transformer_config),
                },
            )

    save_checkpoint(
        output_checkpoint,
        model,
        optimizer,
        train_config.steps,
        metadata={
            "method": "transformer",
            "train_config": asdict(train_config),
            "transformer_config": asdict(transformer_config),
        },
    )
    return {
        "method": "train_transformer",
        "runtime_seconds": time.perf_counter() - started_at,
        "checkpoint": str(output_checkpoint),
        "train_config": asdict(train_config),
        "transformer_config": asdict(transformer_config),
        "history": history,
    }
