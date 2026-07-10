from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
DEFAULT_CONTENT_LAYER = "relu4_2"
DEFAULT_ADAIN_LAYER = "relu4_1"


@dataclass(frozen=True)
class ImageConfig:
    image_size: int = 512
    keep_aspect_ratio: bool = True


@dataclass(frozen=True)
class NSTConfig:
    steps: int = 300
    content_weight: float = 1.0
    style_weight: float = 100_000.0
    tv_weight: float = 1e-6
    learning_rate: float = 0.03
    optimizer: str = "adam"
    content_layer: str = DEFAULT_CONTENT_LAYER
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS
    log_every: int = 25


@dataclass(frozen=True)
class AdaINConfig:
    alpha: float = 0.8
    layer: str = DEFAULT_ADAIN_LAYER
    inversion_steps: int = 200
    inversion_learning_rate: float = 0.05
    tv_weight: float = 1e-6
    log_every: int = 25


@dataclass(frozen=True)
class EvaluationConfig:
    content_layer: str = DEFAULT_CONTENT_LAYER
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS


def parse_alphas(values: Iterable[str | float]) -> list[float]:
    alphas = [float(value) for value in values]
    for alpha in alphas:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alphas


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

