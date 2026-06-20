from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn


VGG_LAYER_NAMES = {
    1: "relu1_1",
    3: "relu1_2",
    6: "relu2_1",
    8: "relu2_2",
    11: "relu3_1",
    13: "relu3_2",
    15: "relu3_3",
    17: "relu3_4",
    20: "relu4_1",
    22: "relu4_2",
    24: "relu4_3",
    26: "relu4_4",
    29: "relu5_1",
    31: "relu5_2",
    33: "relu5_3",
    35: "relu5_4",
}


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        features: nn.Sequential,
        default_layers: Sequence[str] | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.features = features.eval()
        self.default_layers = tuple(default_layers or ("relu4_2",))
        self.normalize = normalize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        image: torch.Tensor,
        layers: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        required = set(layers or self.default_layers)
        if self.normalize:
            image = (image - self.mean) / self.std
        outputs: dict[str, torch.Tensor] = {}
        x = image
        for index, layer in enumerate(self.features):
            x = layer(x)
            name = VGG_LAYER_NAMES.get(index)
            if name in required:
                outputs[name] = x
            if required.issubset(outputs.keys()):
                break
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"Requested VGG layers were not reached: {sorted(missing)}")
        return outputs


def _load_torchvision_vgg19(pretrained: str) -> nn.Sequential:
    from torchvision import models

    if pretrained not in {"auto", "yes", "no"}:
        raise ValueError("pretrained must be one of: auto, yes, no")

    if pretrained == "no":
        return models.vgg19(weights=None).features

    try:
        weights = models.VGG19_Weights.IMAGENET1K_V1
        return models.vgg19(weights=weights).features
    except Exception as exc:
        if pretrained == "yes":
            raise RuntimeError(
                "Unable to load pretrained VGG19 weights. Install torchvision "
                "with model weights available or use --pretrained no for smoke tests."
            ) from exc
        warnings.warn(
            "Falling back to randomly initialized VGG19 features because pretrained "
            f"weights could not be loaded: {exc}",
            RuntimeWarning,
        )
        return models.vgg19(weights=None).features


def build_vgg_extractor(
    layers: Sequence[str] | None = None,
    device: torch.device | str | None = None,
    pretrained: str = "auto",
) -> VGGFeatureExtractor:
    extractor = VGGFeatureExtractor(
        _load_torchvision_vgg19(pretrained=pretrained),
        default_layers=layers,
    )
    if device is not None:
        extractor = extractor.to(device)
    return extractor.eval()

