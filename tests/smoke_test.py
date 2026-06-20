from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image


def main() -> None:
    import torch

    from style_transfer.io import load_image, save_image
    from style_transfer.losses import feature_mean_std, gram_matrix
    from style_transfer.methods.adain import adaptive_instance_normalization

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        content_path = root / "content.png"
        output_path = root / "output.png"
        Image.new("RGB", (32, 32), (120, 80, 40)).save(content_path)

        image = load_image(content_path, image_size=32)
        assert image.shape == (1, 3, 32, 32)
        save_image(image, output_path)
        assert output_path.exists()

    content_features = torch.randn(1, 8, 4, 4)
    style_features = torch.randn(1, 8, 4, 4) * 2.0 + 3.0
    stylized = adaptive_instance_normalization(content_features, style_features)
    assert stylized.shape == content_features.shape
    assert gram_matrix(stylized).shape == (1, 8, 8)
    mean, std = feature_mean_std(stylized)
    assert mean.shape == (1, 8, 1, 1)
    assert std.shape == (1, 8, 1, 1)


if __name__ == "__main__":
    main()

