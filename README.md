# Image Style Transfer

This project implements a small, reproducible image style transfer system for
comparing quality and efficiency across classic optimization-based neural style
transfer, AdaIN arbitrary style transfer, and an optional transformer stylizer.

## What Is Included

- Classic NST baseline using VGG content loss and Gram-matrix style loss.
- AdaIN arbitrary style transfer with a tunable strength value `alpha`.
- AdaIN feature-inversion fallback for experiments without a trained decoder.
- Optional lightweight transformer stylizer architecture for trained experiments.
- VGG-based content preservation and style matching metrics.
- Strength ablation over `alpha in {0.2, 0.5, 0.8, 1.0}`.
- Gallery generation for content-by-style qualitative grids.

## Project Layout

```text
configs/
  eval_small.json              Example fixed evaluation protocol
scripts/
  run_nst.py                   Classic optimization baseline
  run_adain.py                 AdaIN transfer
  run_transformer.py           Transformer transfer
  run_batch.py                 Run one method over a content-style grid
  train_adain_decoder.py       Train the AdaIN decoder
  train_transformer.py         Train the transformer stylizer
  evaluate.py                  Metric evaluation
  make_gallery.py              Qualitative grid builder
  run_ablation.py              AdaIN alpha sweep
src/style_transfer/
  cli.py                       Command-line interface
  io.py                        Image loading, saving, grid utilities
  losses.py                    Gram, perceptual, TV losses
  metrics.py                   Evaluation metrics
  vgg.py                       VGG19 feature extraction
  methods/
    nst.py                     Optimization-based NST
    adain.py                   AdaIN and feature inversion
    transformer.py             Optional transformer stylizer
tests/
  smoke_test.py                Lightweight tensor and image checks
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The first run with pretrained VGG features may download torchvision VGG19
weights. For fully offline smoke checks, pass `--pretrained no`; quality metrics
and meaningful style transfer should use pretrained VGG features.

## Data

Place images in two folders:

```text
data/
  content/
    photo_001.jpg
    photo_002.jpg
  styles/
    starry_night.jpg
    mosaic.jpg
```

For the suggested evaluation protocol, use 20-50 content images and 5-10 style
images. The default config resizes the shorter side to 512 px and center-crops to
a fixed square for comparable metrics.

## Classic NST Baseline

```bash
python scripts/run_nst.py ^
  --content data/content/photo_001.jpg ^
  --style data/styles/starry_night.jpg ^
  --output outputs/nst/photo_001__starry_night.png ^
  --image-size 512 ^
  --steps 300
```

NST directly optimizes the output image. It is slower but usually strong for a
single content-style pair.

To run NST over a fixed evaluation grid:

```bash
python scripts/run_batch.py ^
  --method nst ^
  --content-dir data/content ^
  --style-dir data/styles ^
  --output-dir outputs/nst ^
  --content-limit 20 ^
  --style-limit 5 ^
  --steps 300
```

## AdaIN

Fast feed-forward AdaIN requires a trained decoder checkpoint:

```bash
python scripts/run_adain.py ^
  --content data/content/photo_001.jpg ^
  --style data/styles/starry_night.jpg ^
  --output outputs/adain/photo_001__starry_night.png ^
  --decoder-checkpoint checkpoints/adain_decoder.pth ^
  --alpha 0.8
```

Without a decoder checkpoint, the command uses AdaIN feature inversion by
default. This still uses the AdaIN target feature statistics and strength knob,
but reconstructs the image by iterative optimization:

```bash
python scripts/run_adain.py ^
  --content data/content/photo_001.jpg ^
  --style data/styles/starry_night.jpg ^
  --output outputs/adain_inversion/photo_001__starry_night.png ^
  --alpha 0.8 ^
  --mode inversion ^
  --inversion-steps 200
```

Train the provided decoder on your own content/style folders:

```bash
python scripts/train_adain_decoder.py ^
  --content-dir data/content ^
  --style-dir data/styles ^
  --output-checkpoint checkpoints/adain_decoder.pth ^
  --image-size 256 ^
  --batch-size 4 ^
  --steps 20000
```

Then generate a full evaluation grid:

```bash
python scripts/run_batch.py ^
  --method adain ^
  --content-dir data/content ^
  --style-dir data/styles ^
  --output-dir outputs/adain ^
  --decoder-checkpoint checkpoints/adain_decoder.pth ^
  --alpha 0.8
```

## Transformer Stylizer

The transformer model is included for compute-available experiments. It must be
trained or loaded from a checkpoint before qualitative use:

```bash
python scripts/run_transformer.py ^
  --content data/content/photo_001.jpg ^
  --style data/styles/starry_night.jpg ^
  --output outputs/transformer/photo_001__starry_night.png ^
  --checkpoint checkpoints/transformer_stylizer.pth
```

Train it with the same perceptual objective:

```bash
python scripts/train_transformer.py ^
  --content-dir data/content ^
  --style-dir data/styles ^
  --output-checkpoint checkpoints/transformer_stylizer.pth ^
  --image-size 256 ^
  --batch-size 4 ^
  --steps 20000
```

## Strength Ablation

```bash
python scripts/run_ablation.py ^
  --content data/content/photo_001.jpg ^
  --style data/styles/starry_night.jpg ^
  --output-dir outputs/ablation/photo_001__starry_night ^
  --alphas 0.2 0.5 0.8 1.0
```

This creates one image per alpha value and a small JSON metadata file with
runtime information.

## Evaluation

Generate outputs with names containing both content and style stems, for example:

```text
outputs/adain/photo_001__starry_night__alpha0.8.png
```

Then run:

```bash
python scripts/evaluate.py ^
  --content-dir data/content ^
  --style-dir data/styles ^
  --output-dir outputs/adain ^
  --metrics-csv outputs/adain_metrics.csv
```

Metrics:

- `content_vgg_mse`: VGG feature distance between output and content.
- `style_gram_mse`: Gram-matrix distance between output and style.
- `style_stats_mse`: feature mean/std distance between output and style.

Lower values indicate closer preservation or matching for the corresponding
proxy. These are proxies, not human preference scores.

## Gallery

```bash
python scripts/make_gallery.py ^
  --content-dir data/content ^
  --style-dir data/styles ^
  --output-dir outputs/adain ^
  --gallery-path outputs/adain_gallery.jpg
```

The gallery aligns content images as rows and styles as columns. Missing outputs
are left blank, which makes incomplete experiments easy to spot.

## Notes On Analysis

When reporting results, compare:

- Runtime per image pair for NST, AdaIN decoder, AdaIN inversion, and transformer.
- Content preservation against style matching metrics.
- Qualitative failure modes such as content distortion, weak texture transfer,
  style leakage into object boundaries, and color shift.
- Alpha sweep behavior: low alpha should preserve content more strongly; high
  alpha should match style statistics more aggressively.

## References

- Gatys, Ecker, and Bethge, 2016. Image Style Transfer Using Convolutional
  Neural Networks.
- Johnson, Alahi, and Fei-Fei, 2016. Perceptual Losses for Real-Time Style
  Transfer and Super-Resolution.
- Dumoulin, Shlens, and Kudlur, 2017. A Learned Representation for Artistic
  Style.
- Huang and Belongie, 2017. Arbitrary Style Transfer in Real-time with Adaptive
  Instance Normalization.
