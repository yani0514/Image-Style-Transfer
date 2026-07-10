# Image Style Transfer

This project implements a small, reproducible image style transfer system for
comparing quality and efficiency across optimization-based neural style
transfer, AdaIN arbitrary style transfer, and a transformer stylizer.

## What Is Included

- NST baseline using VGG content loss and Gram-matrix style loss.
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
  run_nst.py                   NST baseline
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
    transformer.py             Transformer stylizer
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first run downloads torchvision's pretrained VGG19 weights if they are not
cached.

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

## NST Baseline

```bash
python scripts/run_nst.py \
  --content data/content/horses.png \
  --style data/styles/Starry_Night_Van_Gogh.jpg \
  --output outputs/nst/horses_Starry_Night_Van_Gogh_nst.png \
  --image-size 512 \
  --steps 300
```

NST directly optimizes the output image. It is slower but usually strong for a
single content-style pair.

To run NST over a fixed evaluation grid:

```bash
python scripts/run_batch.py \
  --method nst \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/nst \
  --content-limit 20 \
  --style-limit 5 \
  --steps 300
```

## AdaIN

Fast feed-forward AdaIN requires a trained decoder checkpoint:

Without a decoder checkpoint, the command uses AdaIN feature inversion by
default. This still uses the AdaIN target feature statistics and strength knob,
but reconstructs the image by iterative optimization:

```bash
python scripts/run_adain.py \
  --content data/content/horses.png \
  --style data/styles/Starry_Night_Van_Gogh.jpg \
  --output outputs/adain_inversion/horses_Starry_Night_Van_Gogh_adain_inv.png \
  --alpha 0.8 \
  --mode inversion \
  --inversion-steps 200
```

Train the decoder with COCO 2017 content downloaded through FiftyOne's COCO
utility (the default) and the artwork folder. This path does not require a
MongoDB service or a registered FiftyOne dataset. `--content-limit` is useful for an initial
pipeline check; omit it for the complete 118,287-image training split. The full
split requires substantial disk space and download time.

Downloads created by older versions under `~/fiftyone/coco-2017` are detected
and reused automatically, including downloads that completed before a MongoDB
startup error.

First verify the pipeline with a small subset:

```bash
python scripts/train_adain_decoder.py \
  --style-dir data/styles \
  --output-checkpoint checkpoints/adain_decoder_test.pth \
  --coco-dataset-dir data/coco \
  --content-limit 500 \
  --steps 100
```

```bash
python scripts/run_adain.py \
  --content data/content/horses.png \
  --style data/styles/Starry_Night_Van_Gogh.jpg \
  --output outputs/adain/horses_Starry_Night_Van_Gogh_adain.png \
  --decoder-checkpoint checkpoints/adain_decoder_test.pth \
  --alpha 0.8
```

Then train on the full split:

```bash
python scripts/train_adain_decoder.py \
  --style-dir data/styles \
  --output-checkpoint checkpoints/adain_decoder.pth \
  --coco-dataset-dir data/coco \
  --image-size 256 \
  --batch-size 4 \
  --steps 10000
```

```bash
python scripts/run_adain.py \
  --content data/content/horses.png \
  --style data/styles/Starry_Night_Van_Gogh.jpg \
  --output outputs/adain/horses_Starry_Night_Van_Gogh_adain.png \
  --decoder-checkpoint checkpoints/adain_decoder.pth \
  --alpha 0.8
```



Then generate a full evaluation grid:

```bash
python scripts/run_batch.py \
  --method adain \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/adain \
  --decoder-checkpoint checkpoints/adain_decoder.pth \
  --alpha 0.8
```

## Transformer Stylizer

The transformer model must be trained or loaded from a checkpoint before qualitative use:

```bash
python scripts/run_transformer.py \
  --content data/content/horses.png \
  --style data/styles/Starry_Night_Van_Gogh.jpg \
  --output outputs/transformer/horses_Starry_Night_Van_Gogh_transf.png \
  --checkpoint checkpoints/transformer_stylizer.pth
```

Train it with the same COCO content pipeline and perceptual objective:

```bash
python scripts/train_transformer.py \
  --style-dir data/styles \
  --output-checkpoint checkpoints/transformer_stylizer.pth \
  --coco-dataset-dir data/coco \
  --image-size 256 \
  --batch-size 4 \
  --steps 10000
```

To deliberately train either feed-forward model on a local content collection,
pass `--content-source local --content-dir path/to/images`.

## Evaluation

Batch generation creates output names containing both the content and style
stems. Evaluate each method in its own output directory.

### AdaIN evaluation

```bash
python scripts/run_batch.py \
  --method adain \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/gallery_adain \
  --decoder-checkpoint checkpoints/adain_decoder.pth \
  --content-limit 20 \
  --style-limit 5 \
  --random-sample \
  --seed 42 \
  --alpha 0.8
```

Then run:

```bash
python scripts/evaluate.py \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/adain \
  --metrics-csv outputs/adain_metrics.csv
```

### NST evaluation

```bash
python scripts/run_batch.py \
  --method nst \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/nst \
  --content-limit 20 \
  --style-limit 5 \
  --steps 300

python scripts/evaluate.py \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/nst \
  --content-limit 20 \
  --style-limit 5 \
  --metrics-csv outputs/nst_metrics.csv
```

### Transformer evaluation

```bash
python scripts/run_batch.py \
  --method transformer \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/transformer \
  --checkpoint checkpoints/transformer_stylizer.pth \
  --content-limit 20 \
  --style-limit 5 \
  --alpha 0.8

python scripts/evaluate.py \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/transformer \
  --content-limit 20 \
  --style-limit 5 \
  --metrics-csv outputs/transformer_metrics.csv
```

Metrics:

- `content_vgg_mse`: VGG feature distance between output and content.
- `style_gram_mse`: Gram-matrix distance between output and style.
- `style_stats_mse`: feature mean/std distance between output and style.
- `runtime_seconds`: synchronized per-pair inference time read from each output's
  JSON metadata, alongside `method` and `alpha`.

Evaluation emits one row for every matching output (including every alpha or
method variant) rather than silently selecting the first filename match.

Lower values indicate closer preservation or matching for the corresponding
proxy. These are proxies, not human preference scores.

## Gallery

Each gallery uses the same seeded random sample during batch generation and
layout, producing all 25 combinations of five content and five style images.

### AdaIN gallery

```bash
python scripts/run_batch.py \
  --method adain \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/gallery_adain \
  --decoder-checkpoint checkpoints/adain_decoder.pth \
  --content-limit 5 \
  --style-limit 5 \
  --random-sample \
  --seed 42 \
  --alpha 0.8

python scripts/make_gallery.py \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/gallery_adain \
  --seed 42 \
  --gallery-path outputs/adain_gallery.jpg
```

### NST gallery

```bash
python scripts/run_batch.py \
  --method nst \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/gallery_nst \
  --content-limit 5 \
  --style-limit 5 \
  --random-sample \
  --seed 42 \
  --steps 300

python scripts/make_gallery.py \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/gallery_nst \
  --seed 42 \
  --gallery-path outputs/nst_gallery.jpg
```

### Transformer gallery

```bash
python scripts/run_batch.py \
  --method transformer \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/gallery_transformer \
  --checkpoint checkpoints/transformer_stylizer.pth \
  --content-limit 5 \
  --style-limit 5 \
  --random-sample \
  --seed 42 \
  --alpha 0.8

python scripts/make_gallery.py \
  --content-dir data/content \
  --style-dir data/styles \
  --output-dir outputs/gallery_transformer \
  --seed 42 \
  --gallery-path outputs/transformer_gallery.jpg
```

Change `--seed` in both commands of a pair to select another reproducible 5×5
sample. Gallery creation now fails clearly if any of the selected combinations
has not been generated.

## Notes On Analysis

Results compare:

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
