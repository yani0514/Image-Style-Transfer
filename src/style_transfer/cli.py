from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .config import AdaINConfig, NSTConfig, ensure_parent, parse_alphas
from .io import (
    find_pair_output,
    list_images,
    load_image,
    make_labeled_grid,
    save_image,
)
from .metrics import evaluate_output_directory
from .methods.adain import (
    load_adain_decoder,
    run_adain_decoder,
    run_adain_inversion,
)
from .methods.nst import run_nst
from .methods.transformer import load_transformer_stylizer, run_transformer
from .methods.transformer import TransformerConfig
from .training import TrainConfig, train_adain_decoder, train_transformer_stylizer
from .vgg import build_vgg_extractor


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def write_metadata(output_path: Path, metadata: dict[str, object]) -> None:
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def add_common_image_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--content", required=True, help="Path to content image.")
    parser.add_argument("--style", required=True, help="Path to style image.")
    parser.add_argument("--output", required=True, help="Path for the stylized image.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")


def add_vgg_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrained",
        choices=["auto", "yes", "no"],
        default="auto",
        help="How to load VGG19 features. Use yes for strict evaluation.",
    )


def command_nst(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    content = load_image(args.content, image_size=args.image_size, device=device)
    style = load_image(args.style, image_size=args.image_size, device=device)
    config = NSTConfig(
        steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
    )
    extractor = build_vgg_extractor(
        layers=(config.content_layer, *config.style_layers),
        device=device,
        pretrained=args.pretrained,
    )
    output, metadata = run_nst(content, style, extractor, config=config, init=args.init)
    output_path = ensure_parent(args.output)
    save_image(output, output_path)
    write_metadata(output_path, metadata)
    print(f"Saved {output_path}")


def command_adain(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    content = load_image(args.content, image_size=args.image_size, device=device)
    style = load_image(args.style, image_size=args.image_size, device=device)
    config = AdaINConfig(
        alpha=args.alpha,
        inversion_steps=args.inversion_steps,
        inversion_learning_rate=args.learning_rate,
    )
    extractor = build_vgg_extractor(layers=(config.layer,), device=device, pretrained=args.pretrained)

    if args.mode in {"auto", "decoder"} and args.decoder_checkpoint:
        decoder = load_adain_decoder(args.decoder_checkpoint, device=device)
        output, metadata = run_adain_decoder(content, style, extractor, decoder, config=config)
    elif args.mode == "decoder":
        raise ValueError("--mode decoder requires --decoder-checkpoint")
    else:
        output, metadata = run_adain_inversion(content, style, extractor, config=config)

    output_path = ensure_parent(args.output)
    save_image(output, output_path)
    write_metadata(output_path, metadata)
    print(f"Saved {output_path}")


def command_transformer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if not args.checkpoint:
        raise ValueError("Transformer transfer requires --checkpoint with trained weights.")
    content = load_image(args.content, image_size=args.image_size, device=device)
    style = load_image(args.style, image_size=args.image_size, device=device)
    model = load_transformer_stylizer(args.checkpoint, device=device)
    output, metadata = run_transformer(content, style, model=model, alpha=args.alpha)
    output_path = ensure_parent(args.output)
    save_image(output, output_path)
    write_metadata(output_path, metadata)
    print(f"Saved {output_path}")


def command_evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    extractor = build_vgg_extractor(device=device, pretrained=args.pretrained)
    frame = evaluate_output_directory(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        output_dir=args.output_dir,
        extractor=extractor,
        image_size=args.image_size,
        device=device,
        content_limit=args.content_limit,
        style_limit=args.style_limit,
    )
    output_path = ensure_parent(args.metrics_csv)
    frame.to_csv(output_path, index=False)
    ok_rows = frame[frame["status"] == "ok"] if "status" in frame.columns else frame
    summary = ok_rows.describe()
    summary_path = output_path.with_suffix(".summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved {output_path}")
    print(f"Saved {summary_path}")


def command_gallery(args: argparse.Namespace) -> None:
    content_paths = list_images(args.content_dir, limit=args.content_limit)
    style_paths = list_images(args.style_dir, limit=args.style_limit)
    rows = []
    for content_path in content_paths:
        row = []
        for style_path in style_paths:
            output_path = find_pair_output(args.output_dir, content_path.stem, style_path.stem)
            row.append(Image.open(output_path).convert("RGB") if output_path else None)
        rows.append(row)
    gallery = make_labeled_grid(
        rows,
        row_labels=[path.stem for path in content_paths],
        col_labels=[path.stem for path in style_paths],
        cell_size=args.cell_size,
    )
    output_path = ensure_parent(args.gallery_path)
    gallery.save(output_path)
    print(f"Saved {output_path}")


def command_ablation(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    content = load_image(args.content, image_size=args.image_size, device=device)
    style = load_image(args.style, image_size=args.image_size, device=device)
    alphas = parse_alphas(args.alphas)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = build_vgg_extractor(layers=("relu4_1",), device=device, pretrained=args.pretrained)
    decoder = None
    if args.decoder_checkpoint:
        decoder = load_adain_decoder(args.decoder_checkpoint, device=device)

    records = []
    content_stem = Path(args.content).stem
    style_stem = Path(args.style).stem
    for alpha in alphas:
        config = AdaINConfig(
            alpha=alpha,
            inversion_steps=args.inversion_steps,
            inversion_learning_rate=args.learning_rate,
        )
        if decoder is not None and args.mode in {"auto", "decoder"}:
            output, metadata = run_adain_decoder(content, style, extractor, decoder, config=config)
        elif args.mode == "decoder":
            raise ValueError("--mode decoder requires --decoder-checkpoint")
        else:
            output, metadata = run_adain_inversion(content, style, extractor, config=config)
        filename = f"{content_stem}__{style_stem}__alpha{alpha:.1f}.png"
        output_path = output_dir / filename
        save_image(output, output_path)
        records.append({"alpha": alpha, "output": str(output_path), **metadata})

    metadata_path = output_dir / "ablation_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"Saved {metadata_path}")


def command_batch(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    content_paths = list_images(args.content_dir, limit=args.content_limit)
    style_paths = list_images(args.style_dir, limit=args.style_limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    extractor = None
    decoder = None
    transformer_model = None

    if args.method == "nst":
        nst_config = NSTConfig(
            steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            tv_weight=args.tv_weight,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
        )
        extractor = build_vgg_extractor(
            layers=(nst_config.content_layer, *nst_config.style_layers),
            device=device,
            pretrained=args.pretrained,
        )
    elif args.method == "adain":
        adain_config = AdaINConfig(
            alpha=args.alpha,
            inversion_steps=args.inversion_steps,
            inversion_learning_rate=args.learning_rate,
        )
        extractor = build_vgg_extractor(layers=(adain_config.layer,), device=device, pretrained=args.pretrained)
        if args.decoder_checkpoint:
            decoder = load_adain_decoder(args.decoder_checkpoint, device=device)
        elif args.mode == "decoder":
            raise ValueError("--mode decoder requires --decoder-checkpoint")
    elif args.method == "transformer":
        if not args.checkpoint:
            raise ValueError("--method transformer requires --checkpoint")
        transformer_model = load_transformer_stylizer(args.checkpoint, device=device)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    for content_path in content_paths:
        content = load_image(content_path, image_size=args.image_size, device=device)
        for style_path in style_paths:
            style = load_image(style_path, image_size=args.image_size, device=device)
            if args.method == "nst":
                assert extractor is not None
                output, metadata = run_nst(content, style, extractor, config=nst_config, init=args.init)
                suffix = "nst"
            elif args.method == "adain":
                assert extractor is not None
                if decoder is not None and args.mode in {"auto", "decoder"}:
                    output, metadata = run_adain_decoder(content, style, extractor, decoder, config=adain_config)
                    suffix = f"adain_alpha{args.alpha:.1f}"
                else:
                    output, metadata = run_adain_inversion(content, style, extractor, config=adain_config)
                    suffix = f"adain_inversion_alpha{args.alpha:.1f}"
            else:
                assert transformer_model is not None
                output, metadata = run_transformer(content, style, model=transformer_model, alpha=args.alpha)
                suffix = f"transformer_alpha{args.alpha:.1f}"

            filename = f"{content_path.stem}__{style_path.stem}__{suffix}.png"
            output_path = output_dir / filename
            save_image(output, output_path)
            record = {
                "content": str(content_path),
                "style": str(style_path),
                "output": str(output_path),
                **metadata,
            }
            records.append(record)
            write_metadata(output_path, metadata)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"Saved {manifest_path}")


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
    )


def command_train_adain_decoder(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    train_config = build_train_config(args)
    extractor = build_vgg_extractor(device=device, pretrained=args.pretrained)
    metadata = train_adain_decoder(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        extractor=extractor,
        output_checkpoint=args.output_checkpoint,
        config=train_config,
        device=device,
        resume_checkpoint=args.resume_checkpoint,
        content_limit=args.content_limit,
        style_limit=args.style_limit,
    )
    metadata_path = Path(args.output_checkpoint).with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {args.output_checkpoint}")
    print(f"Saved {metadata_path}")


def command_train_transformer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    train_config = build_train_config(args)
    transformer_config = TransformerConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        alpha=args.alpha,
    )
    extractor = build_vgg_extractor(device=device, pretrained=args.pretrained)
    metadata = train_transformer_stylizer(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        extractor=extractor,
        output_checkpoint=args.output_checkpoint,
        train_config=train_config,
        transformer_config=transformer_config,
        device=device,
        content_limit=args.content_limit,
        style_limit=args.style_limit,
    )
    metadata_path = Path(args.output_checkpoint).with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved {args.output_checkpoint}")
    print(f"Saved {metadata_path}")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--content-dir", required=True)
    parser.add_argument("--style-dir", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=10.0)
    parser.add_argument("--tv-weight", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--content-limit", type=int)
    parser.add_argument("--style-limit", type=int)
    parser.add_argument("--device", default="auto")
    add_vgg_arg(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image style transfer experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    nst = subparsers.add_parser("nst", help="Run classic optimization-based NST.")
    add_common_image_args(nst)
    add_vgg_arg(nst)
    nst.add_argument("--steps", type=int, default=300)
    nst.add_argument("--content-weight", type=float, default=1.0)
    nst.add_argument("--style-weight", type=float, default=100_000.0)
    nst.add_argument("--tv-weight", type=float, default=1e-6)
    nst.add_argument("--learning-rate", type=float, default=0.03)
    nst.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    nst.add_argument("--init", choices=["content", "noise"], default="content")
    nst.set_defaults(func=command_nst)

    adain = subparsers.add_parser("adain", help="Run AdaIN transfer.")
    add_common_image_args(adain)
    add_vgg_arg(adain)
    adain.add_argument("--alpha", type=float, default=0.8)
    adain.add_argument("--decoder-checkpoint")
    adain.add_argument("--mode", choices=["auto", "decoder", "inversion"], default="auto")
    adain.add_argument("--inversion-steps", type=int, default=200)
    adain.add_argument("--learning-rate", type=float, default=0.05)
    adain.set_defaults(func=command_adain)

    transformer = subparsers.add_parser("transformer", help="Run transformer stylizer.")
    add_common_image_args(transformer)
    transformer.add_argument("--checkpoint")
    transformer.add_argument("--alpha", type=float, default=0.8)
    transformer.set_defaults(func=command_transformer)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate generated outputs.")
    evaluate.add_argument("--content-dir", required=True)
    evaluate.add_argument("--style-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--metrics-csv", required=True)
    evaluate.add_argument("--image-size", type=int, default=512)
    evaluate.add_argument("--content-limit", type=int)
    evaluate.add_argument("--style-limit", type=int)
    evaluate.add_argument("--device", default="auto")
    add_vgg_arg(evaluate)
    evaluate.set_defaults(func=command_evaluate)

    gallery = subparsers.add_parser("gallery", help="Create a qualitative gallery.")
    gallery.add_argument("--content-dir", required=True)
    gallery.add_argument("--style-dir", required=True)
    gallery.add_argument("--output-dir", required=True)
    gallery.add_argument("--gallery-path", required=True)
    gallery.add_argument("--content-limit", type=int)
    gallery.add_argument("--style-limit", type=int)
    gallery.add_argument("--cell-size", type=int, default=220)
    gallery.set_defaults(func=command_gallery)

    ablation = subparsers.add_parser("ablation", help="Run an AdaIN alpha sweep.")
    ablation.add_argument("--content", required=True)
    ablation.add_argument("--style", required=True)
    ablation.add_argument("--output-dir", required=True)
    ablation.add_argument("--alphas", nargs="+", default=["0.2", "0.5", "0.8", "1.0"])
    ablation.add_argument("--image-size", type=int, default=512)
    ablation.add_argument("--device", default="auto")
    add_vgg_arg(ablation)
    ablation.add_argument("--decoder-checkpoint")
    ablation.add_argument("--mode", choices=["auto", "decoder", "inversion"], default="auto")
    ablation.add_argument("--inversion-steps", type=int, default=200)
    ablation.add_argument("--learning-rate", type=float, default=0.05)
    ablation.set_defaults(func=command_ablation)

    batch = subparsers.add_parser("batch", help="Run a method over a content-style grid.")
    batch.add_argument("--method", choices=["nst", "adain", "transformer"], required=True)
    batch.add_argument("--content-dir", required=True)
    batch.add_argument("--style-dir", required=True)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--image-size", type=int, default=512)
    batch.add_argument("--content-limit", type=int)
    batch.add_argument("--style-limit", type=int)
    batch.add_argument("--device", default="auto")
    add_vgg_arg(batch)
    batch.add_argument("--alpha", type=float, default=0.8)
    batch.add_argument("--decoder-checkpoint")
    batch.add_argument("--mode", choices=["auto", "decoder", "inversion"], default="auto")
    batch.add_argument("--checkpoint")
    batch.add_argument("--steps", type=int, default=300)
    batch.add_argument("--content-weight", type=float, default=1.0)
    batch.add_argument("--style-weight", type=float, default=100_000.0)
    batch.add_argument("--tv-weight", type=float, default=1e-6)
    batch.add_argument("--learning-rate", type=float, default=0.03)
    batch.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    batch.add_argument("--init", choices=["content", "noise"], default="content")
    batch.add_argument("--inversion-steps", type=int, default=200)
    batch.set_defaults(func=command_batch)

    train_adain = subparsers.add_parser("train-adain-decoder", help="Train an AdaIN decoder.")
    add_training_args(train_adain)
    train_adain.add_argument("--resume-checkpoint")
    train_adain.set_defaults(func=command_train_adain_decoder)

    train_transformer = subparsers.add_parser("train-transformer", help="Train the transformer stylizer.")
    add_training_args(train_transformer)
    train_transformer.add_argument("--hidden-dim", type=int, default=256)
    train_transformer.add_argument("--num-heads", type=int, default=8)
    train_transformer.add_argument("--num-layers", type=int, default=4)
    train_transformer.add_argument("--alpha", type=float, default=0.8)
    train_transformer.set_defaults(func=command_train_transformer)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
