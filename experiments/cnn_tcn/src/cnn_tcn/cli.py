from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import inspect_data, validate_paths, write_json
from .inference import load_checkpoint, predict_split
from .train import train


def _add_data_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True, type=Path, help="Read-only root containing streams/ and streams_events.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Writable destination for this command's artifacts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cnn-tcn", description="ABC2027 CNN–TCN experiment tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Read and validate the released train/validation metadata")
    _add_data_paths(inspect_parser)
    inspect_parser.add_argument("--config", required=True, type=Path)
    inspect_parser.set_defaults(handler=_run_inspect)

    train_parser = subparsers.add_parser("train", help="Train V1 or V2 and select checkpoints with the official validation evaluator")
    _add_data_paths(train_parser)
    train_parser.add_argument("--variant", choices=("v1", "v2"), required=True)
    train_parser.add_argument("--config", required=True, type=Path)
    train_parser.add_argument("--evaluator-root", required=True, type=Path, help="Root of the official package containing src/abc2027/evaluate.py")
    train_parser.add_argument("--resume", type=Path)
    train_parser.set_defaults(handler=_run_train)

    infer_parser = subparsers.add_parser("infer", help="Run offline inference on the validation split")
    _add_data_paths(infer_parser)
    infer_parser.add_argument("--checkpoint", required=True, type=Path)
    infer_parser.add_argument("--split", choices=("val",), default="val")
    infer_parser.set_defaults(handler=_run_infer)
    return parser


def _run_inspect(args: argparse.Namespace) -> int:
    data, output = validate_paths(args.data_root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    report = inspect_data(data, config)
    report["config_variant"] = config["variant"]
    destination = output / "data_check.json"
    write_json(destination, report)
    print(f"data_check={destination}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if config["variant"] != args.variant:
        raise ValueError(f"--variant {args.variant} does not match config.variant {config['variant']}")
    train(args.data_root, args.output_dir, args.evaluator_root, config, args.resume)
    return 0


def _run_infer(args: argparse.Namespace) -> int:
    data, output = validate_paths(args.data_root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, config, normalizer, _payload, device = load_checkpoint(args.checkpoint)
    prediction = predict_split(
        model,
        data,
        output,
        config,
        normalizer,
        args.split,
        device,
        checkpoint_label=str(args.checkpoint),
    )
    print(f"predictions={prediction}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
