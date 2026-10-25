from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    variant = config.get("variant")
    if variant not in {"v1", "v2"}:
        raise ValueError("config.variant must be 'v1' or 'v2'")
    for name in ("seed", "sample_rate_hz", "window_frames", "hop_frames", "normalizer_chunk_frames"):
        if name not in config:
            raise ValueError(f"missing config key: {name}")
    if float(config["sample_rate_hz"]) <= 0 or int(config["window_frames"]) <= 0 or int(config["hop_frames"]) <= 0:
        raise ValueError("sample rate, window length and hop must be positive")
    if int(config["hop_frames"]) > int(config["window_frames"]):
        raise ValueError("hop_frames must not exceed window_frames, otherwise inference leaves uncovered frames")
    if int(config["normalizer_chunk_frames"]) <= 0:
        raise ValueError("normalizer_chunk_frames must be positive")

    radar_order = config.get("radar_order")
    if not isinstance(radar_order, list) or len(radar_order) != 3 or len(set(map(str, radar_order))) != 3:
        raise ValueError("radar_order must contain three distinct sensor serials")

    model = config.get("model", {})
    channels = model.get("range_channels", [])
    kernels = model.get("range_kernels", [])
    strides = model.get("range_strides", [])
    if not channels or len(channels) != len(kernels) or len(channels) != len(strides):
        raise ValueError("range_channels, range_kernels and range_strides must have equal nonzero length")
    if any(int(c) <= 0 for c in channels):
        raise ValueError("range_channels must be positive")
    if int(channels[-1]) != int(model.get("features_per_radar", 0)):
        raise ValueError("last range_channels entry must equal features_per_radar")
    if any(int(k) <= 0 or int(k) % 2 == 0 for k in kernels):
        raise ValueError("range kernels must be positive odd integers")
    if any(int(s) <= 0 for s in strides):
        raise ValueError("range strides must be positive")
    if int(model.get("tcn_hidden", 0)) <= 0:
        raise ValueError("model.tcn_hidden must be positive")
    if int(model.get("tcn_kernel", 0)) <= 0 or int(model["tcn_kernel"]) % 2 == 0:
        raise ValueError("model.tcn_kernel must be a positive odd integer")
    if not model.get("tcn_dilations") or any(int(d) <= 0 for d in model["tcn_dilations"]):
        raise ValueError("model.tcn_dilations must contain positive integers")
    if not 0.0 <= float(model.get("dropout", -1)) < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")

    training = config.get("training", {})
    for name in ("batch_size", "epochs", "num_workers"):
        if int(training.get(name, -1)) < 0:
            raise ValueError(f"training.{name} must be nonnegative")
    if int(training.get("batch_size", 0)) <= 0 or int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.batch_size and training.epochs must be positive")
    if float(training.get("learning_rate", 0)) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(training.get("weight_decay", -1)) < 0 or float(training.get("grad_clip_norm", 0)) <= 0:
        raise ValueError("training.weight_decay must be nonnegative and grad_clip_norm positive")
    if training.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu or cuda")

    loss = config.get("loss", {})
    for name in ("fall_weight", "start_weight", "end_weight", "boundary_sigma_s", "ignore_margin_s",
                 "fall_positive_weight", "boundary_positive_weight"):
        if name not in loss or float(loss[name]) < 0:
            raise ValueError(f"loss.{name} must be present and nonnegative")
    if float(loss["boundary_sigma_s"]) <= 0:
        raise ValueError("loss.boundary_sigma_s must be positive")
    if float(loss["fall_positive_weight"]) <= 0 or float(loss["boundary_positive_weight"]) <= 0:
        raise ValueError("positive class weights must be positive")
    if variant == "v1" and (float(loss["start_weight"]) != 0 or float(loss["end_weight"]) != 0):
        raise ValueError("V1 must have zero boundary loss weights")

    decoder = config.get("decoder", {})
    for name in ("fall_threshold", "boundary_threshold"):
        if not 0.0 < float(decoder.get(name, 0)) < 1.0:
            raise ValueError(f"decoder.{name} must be between 0 and 1")
    for name in ("merge_gap_s", "min_duration_s", "boundary_search_radius_s"):
        if float(decoder.get(name, -1)) < 0:
            raise ValueError(f"decoder.{name} must be nonnegative")
