from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import H5WindowDataset, fit_normalizer, validate_paths, write_json
from .inference import predict_split, resolve_device
from .losses import compute_loss
from .model import CNNTemporalModel
from .official_eval import ranking_key, score_validation


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    from contextlib import nullcontext

    return nullcontext()


def train(
    data_root: str | Path,
    output_dir: str | Path,
    evaluator_root: str | Path,
    config: dict[str, Any],
    resume: str | Path | None = None,
) -> None:
    data, output = validate_paths(data_root, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"output directory already contains a different config: {config_path}")
    else:
        write_json(config_path, config)

    device = resolve_device(str(config["training"]["device"]))
    seed = int(config["seed"])
    deterministic = bool(config["training"].get("deterministic", True))
    _set_seed(seed, deterministic)

    if resume is None:
        normalizer = fit_normalizer(data, config, int(config["normalizer_chunk_frames"]))
        start_epoch = 0
        best_key = (float("-inf"), float("-inf"), float("-inf"))
        global_step = 0
        model = CNNTemporalModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        payload: dict[str, Any] = {}
    else:
        payload = torch.load(Path(resume), map_location=device, weights_only=False)
        if payload.get("config") != config:
            raise ValueError("resume checkpoint config differs from the selected config")
        start_epoch = int(payload["epoch"]) + 1
        if start_epoch >= int(config["training"]["epochs"]):
            raise ValueError("checkpoint is already at or beyond the configured epoch count")
        normalizer = {
            "mean": np.asarray(payload["normalizer"]["mean"], dtype=np.float32),
            "std": np.asarray(payload["normalizer"]["std"], dtype=np.float32),
        }
        model = CNNTemporalModel(config).to(device)
        model.load_state_dict(payload["model_state"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        best_key = tuple(float(x) for x in payload["best_key"])
        global_step = int(payload["global_step"])
        if payload.get("rng_state"):
            _restore_rng(payload["rng_state"])

    np.savez(output / "normalizer.npz", mean=normalizer["mean"], std=normalizer["std"])
    dataset = H5WindowDataset(data, "train", config, normalizer, with_targets=True)
    generator = torch.Generator()
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["training"]["num_workers"]) > 0,
        generator=generator,
    )
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if resume is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])

    log_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        model.train()
        generator.manual_seed(seed + epoch)
        totals: dict[str, float] = {}
        batches = 0
        for batch in loader:
            inputs = batch["x"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, amp_enabled):
                outputs = model(inputs)
                loss, parts = compute_loss(outputs, batch, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            batches += 1
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + value
        if batches == 0:
            raise RuntimeError("training DataLoader yielded no batches")
        train_metrics = {f"train_{key}_loss": value / batches for key, value in totals.items()}

        validation_dir = output / "validation"
        prediction_path = predict_split(
            model,
            data,
            validation_dir,
            config,
            normalizer,
            "val",
            device,
            checkpoint_label=f"epoch-{epoch + 1}",
        )
        report = score_validation(evaluator_root, data, prediction_path)
        current_key = ranking_key(report)
        improved = current_key > best_key
        if improved:
            best_key = current_key

        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            **train_metrics,
            "validation_ranking": report["ranking_score"],
            "validation_primary": report["primary"],
            "selected_as_best": improved,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(validation_dir / "official_report.json", report)

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "best_key": list(best_key),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": config,
            "normalizer": normalizer,
            "rng_state": _rng_state(),
        }
        _atomic_torch_save(checkpoint, output / "last.pt")
        if improved:
            _atomic_torch_save(checkpoint, output / "best.pt")
        print(
            f"epoch={epoch + 1} val_f1_mean={current_key[0]:.6f} "
            f"start_median_abs={report['ranking_score']['start_error_median_abs']} "
            f"false_alarms_per_hour={report['ranking_score']['false_alarms_per_hour']} "
            f"best={improved}"
        )
