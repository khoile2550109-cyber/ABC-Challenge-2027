from __future__ import annotations

import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .data import _radar_indices, list_stream_files, window_starts, write_json
from .decode import decode_events
from .model import CNNTemporalModel


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requests CUDA, but CUDA is not available in the runtime")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device: {requested}")


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _normalized_window(
    cir_dataset: h5py.Dataset,
    radar_indices: list[int],
    start: int,
    frame_count: int,
    window_frames: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, int]:
    stop = min(start + window_frames, frame_count)
    valid_count = stop - start
    raw = np.asarray(cir_dataset[:, start:stop, :], dtype=np.float32)
    raw = raw[np.asarray(radar_indices, dtype=np.int64)]
    if not np.isfinite(raw).all():
        raise ValueError(f"non-finite CIR values at frame {start}")
    window = np.zeros((3, window_frames, 368), dtype=np.float32)
    window[:, :valid_count, :] = raw
    window = (window - mean[:, None, :]) / std[:, None, :]
    window[:, valid_count:, :] = 0.0
    return window, valid_count


def write_prediction_csv(records: list[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stream_id", "t_start", "t_end", "score"))
        writer.writeheader()
        for row in records:
            writer.writerow({
                "stream_id": row["stream_id"],
                "t_start": f"{float(row['t_start']):.6f}",
                "t_end": f"{float(row['t_end']):.6f}",
                "score": f"{float(row['score']):.8f}",
            })
    return target


def predict_split(
    model: CNNTemporalModel,
    data_root: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    normalizer: dict[str, np.ndarray],
    split: str,
    device: torch.device,
    checkpoint_label: str = "in-memory",
) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("only released train/val splits are supported")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    records: list[dict[str, Any]] = []
    window_frames = int(config["window_frames"])
    hop_frames = int(config["hop_frames"])
    batch_size = int(config["training"]["batch_size"])
    use_amp = bool(config["training"]["amp"])

    with torch.inference_mode():
        for path in list_stream_files(data_root, split):
            stream_id = f"{split}_{int(path.stem.split('_')[-1]):02d}"
            with h5py.File(path, "r") as handle:
                cir = handle["cir"]
                frame_count = int(cir.shape[1])
                if tuple(cir.shape) != (3, frame_count, 368):
                    raise ValueError(f"{path}: expected cir shape (3,T,368)")
                times = np.asarray(handle["t"][...], dtype=np.float64)
                if times.shape != (frame_count,) or not np.isfinite(times).all():
                    raise ValueError(f"{path}: invalid timestamp vector")
                if frame_count > 1 and np.any(np.diff(times) <= 0):
                    raise ValueError(f"{path}: timestamps must be strictly increasing")
                relative_times = times - times[0]
                serials = np.asarray(handle["radar_serials"][...]).reshape(-1)
                radar_indices = _radar_indices(serials, config["radar_order"])
                starts = window_starts(frame_count, window_frames, hop_frames)
                fall_sum = np.zeros(frame_count, dtype=np.float64)
                start_sum = np.zeros(frame_count, dtype=np.float64)
                end_sum = np.zeros(frame_count, dtype=np.float64)
                counts = np.zeros(frame_count, dtype=np.int32)

                for offset in range(0, len(starts), batch_size):
                    batch_starts = starts[offset : offset + batch_size]
                    windows: list[np.ndarray] = []
                    valid_counts: list[int] = []
                    for window_start in batch_starts:
                        window, valid_count = _normalized_window(
                            cir,
                            radar_indices,
                            window_start,
                            frame_count,
                            window_frames,
                            normalizer["mean"],
                            normalizer["std"],
                        )
                        windows.append(window)
                        valid_counts.append(valid_count)
                    tensor = torch.from_numpy(np.stack(windows)).to(device)
                    with _amp_context(device, use_amp):
                        outputs = model(tensor)
                    fall = torch.sigmoid(outputs["fall"]).float().cpu().numpy()
                    start_prob = torch.sigmoid(outputs["start"]).float().cpu().numpy() if "start" in outputs else None
                    end_prob = torch.sigmoid(outputs["end"]).float().cpu().numpy() if "end" in outputs else None
                    for row, (window_start, valid_count) in enumerate(zip(batch_starts, valid_counts)):
                        window_stop = window_start + valid_count
                        fall_sum[window_start:window_stop] += fall[row, :valid_count]
                        if start_prob is not None and end_prob is not None:
                            start_sum[window_start:window_stop] += start_prob[row, :valid_count]
                            end_sum[window_start:window_stop] += end_prob[row, :valid_count]
                        counts[window_start:window_stop] += 1

            if np.any(counts == 0):
                raise RuntimeError(f"window schedule left uncovered frames in {stream_id}")
            fall_probability = (fall_sum / counts).astype(np.float32)
            start_probability = (start_sum / counts).astype(np.float32) if config["variant"] == "v2" else None
            end_probability = (end_sum / counts).astype(np.float32) if config["variant"] == "v2" else None
            events = decode_events(
                fall_probability,
                relative_times,
                config,
                start_probability,
                end_probability,
                stream_duration_s=frame_count / float(config["sample_rate_hz"]),
            )
            for event in events:
                records.append({"stream_id": stream_id, **event})

    prediction_path = output / f"{split}_predictions.csv"
    write_prediction_csv(records, prediction_path)
    write_json(
        output / "prediction_manifest.json",
        {
            "split": split,
            "variant": config["variant"],
            "checkpoint": checkpoint_label,
            "prediction_file": prediction_path.name,
            "columns": ["stream_id", "t_start", "t_end", "score"],
            "t_alarm_written": False,
            "inference_mode": "offline",
        },
    )
    return prediction_path


def load_checkpoint(
    checkpoint_path: str | Path,
) -> tuple[CNNTemporalModel, dict[str, Any], dict[str, np.ndarray], dict[str, Any], torch.device]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = payload["config"]
    device = resolve_device(str(config["training"]["device"]))
    normalizer = {
        "mean": np.asarray(payload["normalizer"]["mean"], dtype=np.float32),
        "std": np.asarray(payload["normalizer"]["std"], dtype=np.float32),
    }
    model = CNNTemporalModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model, config, normalizer, payload, device

