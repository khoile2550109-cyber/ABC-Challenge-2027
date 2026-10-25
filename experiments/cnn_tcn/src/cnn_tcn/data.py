from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_EVENT_COLUMNS = {"stream_id", "t_start", "t_end"}
IGNORE_MARGIN_DEFAULT_S = 2.0


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _serial_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def stream_id_for(path: Path) -> str:
    split = path.parent.name
    if split not in {"train", "val"}:
        raise ValueError(f"only train and val streams are supported, got {path}")
    index = path.stem.removeprefix("stream_")
    if not index.isdigit():
        raise ValueError(f"unexpected stream filename: {path.name}")
    return f"{split}_{int(index):02d}"


def list_stream_files(data_root: str | Path, split: str) -> list[Path]:
    if split not in {"train", "val"}:
        raise ValueError("split must be train or val; test streams are not available to this experiment")
    folder = Path(data_root) / "streams" / split
    files = sorted(folder.glob("stream_*.h5"))
    if not files:
        raise FileNotFoundError(f"no stream_*.h5 files found under {folder}")
    return files


def load_events(path: str | Path, ignore_flagged: bool = True) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_EVENT_COLUMNS - fields
        if missing:
            raise ValueError(f"{path} is missing event columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            stream_id = (row.get("stream_id") or "").strip()
            start, end = float(row["t_start"]), float(row["t_end"])
            if not stream_id or not np.isfinite(start) or not np.isfinite(end) or start >= end:
                raise ValueError(f"invalid event at {path}:{row_number}")
            flagged = (row.get("quality") or "").strip().lower() == "low" or parse_bool(row.get("needs_review", ""))
            ignored = parse_bool(row.get("ignore", "")) or (ignore_flagged and flagged)
            grouped[stream_id].append({"start": start, "end": end, "ignore": ignored})
    return grouped


def window_starts(frame_count: int, window_frames: int, hop_frames: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("stream contains no frames")
    if window_frames <= 0 or hop_frames <= 0:
        raise ValueError("window_frames and hop_frames must be positive")
    if frame_count <= window_frames:
        return [0]
    last = frame_count - window_frames
    starts = list(range(0, last + 1, hop_frames))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _radar_indices(serials: Iterable[Any], radar_order: Iterable[Any]) -> list[int]:
    found = [_serial_text(x) for x in serials]
    expected = [_serial_text(x) for x in radar_order]
    if len(found) != 3 or len(expected) != 3 or set(found) != set(expected):
        raise ValueError(f"radar serial mismatch: file={found}, config={expected}")
    return [found.index(serial) for serial in expected]


def _stream_metadata(path: Path, radar_order: Iterable[Any]) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        if "cir" not in handle or "t" not in handle or "radar_serials" not in handle:
            raise ValueError(f"{path} must contain cir, t and radar_serials")
        shape = tuple(handle["cir"].shape)
        if len(shape) != 3 or shape[0] != 3 or shape[2] != 368:
            raise ValueError(f"{path}: expected cir shape (3,T,368), got {shape}")
        if handle["t"].shape != (shape[1],):
            raise ValueError(f"{path}: t must have shape ({shape[1]},)")
        serials = [_serial_text(x) for x in np.asarray(handle["radar_serials"][...]).reshape(-1)]
        indices = _radar_indices(serials, radar_order)
        if shape[1] == 0:
            raise ValueError(f"{path}: empty stream")
        origin = float(handle["t"][0])
    return {"path": path, "stream_id": stream_id_for(path), "frames": shape[1], "origin": origin, "radar_indices": indices}


def _relative_times(raw: np.ndarray, origin: float, valid_count: int, sample_rate_hz: float) -> np.ndarray:
    result = np.asarray(raw, dtype=np.float64) - origin
    if valid_count:
        valid = result[:valid_count]
        if not np.isfinite(valid).all() or (len(valid) > 1 and np.any(np.diff(valid) <= 0)):
            raise ValueError("timestamps must be finite and strictly increasing")
    if valid_count < len(result):
        step = 1.0 / sample_rate_hz
        last = result[valid_count - 1] if valid_count else 0.0
        result[valid_count:] = last + step * np.arange(1, len(result) - valid_count + 1)
    return result.astype(np.float32)


def build_window_targets(
    times: np.ndarray,
    valid_mask: np.ndarray,
    events: list[dict[str, Any]],
    boundary_sigma_s: float,
    ignore_margin_s: float = IGNORE_MARGIN_DEFAULT_S,
) -> dict[str, np.ndarray]:
    length = len(times)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != (length,):
        raise ValueError("valid_mask and times must have matching one-dimensional shapes")
    fall = np.zeros(length, dtype=np.float32)
    start_target = np.zeros(length, dtype=np.float32)
    end_target = np.zeros(length, dtype=np.float32)
    ignore_zone = np.zeros(length, dtype=bool)
    if not valid.any():
        return {
            "fall": fall,
            "start": start_target,
            "end": end_target,
            "loss_mask": np.zeros(length, dtype=bool),
        }

    real_times = np.asarray(times, dtype=np.float64)
    valid_times = real_times[valid]
    step = float(np.median(np.diff(valid_times))) if len(valid_times) > 1 else 1.0 / 50.0
    low, high = float(valid_times[0]), float(valid_times[-1] + step)

    def add_soft_boundary(target: np.ndarray, boundary: float) -> None:
        if low <= boundary <= high:
            distances = np.abs(real_times - boundary)
            support = valid & (distances <= 3.0 * boundary_sigma_s)
            values = np.exp(-0.5 * (distances / boundary_sigma_s) ** 2).astype(np.float32)
            target[support] = np.maximum(target[support], values[support])

    positive_region = np.zeros(length, dtype=bool)
    boundary_region = np.zeros(length, dtype=bool)
    for event in events:
        event_start, event_end = float(event["start"]), float(event["end"])
        if event["ignore"]:
            ignore_zone |= valid & (real_times >= event_start - ignore_margin_s) & (real_times < event_end + ignore_margin_s)
            continue
        active = valid & (real_times >= event_start) & (real_times < event_end)
        fall[active] = 1.0
        positive_region |= active
        before_start = start_target.copy()
        before_end = end_target.copy()
        add_soft_boundary(start_target, event_start)
        add_soft_boundary(end_target, event_end)
        boundary_region |= (start_target > before_start) | (end_target > before_end)

    loss_mask = valid & (~ignore_zone | positive_region | boundary_region)
    return {"fall": fall, "start": start_target, "end": end_target, "loss_mask": loss_mask}


class H5WindowDataset(Dataset):
    """Lazy HDF5 window reader. Each worker process maintains its own file handles."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        config: dict[str, Any],
        normalizer: dict[str, np.ndarray],
        with_targets: bool = True,
    ) -> None:
        self.config = config
        self.split = split
        self.window_frames = int(config["window_frames"])
        self.sample_rate_hz = float(config["sample_rate_hz"])
        self.with_targets = with_targets
        self.normalizer = normalizer
        self.ignore_margin_s = float(config["loss"]["ignore_margin_s"])
        self.boundary_sigma_s = float(config["loss"]["boundary_sigma_s"])
        self.events = load_events(Path(data_root) / "streams_events.csv", bool(config["loss"]["ignore_flagged"])) if with_targets else {}
        self.streams = [_stream_metadata(path, config["radar_order"]) for path in list_stream_files(data_root, split)]
        self.descriptors: list[tuple[int, int]] = []
        self._handles: dict[str, h5py.File] = {}
        self._pid: int | None = None
        for stream_index, info in enumerate(self.streams):
            for start in window_starts(info["frames"], self.window_frames, int(config["hop_frames"])):
                self.descriptors.append((stream_index, start))

    def __len__(self) -> int:
        return len(self.descriptors)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_pid"] = None
        return state

    def _handle(self, path: Path) -> h5py.File:
        pid = os.getpid()
        if self._pid != pid:
            self.close()
            self._pid = pid
        key = str(path)
        if key not in self._handles:
            self._handles[key] = h5py.File(path, "r")
        return self._handles[key]

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        self._handles = {}

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        stream_index, start = self.descriptors[index]
        info = self.streams[stream_index]
        handle = self._handle(info["path"])
        stop = min(start + self.window_frames, info["frames"])
        valid_count = stop - start
        raw = np.asarray(handle["cir"][:, start:stop, :], dtype=np.float32)
        raw = raw[np.asarray(info["radar_indices"], dtype=np.int64)]
        x = np.zeros((3, self.window_frames, 368), dtype=np.float32)
        x[:, :valid_count, :] = raw
        if not np.isfinite(x[:, :valid_count]).all():
            raise ValueError(f"non-finite CIR values in {info['stream_id']} at frame {start}")
        x = (x - self.normalizer["mean"][:, None, :]) / self.normalizer["std"][:, None, :]
        x[:, valid_count:, :] = 0.0

        raw_times = np.asarray(handle["t"][start:stop], dtype=np.float64)
        times = np.zeros(self.window_frames, dtype=np.float32)
        times[:valid_count] = _relative_times(raw_times, info["origin"], valid_count, self.sample_rate_hz)
        times = _relative_times(times, 0.0, valid_count, self.sample_rate_hz)
        valid = np.zeros(self.window_frames, dtype=bool)
        valid[:valid_count] = True

        item: dict[str, Any] = {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "times": torch.from_numpy(times),
            "valid_mask": torch.from_numpy(valid),
            "stream_id": info["stream_id"],
            "window_start": start,
        }
        if self.with_targets:
            targets = build_window_targets(
                times,
                valid,
                self.events.get(info["stream_id"], []),
                self.boundary_sigma_s,
                self.ignore_margin_s,
            )
            for name, value in targets.items():
                item[f"{name}_target" if name != "loss_mask" else "loss_mask"] = torch.from_numpy(value)
        return item


def fit_normalizer(data_root: str | Path, config: dict[str, Any], chunk_frames: int = 4096) -> dict[str, np.ndarray]:
    """Streaming population mean/std by ordered radar and range bin, using train only."""
    total = np.zeros((3, 368), dtype=np.float64)
    square_total = np.zeros((3, 368), dtype=np.float64)
    count = 0
    for path in list_stream_files(data_root, "train"):
        info = _stream_metadata(path, config["radar_order"])
        with h5py.File(path, "r") as handle:
            cir = handle["cir"]
            for start in range(0, info["frames"], chunk_frames):
                stop = min(start + chunk_frames, info["frames"])
                block = np.asarray(cir[:, start:stop, :], dtype=np.float64)
                block = block[np.asarray(info["radar_indices"], dtype=np.int64)]
                if not np.isfinite(block).all():
                    raise ValueError(f"non-finite CIR values in {info['stream_id']} at frame {start}")
                total += block.sum(axis=1)
                square_total += np.square(block).sum(axis=1)
                count += block.shape[1]
    if count == 0:
        raise ValueError("cannot fit normalizer: no training frames")
    mean = total / count
    variance = np.maximum(square_total / count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def inspect_data(data_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(data_root)
    event_map = load_events(root / "streams_events.csv", bool(config["loss"]["ignore_flagged"]))
    result: dict[str, Any] = {"data_root": str(root), "splits": {}, "warnings": []}
    known_ids: set[str] = set()
    for split in ("train", "val"):
        files = list_stream_files(root, split)
        split_info = []
        for path in files:
            info = _stream_metadata(path, config["radar_order"])
            with h5py.File(path, "r") as handle:
                times = np.asarray(handle["t"][...], dtype=np.float64)
                if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                    raise ValueError(f"{path}: timestamps must be finite and strictly increasing")
                if len(times) > 1:
                    rate = 1.0 / float(np.median(np.diff(times)))
                    if not np.isclose(rate, float(config["sample_rate_hz"]), rtol=0.02, atol=0.05):
                        result["warnings"].append(f"{path}: observed timestamp rate {rate:.4f} Hz")
            known_ids.add(info["stream_id"])
            split_info.append({"stream_id": info["stream_id"], "frames": info["frames"], "seconds": info["frames"] / float(config["sample_rate_hz"])})
        result["splits"][split] = {"stream_count": len(split_info), "streams": split_info}
    unknown = sorted(set(event_map) - known_ids)
    if unknown:
        raise ValueError(f"annotations refer to streams outside train/val: {unknown}")
    result["annotation_rows_by_stream"] = {key: len(value) for key, value in sorted(event_map.items()) if key in known_ids}
    result["ignored_rows"] = sum(sum(bool(event["ignore"]) for event in rows) for rows in event_map.values())
    return result


def validate_paths(data_root: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    data = Path(data_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {data}")
    if output == data or data in output.parents:
        raise ValueError("output-dir must not be the data root or a child of it")
    return data, output


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


