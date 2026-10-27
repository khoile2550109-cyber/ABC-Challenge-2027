# %% [markdown]
# CELL 0 - ABC2027 UWB FALL DETECTION PIPELINE
#
# Quy trình gồm:
# 1. Zoom-in EDA và trực quan hóa raw/filtered signal.
# 2. Chebychev filtering, sliding windows và gán nhãn theo streams_events.csv.
# 3. Huấn luyện ResNet-18 với train/validation monitoring.
# 4. Inference, HMM-Viterbi và NMS.
# 5. Official metric, tuning hậu xử lý và xuất kết quả.
#
# Có thể chạy từng cell trong Kaggle Notebook hoặc dùng các hàm
# run_notebook_* ở cuối file.

"""ABC2027 UWB fall detection pipeline.

The pipeline is intentionally command based:

    python pipeline.py zoom --stream-id train_00
    python pipeline.py train --epochs 5
    python pipeline.py predict --split val
    python pipeline.py evaluate --prediction-path /kaggle/working/submission.csv

The HDF5 streams contain ``cir`` with shape ``(3, frames, 368)``. Ground-truth
events are read from ``streams_events.csv`` and use stream ids such as
``train_00`` and ``val_01``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


# %% [markdown]
# CELL 1 - PATHS, CONSTANTS AND REPRODUCIBILITY

# %%
# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
KAGGLE_DATASET_ROOT = Path(
    "/kaggle/input/datasets/hieu1lth/abc-2027-all-dataset/abc2027_uwb_train_v1"
)
KAGGLE_TRAIN_STREAMS_DIR = KAGGLE_DATASET_ROOT / "streams" / "train"
KAGGLE_VAL_STREAMS_DIR = KAGGLE_DATASET_ROOT / "streams" / "val"
KAGGLE_TEST_STREAMS_DIR = KAGGLE_DATASET_ROOT / "streams" / "test"
KAGGLE_SEGMENTS_DIR = KAGGLE_DATASET_ROOT / "segments"
KAGGLE_EVENTS_PATH = KAGGLE_DATASET_ROOT / "streams_events.csv"
KAGGLE_COMPOSITION_PATH = KAGGLE_DATASET_ROOT / "streams_composition.csv"
KAGGLE_SAMPLE_SUBMISSION_PATH = KAGGLE_DATASET_ROOT / "sample_submission.csv"
KAGGLE_OUTPUT_DIR = Path("/kaggle/working")
KAGGLE_CHECKPOINT_DIR = KAGGLE_OUTPUT_DIR / "checkpoints"
KAGGLE_SUBMISSION_PATH = KAGGLE_OUTPUT_DIR / "submission.csv"
KAGGLE_ZOOM_DIR = KAGGLE_OUTPUT_DIR / "eda_zoom"
KAGGLE_PLOT_DIR = KAGGLE_OUTPUT_DIR / "plots"

RATE_HZ = 50.0
WINDOW_FRAMES = 200
HOP_FRAMES = 50
WINDOW_SECONDS = WINDOW_FRAMES / RATE_HZ
HOP_SECONDS = HOP_FRAMES / RATE_HZ
FILTER_LOW_HZ = 1.0
FILTER_HIGH_HZ = 5.0
SEED = 2027


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stream_path(stream_id: str) -> Path:
    parts = stream_id.split("_")
    if len(parts) != 2 or parts[0] not in {"train", "val", "test"} or not parts[1].isdigit():
        raise ValueError(f"Invalid stream id: {stream_id!r}; expected train_00, val_00 or test_00")
    folders = {
        "train": KAGGLE_TRAIN_STREAMS_DIR,
        "val": KAGGLE_VAL_STREAMS_DIR,
        "test": KAGGLE_TEST_STREAMS_DIR,
    }
    folder = folders[parts[0]]
    path = folder / f"stream_{int(parts[1]):02d}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Stream not found: {path}")
    return path


def stream_id_from_path(path: Path) -> str:
    split = path.parent.name
    return f"{split}_{int(path.stem.split('_')[1]):02d}"


def list_stream_ids(split: str) -> list[str]:
    folders = {
        "train": KAGGLE_TRAIN_STREAMS_DIR,
        "val": KAGGLE_VAL_STREAMS_DIR,
        "test": KAGGLE_TEST_STREAMS_DIR,
    }
    if split not in folders:
        raise ValueError(f"Unsupported split: {split!r}")
    folder = folders[split]
    paths = sorted(folder.glob("stream_*.h5"))
    if not paths:
        if split == "test":
            raise FileNotFoundError(
                f"No test streams found in {folder}. "
                "The published ABC2027 package does not include official test streams."
            )
        raise FileNotFoundError(f"No HDF5 streams found in {folder}")
    return [stream_id_from_path(path) for path in paths]


# %% [markdown]
# CELL 2 - DATASET LOADING AND EVENT LABELS

# %%
# ---------------------------------------------------------------------------
# Dataset loading and labels
# ---------------------------------------------------------------------------
def load_stream(path_or_id: str | Path) -> np.ndarray:
    path = stream_path(path_or_id) if isinstance(path_or_id, str) else Path(path_or_id)
    if not path.is_file():
        raise FileNotFoundError(f"Stream not found: {path}")
    with h5py.File(path, "r") as handle:
        if "cir" not in handle:
            raise KeyError(f"{path} does not contain HDF5 dataset 'cir'")
        cir = np.asarray(handle["cir"], dtype=np.float32)
    if cir.ndim != 3 or cir.shape[0] != 3 or cir.shape[2] != 368:
        raise ValueError(f"Expected cir shape (3, frames, 368), got {cir.shape}")
    return cir


def load_events(path: Path = KAGGLE_EVENTS_PATH) -> dict[str, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    events: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event = {
                "stream_id": row["stream_id"],
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "ignore": row.get("ignore", "").strip().lower() in {"true", "1", "yes"},
                "quality": row.get("quality", "").strip().lower(),
                "needs_review": row.get("needs_review", "").strip().lower()
                in {"true", "1", "yes"},
            }
            events.setdefault(event["stream_id"], []).append(event)
    return events


def event_intervals(
    events: Iterable[dict],
    *,
    ignore_flagged: bool = False,
) -> list[tuple[float, float]]:
    intervals = []
    for event in events:
        ignored = event["ignore"] or (
            ignore_flagged and (event["quality"] == "low" or event["needs_review"])
        )
        if not ignored:
            intervals.append((event["t_start"], event["t_end"]))
    return intervals


# %% [markdown]
# CELL 3 - ZOOM-IN EDA AND PREPROCESSING

# %%
# ---------------------------------------------------------------------------
# EDA and preprocessing
# ---------------------------------------------------------------------------
def chebychev_filter(cir: np.ndarray) -> np.ndarray:
    nyquist = RATE_HZ / 2.0
    coefficients = signal.cheby1(
        N=4,
        rp=0.5,
        Wn=[FILTER_LOW_HZ / nyquist, FILTER_HIGH_HZ / nyquist],
        btype="band",
    )
    return signal.filtfilt(*coefficients, cir, axis=1).astype(np.float32)


def perform_zoom_eda(
    stream_id: str,
    event_index: int = 0,
    margin_seconds: float = 5.0,
    output_dir: Path = KAGGLE_ZOOM_DIR,
) -> Path:
    events_by_stream = load_events()
    events = events_by_stream.get(stream_id, [])
    usable_events = [event for event in events if not event["ignore"]]
    if not usable_events:
        raise ValueError(f"No usable labelled events found for {stream_id}")
    if event_index < 0 or event_index >= len(usable_events):
        raise IndexError(f"event_index must be in [0, {len(usable_events) - 1}]")

    cir = load_stream(stream_id)
    filtered = chebychev_filter(cir)
    event = usable_events[event_index]
    start = max(0.0, event["t_start"] - margin_seconds)
    end = min(cir.shape[1] / RATE_HZ, event["t_end"] + margin_seconds)
    first = int(start * RATE_HZ)
    last = min(cir.shape[1], int(math.ceil(end * RATE_HZ)))
    times = np.arange(first, last) / RATE_HZ

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stream_id}_event_{event_index:04d}_zoom.png"
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    for radar in range(3):
        raw = cir[radar, first:last]
        filt = filtered[radar, first:last]
        raw_image = axes[radar, 0].imshow(
            raw.T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], 0, cir.shape[2]],
            cmap="viridis",
        )
        axes[radar, 0].axvspan(event["t_start"], event["t_end"], color="red", alpha=0.2)
        axes[radar, 0].set_title(f"Radar {radar + 1} - raw")
        axes[radar, 0].set_ylabel("Range bin")
        fig.colorbar(raw_image, ax=axes[radar, 0], shrink=0.8)
        filtered_image = axes[radar, 1].imshow(
            filt.T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], 0, cir.shape[2]],
            cmap="magma",
        )
        axes[radar, 1].axvspan(event["t_start"], event["t_end"], color="cyan", alpha=0.2)
        axes[radar, 1].set_title(f"Radar {radar + 1} - Chebychev 1-5 Hz")
        fig.colorbar(filtered_image, ax=axes[radar, 1], shrink=0.8)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{stream_id}: event {event_index} [{event['t_start']:.2f}, {event['t_end']:.2f}] s"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved zoom EDA: {output_path}")
    return output_path


def window_bounds(frame_count: int) -> list[tuple[int, int]]:
    if frame_count < WINDOW_FRAMES:
        return []
    return [
        (start, start + WINDOW_FRAMES)
        for start in range(0, frame_count - WINDOW_FRAMES + 1, HOP_FRAMES)
    ]


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - overlap
    return overlap / union if union > 0 else 0.0


def window_labels(
    bounds: list[tuple[int, int]],
    events: Iterable[tuple[float, float]],
    positive_iou: float = 0.1,
) -> np.ndarray:
    intervals = list(events)
    labels = np.zeros(len(bounds), dtype=np.int64)
    for index, (first, last) in enumerate(bounds):
        start, end = first / RATE_HZ, last / RATE_HZ
        labels[index] = int(
            any(temporal_iou(start, end, event_start, event_end) >= positive_iou
                for event_start, event_end in intervals)
        )
    return labels


class WindowDataset(Dataset):
    def __init__(self, windows: list[tuple[np.ndarray, int]]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window, label = self.windows[index]
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.long)


class StreamWindowDataset(Dataset):
    """Read one filtered stream at a time instead of materializing all windows."""

    def __init__(self, records: list[tuple[str, int, int, int]]) -> None:
        self.records = records
        self._stream_id: str | None = None
        self._filtered: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        stream_id, first, last, label = self.records[index]
        if stream_id != self._stream_id:
            self._filtered = chebychev_filter(load_stream(stream_id))
            self._stream_id = stream_id
        assert self._filtered is not None
        return (
            torch.from_numpy(self._filtered[:, first:last, :]),
            torch.tensor(label, dtype=torch.long),
        )


def make_training_windows(stream_ids: Iterable[str]) -> list[tuple[str, int, int, int]]:
    events_by_stream = load_events()
    all_windows: list[tuple[str, int, int, int]] = []
    for stream_id in stream_ids:
        frame_count = load_stream(stream_id).shape[1]
        bounds = window_bounds(frame_count)
        labels = window_labels(
            bounds,
            event_intervals(events_by_stream.get(stream_id, [])),
        )
        for (first, last), label in zip(bounds, labels):
            all_windows.append((stream_id, first, last, int(label)))
        print(
            f"[INFO] {stream_id}: windows={len(bounds)}, "
            f"positive={int(labels.sum())}"
        )
    if not all_windows or len({label for _, label in all_windows}) < 2:
        raise ValueError("Training windows must contain both negative and positive labels")
    return all_windows


# %% [markdown]
# CELL 4 - WINDOW DATASET AND RESNET-18 BACKBONE

# %%
# ---------------------------------------------------------------------------
# Backbone, training and prediction
# ---------------------------------------------------------------------------
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def evaluate_window_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += int((outputs.argmax(dim=1) == labels).sum())
            count += len(labels)
    if count == 0:
        raise ValueError("Validation loader is empty")
    return total_loss / count, correct / count


def train_model(
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    checkpoint_path: Path = KAGGLE_CHECKPOINT_DIR / "resnet18_abc2027.pt",
) -> Path:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_windows = make_training_windows(list_stream_ids("train"))
    val_windows = make_training_windows(list_stream_ids("val"))
    train_loader = DataLoader(
        StreamWindowDataset(train_windows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        StreamWindowDataset(val_windows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = ResNet18Backbone().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += int((outputs.argmax(dim=1) == labels).sum())
            count += len(labels)
        train_loss = total_loss / count
        train_accuracy = correct / count
        val_loss, val_accuracy = evaluate_window_model(
            model, val_loader, device, criterion
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"[TRAIN] epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.5f} train_accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.5f} val_accuracy={val_accuracy:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "rate_hz": RATE_HZ,
                    "history": history,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_plot(history, KAGGLE_PLOT_DIR / "training_history.png")
    (KAGGLE_OUTPUT_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def save_training_plot(history: dict[str, list[float]], output_path: Path) -> Path:
    """Save loss and accuracy curves from the training run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved training plot: {output_path}")
    return output_path


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = ResNet18Backbone().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def model_scores(
    stream_id: str,
    checkpoint_path: Path,
    batch_size: int = 64,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_count = load_stream(stream_id).shape[1]
    bounds = window_bounds(frame_count)
    records = [(stream_id, first, last, 0) for first, last in bounds]
    loader = DataLoader(
        StreamWindowDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = load_model(checkpoint_path, device)
    scores: list[float] = []
    with torch.no_grad():
        for inputs, _ in loader:
            probabilities = torch.softmax(model(inputs.to(device)), dim=1)[:, 1]
            scores.extend(probabilities.cpu().numpy().tolist())
    return [(first / RATE_HZ, last / RATE_HZ) for first, last in bounds], np.asarray(scores)


# %% [markdown]
# CELL 5 - HMM-VITERBI, EVENT EXTRACTION AND NMS

# %%
# ---------------------------------------------------------------------------
# HMM and event post-processing
# ---------------------------------------------------------------------------
class ViterbiDecoder:
    def __init__(self) -> None:
        self.transition = np.asarray(
            [[0.97, 0.03, 0.00], [0.02, 0.80, 0.18], [0.03, 0.00, 0.97]],
            dtype=np.float64,
        )

    def decode(self, fall_probabilities: np.ndarray) -> np.ndarray:
        if len(fall_probabilities) == 0:
            return np.empty(0, dtype=np.int64)
        probability = np.clip(fall_probabilities, 1e-6, 1.0 - 1e-6)
        emissions = np.column_stack(
            [1.0 - probability, probability * 0.7, probability * 0.3]
        )
        log_transition = np.log(np.maximum(self.transition, 1e-12))
        log_emissions = np.log(np.maximum(emissions, 1e-12))
        values = np.full((len(probability), 3), -np.inf)
        backpointers = np.zeros((len(probability), 3), dtype=np.int64)
        values[0] = np.log(np.asarray([0.9, 0.1, 1e-6])) + log_emissions[0]
        for index in range(1, len(probability)):
            candidates = values[index - 1][:, None] + log_transition
            backpointers[index] = np.argmax(candidates, axis=0)
            values[index] = candidates[backpointers[index], np.arange(3)] + log_emissions[index]
        states = np.empty(len(probability), dtype=np.int64)
        states[-1] = int(np.argmax(values[-1]))
        for index in range(len(states) - 2, -1, -1):
            states[index] = backpointers[index + 1, states[index + 1]]
        return states


def states_to_events(
    bounds: list[tuple[float, float]],
    states: np.ndarray,
    scores: np.ndarray,
    min_duration: float = 0.8,
    nms_iou_threshold: float = 0.3,
) -> list[dict]:
    active = np.isin(states, [1, 2])
    indices = np.flatnonzero(active)
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0]), int(indices[0])]]
    for index in indices[1:]:
        if bounds[int(index)][0] - bounds[groups[-1][1]][1] <= HOP_SECONDS:
            groups[-1][1] = int(index)
        else:
            groups.append([int(index), int(index)])
    result = []
    for first, last in groups:
        start, end = bounds[first][0], bounds[last][1]
        if end - start >= min_duration:
            result.append(
                {"t_start": start, "t_end": end, "score": float(np.max(scores[first:last + 1]))}
            )
    return non_maximum_suppression(result, nms_iou_threshold)


def non_maximum_suppression(events: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    kept: list[dict] = []
    for event in sorted(events, key=lambda item: item["score"], reverse=True):
        if all(
            temporal_iou(event["t_start"], event["t_end"], other["t_start"], other["t_end"])
            < iou_threshold
            for other in kept
        ):
            kept.append(event)
    return sorted(kept, key=lambda item: item["t_start"])


def save_prediction_diagnostic(
    stream_id: str,
    bounds: list[tuple[float, float]],
    scores: np.ndarray,
    output_dir: Path = KAGGLE_PLOT_DIR,
) -> Path:
    """Plot window scores and labelled intervals for train/validation streams."""
    events = load_events().get(stream_id, [])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stream_id}_prediction_diagnostics.png"
    centers = np.asarray([(start + end) / 2.0 for start, end in bounds])
    labels = window_labels(
        [(int(start * RATE_HZ), int(end * RATE_HZ)) for start, end in bounds],
        event_intervals(events),
    )
    predicted = scores >= 0.5
    false_positive = (predicted == 1) & (labels == 0)
    false_negative = (predicted == 0) & (labels == 1)
    fig, axis = plt.subplots(figsize=(16, 4))
    axis.plot(centers, scores, linewidth=0.8, label="P(fall)")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="0.5")
    axis.scatter(
        centers[false_positive],
        scores[false_positive],
        color="red",
        s=12,
        label="false positive window",
    )
    axis.scatter(
        centers[false_negative],
        scores[false_negative],
        color="purple",
        s=12,
        label="false negative window",
    )
    for event in events:
        if not event["ignore"]:
            axis.axvspan(event["t_start"], event["t_end"], color="green", alpha=0.2)
    axis.set_title(f"{stream_id}: window fall probabilities and ground truth")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("P(fall)")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(
        f"[INFO] {stream_id}: false_positive_windows={int(false_positive.sum())}, "
        f"false_negative_windows={int(false_negative.sum())}"
    )
    return output_path


def predict_split(
    split: str,
    checkpoint_path: Path,
    output_path: Path = KAGGLE_SUBMISSION_PATH,
    min_duration: float = 0.8,
    nms_iou_threshold: float = 0.3,
) -> Path:
    predictions = []
    for stream_id in list_stream_ids(split):
        bounds, scores = model_scores(stream_id, checkpoint_path)
        if split != "test":
            diagnostic_path = save_prediction_diagnostic(stream_id, bounds, scores)
            print(f"[INFO] Saved prediction diagnostic: {diagnostic_path}")
        states = ViterbiDecoder().decode(scores)
        for event in states_to_events(
            bounds,
            states,
            scores,
            min_duration=min_duration,
            nms_iou_threshold=nms_iou_threshold,
        ):
            predictions.append({"stream_id": stream_id, **event, "t_alarm": event["t_start"]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"]
        )
        writer.writeheader()
        writer.writerows(predictions)
    print(f"[INFO] Saved {len(predictions)} predictions: {output_path}")
    return output_path


def tune_postprocessing(
    checkpoint_path: Path,
    split: str = "val",
    min_durations: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2),
    nms_thresholds: tuple[float, ...] = (0.2, 0.3, 0.5),
) -> dict:
    """Select post-processing parameters by official validation ranking score."""
    if split != "val":
        raise ValueError("Tuning must use the validation split, not train or test")
    cached_scores = {
        stream_id: model_scores(stream_id, checkpoint_path)
        for stream_id in list_stream_ids(split)
    }
    best: dict | None = None
    for min_duration in min_durations:
        for nms_threshold in nms_thresholds:
            candidate_path = KAGGLE_OUTPUT_DIR / "tuning_candidate.csv"
            with candidate_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"],
                )
                writer.writeheader()
                for stream_id, (bounds, scores) in cached_scores.items():
                    states = ViterbiDecoder().decode(scores)
                    for event in states_to_events(
                        bounds,
                        states,
                        scores,
                        min_duration=min_duration,
                        nms_iou_threshold=nms_threshold,
                    ):
                        writer.writerow(
                            {
                                "stream_id": stream_id,
                                **event,
                                "t_alarm": event["t_start"],
                            }
                        )
            report = score_predictions(candidate_path, split="val", ignore_flagged=True)
            ranking = report["ranking_score"]["f1_mean"]
            if best is None or ranking > best["ranking_score"]:
                best = {
                    "min_duration": min_duration,
                    "nms_iou_threshold": nms_threshold,
                    "ranking_score": ranking,
                    "report": report,
                }
    if best is None:
        raise RuntimeError("No tuning candidates were evaluated")
    (KAGGLE_OUTPUT_DIR / "tuning_report.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    print(
        f"[TUNE] min_duration={best['min_duration']} "
        f"nms_iou_threshold={best['nms_iou_threshold']} "
        f"f1_mean={best['ranking_score']:.4f}"
    )
    return best


# %% [markdown]
# CELL 6 - OFFICIAL METRICS, TUNING AND NOTEBOOK RUNNERS

# %%
# ---------------------------------------------------------------------------
# Official-style event metric
# ---------------------------------------------------------------------------
def score_predictions(
    prediction_path: Path,
    split: str = "val",
    ignore_flagged: bool = False,
) -> dict:
    evaluator_src = KAGGLE_DATASET_ROOT / "src"
    if not evaluator_src.is_dir():
        raise FileNotFoundError(f"Official evaluator package not found: {evaluator_src}")
    if str(evaluator_src) not in sys.path:
        sys.path.insert(0, str(evaluator_src))
    from abc2027.evaluate import (
        evaluate,
        load_ground_truth,
        stream_durations,
        validate_predictions,
    )

    streams_dir = (
        KAGGLE_TRAIN_STREAMS_DIR if split == "train" else KAGGLE_VAL_STREAMS_DIR
    )
    durations = stream_durations(streams_dir)
    if not durations:
        raise FileNotFoundError(f"No streams found for split {split!r}: {streams_dir}")
    predictions = validate_predictions(prediction_path, durations)
    ground_truth = load_ground_truth(KAGGLE_EVENTS_PATH, ignore_flagged=ignore_flagged)
    return evaluate(ground_truth, predictions, durations)


def run_notebook_zoom(
    stream_id: str = "train_00",
    event_index: int = 0,
    margin_seconds: float = 5.0,
) -> Path:
    """Run zoom EDA directly from a Kaggle Notebook cell."""
    return perform_zoom_eda(stream_id, event_index, margin_seconds)


def run_notebook_train(
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> Path:
    """Train and save the ResNet-18 checkpoint from a Kaggle Notebook cell."""
    return train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        checkpoint_path=KAGGLE_CHECKPOINT_DIR / "resnet18_abc2027.pt",
    )


def run_notebook_predict(
    checkpoint_path: str | Path = KAGGLE_CHECKPOINT_DIR / "resnet18_abc2027.pt",
    split: str = "val",
) -> Path:
    """Generate a submission directly from a Kaggle Notebook cell."""
    return predict_split(split, Path(checkpoint_path), KAGGLE_SUBMISSION_PATH)


def run_notebook_all(
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> dict[str, Path]:
    """Run EDA-ready training, tuning, prediction and validation evaluation."""
    outputs: dict[str, Path] = {}
    outputs["zoom_eda"] = run_notebook_zoom(
        stream_id="train_00",
        event_index=0,
        margin_seconds=5.0,
    )
    checkpoint_path = run_notebook_train(epochs, batch_size, learning_rate)
    tuning = tune_postprocessing(checkpoint_path)
    best_duration = tuning["min_duration"]
    best_nms = tuning["nms_iou_threshold"]
    outputs["checkpoint"] = checkpoint_path
    outputs["train_submission"] = predict_split(
        "train",
        checkpoint_path,
        KAGGLE_OUTPUT_DIR / "train_submission.csv",
        min_duration=best_duration,
        nms_iou_threshold=best_nms,
    )
    outputs["val_submission"] = predict_split(
        "val",
        checkpoint_path,
        KAGGLE_OUTPUT_DIR / "val_submission.csv",
        min_duration=best_duration,
        nms_iou_threshold=best_nms,
    )
    val_report = score_predictions(
        KAGGLE_OUTPUT_DIR / "val_submission.csv",
        split="val",
        ignore_flagged=True,
    )
    (KAGGLE_OUTPUT_DIR / "val_report.json").write_text(
        json.dumps(val_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(val_report, indent=2))
    try:
        outputs["test_submission"] = predict_split(
            "test",
            checkpoint_path,
            KAGGLE_OUTPUT_DIR / "test_submission.csv",
            min_duration=best_duration,
            nms_iou_threshold=best_nms,
        )
    except FileNotFoundError as error:
        print(f"[INFO] Test prediction skipped: {error}")
    return outputs


def run_notebook_evaluate(
    prediction_path: str | Path = KAGGLE_SUBMISSION_PATH,
    split: str = "val",
    ignore_flagged: bool = True,
) -> dict:
    """Evaluate predictions directly from a Kaggle Notebook cell."""
    report = score_predictions(Path(prediction_path), split, ignore_flagged)
    print(json.dumps(report, indent=2))
    return report


# %% [markdown]
# CELL 7 - RUN THE COMPLETE PIPELINE IN KAGGLE
#
# Sau khi copy từng cell vào Kaggle, chạy:
#
# outputs = run_notebook_all(
#     epochs=5,
#     batch_size=32,
#     learning_rate=1e-4,
# )
# display(outputs)
#
# Cell này sẽ:
# - tạo ảnh Zoom-in EDA;
# - train ResNet-18 với train/validation;
# - lưu loss/accuracy;
# - tune min_duration và NMS IoU trên validation;
# - tạo train_submission.csv và val_submission.csv;
# - đánh giá validation bằng official evaluator;
# - dự đoán test nếu thư mục test tồn tại.


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ABC2027 UWB fall pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    zoom = subparsers.add_parser("zoom", help="Create raw/filtered close-up plots")
    zoom.add_argument("--stream-id", default="train_00")
    zoom.add_argument("--event-index", type=int, default=0)
    zoom.add_argument("--margin-seconds", type=float, default=5.0)

    train = subparsers.add_parser("train", help="Train the ResNet-18 window classifier")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--checkpoint-path", type=Path, default=KAGGLE_CHECKPOINT_DIR / "resnet18_abc2027.pt")

    predict = subparsers.add_parser("predict", help="Predict events and write submission CSV")
    predict.add_argument("--split", choices=("train", "val", "test"), default="val")
    predict.add_argument("--checkpoint-path", type=Path, required=True)
    predict.add_argument("--output-path", type=Path, default=KAGGLE_SUBMISSION_PATH)

    evaluate = subparsers.add_parser("evaluate", help="Calculate event-level metrics")
    evaluate.add_argument("--prediction-path", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "val"), default="val")
    evaluate.add_argument("--ignore-flagged", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "zoom":
        perform_zoom_eda(args.stream_id, args.event_index, args.margin_seconds)
    elif args.command == "train":
        train_model(args.epochs, args.batch_size, args.learning_rate, args.checkpoint_path)
    elif args.command == "predict":
        predict_split(args.split, args.checkpoint_path, args.output_path)
    elif args.command == "evaluate":
        print(json.dumps(score_predictions(args.prediction_path, args.split, args.ignore_flagged), indent=2))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in {
    "zoom",
    "train",
    "predict",
    "evaluate",
}:
    main()
