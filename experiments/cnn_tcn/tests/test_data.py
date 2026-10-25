from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from cnn_tcn.data import H5WindowDataset, build_window_targets, fit_normalizer, validate_paths, window_starts


RADARS = ["29171", "29444", "30722"]


def _write_stream(path: Path, value: float, frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cir = np.full((3, frames, 368), value, dtype=np.float32)
    times = np.arange(frames, dtype=np.float64) / 50.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cir", data=cir)
        handle.create_dataset("t", data=times)
        handle.create_dataset("radar_serials", data=np.asarray(RADARS, dtype="S8"))


def _config(window_frames: int = 10) -> dict:
    return {
        "sample_rate_hz": 50.0,
        "window_frames": window_frames,
        "hop_frames": 5,
        "radar_order": RADARS,
        "loss": {
            "boundary_sigma_s": 0.02,
            "ignore_margin_s": 0.02,
            "ignore_flagged": True,
        },
    }


def test_window_starts_covers_last_frame_without_extra_padding() -> None:
    assert window_starts(25, 10, 6) == [0, 6, 12, 15]
    assert window_starts(7, 10, 5) == [0]


def test_boundary_is_not_fabricated_when_event_crosses_window_edge() -> None:
    times = np.arange(5, dtype=np.float32) / 50.0
    valid = np.ones(5, dtype=bool)
    targets = build_window_targets(
        times,
        valid,
        [{"start": -0.04, "end": 0.06, "ignore": False}],
        boundary_sigma_s=0.01,
        ignore_margin_s=0.0,
    )
    assert targets["fall"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert not targets["start"].any()
    assert targets["end"].any()


def test_padding_and_ignore_regions_are_masked(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_stream(root / "streams" / "train" / "stream_00.h5", 1.0, frames=8)
    (root / "streams_events.csv").write_text(
        "stream_id,t_start,t_end,ignore,quality,needs_review\n"
        "train_00,0.02,0.06,false,,false\n"
        "train_00,0.10,0.14,true,,false\n",
        encoding="utf-8",
    )
    normalizer = {"mean": np.zeros((3, 368), dtype=np.float32), "std": np.ones((3, 368), dtype=np.float32)}
    dataset = H5WindowDataset(root, "train", _config(), normalizer, with_targets=True)
    item = dataset[0]
    assert item["x"].shape == (3, 10, 368)
    assert item["valid_mask"].tolist() == [True] * 8 + [False, False]
    assert item["loss_mask"][-2:].sum().item() == 0
    assert item["fall_target"][:3].sum().item() == 2
    assert item["loss_mask"][5].item() is False
    dataset.close()


def test_output_dir_cannot_be_inside_read_only_data_root(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    try:
        validate_paths(root, root / "results")
    except ValueError:
        return
    raise AssertionError("output directory inside data root must be rejected")


def test_normalizer_reads_only_train_streams(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_stream(root / "streams" / "train" / "stream_00.h5", 2.0, frames=4)
    _write_stream(root / "streams" / "val" / "stream_00.h5", 99.0, frames=4)
    normalizer = fit_normalizer(root, {"radar_order": RADARS}, chunk_frames=2)
    assert np.allclose(normalizer["mean"], 2.0)
    assert np.allclose(normalizer["std"], 1e-6)



