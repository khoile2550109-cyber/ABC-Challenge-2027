from __future__ import annotations

import csv
from pathlib import Path

from cnn_tcn.inference import write_prediction_csv


def test_prediction_csv_has_only_official_required_columns(tmp_path: Path) -> None:
    target = tmp_path / "val_predictions.csv"
    write_prediction_csv(
        [{"stream_id": "val_00", "t_start": 1.0, "t_end": 2.0, "score": 0.75}],
        target,
    )
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["stream_id", "t_start", "t_end", "score"]
        rows = list(reader)
    assert rows == [{"stream_id": "val_00", "t_start": "1.000000", "t_end": "2.000000", "score": "0.75000000"}]
