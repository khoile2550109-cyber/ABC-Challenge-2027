from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def load_official_evaluator(evaluator_root: str | Path) -> ModuleType:
    source = Path(evaluator_root).expanduser().resolve() / "src" / "abc2027" / "evaluate.py"
    if not source.is_file():
        raise FileNotFoundError(f"official evaluator not found at {source}")
    spec = importlib.util.spec_from_file_location("abc2027_official_evaluate", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_validation(
    evaluator_root: str | Path,
    data_root: str | Path,
    prediction_path: str | Path,
) -> dict[str, Any]:
    evaluator = load_official_evaluator(evaluator_root)
    root = Path(data_root)
    validation_streams = root / "streams" / "val"
    durations = evaluator.stream_durations(validation_streams)
    if not durations or any(not stream_id.startswith("val_") for stream_id in durations):
        raise ValueError("official evaluation must receive only streams/val")
    predictions = evaluator.validate_predictions(prediction_path, durations)
    ground_truth = evaluator.load_ground_truth(root / "streams_events.csv", ignore_flagged=True)
    return evaluator.evaluate(ground_truth, predictions, durations, theta=0.3)


def ranking_key(report: dict[str, Any]) -> tuple[float, float, float]:
    rank = report["ranking_score"]
    f1 = float(rank["f1_mean"])
    start_error = rank["start_error_median_abs"]
    false_alarms = rank["false_alarms_per_hour"]
    start_value = float(start_error) if start_error is not None else float("inf")
    false_value = float(false_alarms) if false_alarms is not None else float("inf")
    return (f1, -start_value, -false_value)
