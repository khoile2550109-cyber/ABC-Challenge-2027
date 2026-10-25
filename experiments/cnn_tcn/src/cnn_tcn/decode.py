from __future__ import annotations

from typing import Any

import numpy as np


def _frame_edges(times: np.ndarray, sample_rate_hz: float, stream_duration_s: float | None) -> np.ndarray:
    if len(times) == 0:
        return np.zeros(0, dtype=np.float64)
    step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / sample_rate_hz
    edges = np.concatenate([np.asarray(times, dtype=np.float64), [float(times[-1]) + step]])
    if stream_duration_s is not None:
        if not np.isfinite(stream_duration_s) or stream_duration_s <= 0:
            raise ValueError("stream_duration_s must be finite and positive")
        edges[-1] = min(edges[-1], float(stream_duration_s))
    return edges


def _threshold_runs(probabilities: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    active = np.flatnonzero(np.asarray(probabilities) >= threshold)
    if not len(active):
        return []
    cuts = np.flatnonzero(np.diff(active) > 1)
    starts = np.concatenate(([active[0]], active[cuts + 1]))
    ends = np.concatenate((active[cuts] + 1, [active[-1] + 1]))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _merge_runs(
    runs: list[tuple[int, int]],
    edges: np.ndarray,
    merge_gap_s: float,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and float(edges[start] - edges[merged[-1][1]]) <= merge_gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(item[0], item[1]) for item in merged]


def _peak_in_time_radius(
    probabilities: np.ndarray,
    times: np.ndarray,
    target_time: float,
    radius_s: float,
    threshold: float,
) -> int | None:
    left = int(np.searchsorted(times, target_time - radius_s, side="left"))
    right = int(np.searchsorted(times, target_time + radius_s, side="right"))
    left = max(0, min(left, len(times)))
    right = max(left, min(right, len(times)))
    if left == right:
        return None
    index = left + int(np.argmax(probabilities[left:right]))
    return index if float(probabilities[index]) >= threshold else None


def decode_events(
    fall_probability: np.ndarray,
    times: np.ndarray,
    config: dict[str, Any],
    start_probability: np.ndarray | None = None,
    end_probability: np.ndarray | None = None,
    stream_duration_s: float | None = None,
) -> list[dict[str, float]]:
    fall = np.asarray(fall_probability, dtype=np.float64)
    time_values = np.asarray(times, dtype=np.float64)
    if fall.ndim != 1 or time_values.shape != fall.shape:
        raise ValueError("fall_probability and times must be matching one-dimensional arrays")
    if not np.isfinite(fall).all() or not np.isfinite(time_values).all():
        raise ValueError("decoder inputs must be finite")
    if len(fall) > 1 and np.any(np.diff(time_values) <= 0):
        raise ValueError("times must be strictly increasing")
    variant = config["variant"]
    decoder = config["decoder"]
    edges = _frame_edges(time_values, float(config["sample_rate_hz"]), stream_duration_s)
    runs = _threshold_runs(fall, float(decoder["fall_threshold"]))
    runs = _merge_runs(runs, edges, float(decoder["merge_gap_s"]))
    if variant == "v2":
        if start_probability is None or end_probability is None:
            raise ValueError("V2 requires start_probability and end_probability")
        start_prob = np.asarray(start_probability, dtype=np.float64)
        end_prob = np.asarray(end_probability, dtype=np.float64)
        if start_prob.shape != fall.shape or end_prob.shape != fall.shape:
            raise ValueError("V2 boundary probabilities must match fall_probability")
        if not np.isfinite(start_prob).all() or not np.isfinite(end_prob).all():
            raise ValueError("boundary probabilities must be finite")
    else:
        start_prob = end_prob = None

    result: list[dict[str, float]] = []
    for first, stop in runs:
        start_index, end_index = first, stop - 1
        if variant == "v2" and len(time_values):
            radius = float(decoder["boundary_search_radius_s"])
            threshold = float(decoder["boundary_threshold"])
            proposed_start = _peak_in_time_radius(start_prob, time_values, float(edges[first]), radius, threshold)
            proposed_end = _peak_in_time_radius(end_prob, time_values, float(edges[stop]), radius, threshold)
            if proposed_start is not None and proposed_end is not None and proposed_start <= proposed_end:
                start_index, end_index = proposed_start, proposed_end

        start_time = float(time_values[start_index])
        end_time = float(edges[end_index + 1])
        if stream_duration_s is not None:
            start_time = max(0.0, min(start_time, float(stream_duration_s)))
            end_time = max(0.0, min(end_time, float(stream_duration_s)))
        duration = end_time - start_time
        if duration < float(decoder["min_duration_s"]):
            continue
        score = float(np.mean(fall[start_index : end_index + 1]))
        result.append({"t_start": start_time, "t_end": end_time, "score": score})
    return result
