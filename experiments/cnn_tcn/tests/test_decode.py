from __future__ import annotations

import numpy as np

from cnn_tcn.decode import decode_events


def _config(variant: str) -> dict:
    return {
        "variant": variant,
        "sample_rate_hz": 50.0,
        "decoder": {
            "fall_threshold": 0.5,
            "boundary_threshold": 0.5,
            "merge_gap_s": 0.04,
            "min_duration_s": 0.02,
            "boundary_search_radius_s": 0.08,
        },
    }


def test_v1_merges_short_gap_and_omits_alarm_time() -> None:
    times = np.arange(12, dtype=np.float64) / 50.0
    probability = np.zeros(12, dtype=np.float32)
    probability[2:4] = 0.9
    probability[5:7] = 0.8
    events = decode_events(probability, times, _config("v1"))
    assert len(events) == 1
    assert events[0]["t_start"] == times[2]
    assert events[0]["t_end"] == times[7]
    assert "t_alarm" not in events[0]


def test_v2_refines_boundaries_only_when_peaks_are_confident() -> None:
    times = np.arange(20, dtype=np.float64) / 50.0
    fall = np.zeros(20, dtype=np.float32)
    fall[5:10] = 0.9
    start = np.zeros(20, dtype=np.float32)
    end = np.zeros(20, dtype=np.float32)
    start[4] = 0.9
    end[10] = 0.9
    events = decode_events(fall, times, _config("v2"), start, end)
    assert len(events) == 1
    assert events[0]["t_start"] == times[4]
    assert events[0]["t_end"] == times[11]
