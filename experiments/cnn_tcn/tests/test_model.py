from __future__ import annotations

import torch

from cnn_tcn.model import CNNTemporalModel


def _config(variant: str) -> dict:
    return {
        "variant": variant,
        "model": {
            "range_channels": [4, 8],
            "range_kernels": [5, 3],
            "range_strides": [2, 1],
            "features_per_radar": 8,
            "tcn_hidden": 16,
            "tcn_kernel": 3,
            "tcn_dilations": [1, 2],
            "dropout": 0.0,
        },
    }


def test_v1_emits_one_fall_logit_per_frame() -> None:
    model = CNNTemporalModel(_config("v1"))
    output = model(torch.zeros(2, 3, 12, 368))
    assert set(output) == {"fall"}
    assert output["fall"].shape == (2, 12)


def test_v2_emits_fall_and_two_boundary_logits_per_frame() -> None:
    model = CNNTemporalModel(_config("v2"))
    output = model(torch.zeros(2, 3, 12, 368))
    assert set(output) == {"fall", "start", "end"}
    assert all(value.shape == (2, 12) for value in output.values())
