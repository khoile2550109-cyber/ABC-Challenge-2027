from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class CNNTemporalModel(nn.Module):
    """Shared range-bin CNN followed by a temporal convolution stack."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        channels = [int(x) for x in model["range_channels"]]
        kernels = [int(x) for x in model["range_kernels"]]
        strides = [int(x) for x in model["range_strides"]]
        range_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel, stride in zip(channels, kernels, strides):
            range_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        range_layers.append(nn.AdaptiveAvgPool1d(1))
        self.range_encoder = nn.Sequential(*range_layers)
        self.features_per_radar = int(model["features_per_radar"])

        input_channels = 3 * self.features_per_radar
        hidden = int(model["tcn_hidden"])
        self.temporal_projection = nn.Conv1d(input_channels, hidden, kernel_size=1)
        self.temporal_blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(hidden, int(model["tcn_kernel"]), int(dilation), float(model["dropout"]))
                for dilation in model["tcn_dilations"]
            ]
        )
        self.fall_head = nn.Conv1d(hidden, 1, kernel_size=1)
        self.variant = config["variant"]
        if self.variant == "v2":
            self.start_head = nn.Conv1d(hidden, 1, kernel_size=1)
            self.end_head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        if value.ndim != 4 or value.shape[1] != 3 or value.shape[-1] != 368:
            raise ValueError(f"expected input (B,3,L,368), got {tuple(value.shape)}")
        batch, radars, frames, bins = value.shape
        per_frame = value.reshape(batch * radars * frames, 1, bins)
        encoded = self.range_encoder(per_frame).squeeze(-1)
        encoded = encoded.reshape(batch, radars, frames, self.features_per_radar)
        encoded = encoded.permute(0, 2, 1, 3).reshape(batch, frames, radars * self.features_per_radar)
        temporal = self.temporal_projection(encoded.transpose(1, 2))
        temporal = self.temporal_blocks(temporal)
        result = {"fall": self.fall_head(temporal).squeeze(1)}
        if self.variant == "v2":
            result["start"] = self.start_head(temporal).squeeze(1)
            result["end"] = self.end_head(temporal).squeeze(1)
        return result
