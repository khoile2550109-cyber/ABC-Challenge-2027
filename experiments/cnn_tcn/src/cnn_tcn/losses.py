from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    weight = torch.tensor(float(positive_weight), dtype=logits.dtype, device=logits.device)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        reduction="none",
        pos_weight=weight,
    )
    valid = mask.to(dtype=losses.dtype)
    return (losses * valid).sum() / valid.sum().clamp_min(1.0)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = config["loss"]
    mask = batch["loss_mask"].to(device=outputs["fall"].device, dtype=torch.bool)
    fall_loss = _masked_bce(
        outputs["fall"],
        batch["fall_target"].to(outputs["fall"].device),
        mask,
        float(loss_config["fall_positive_weight"]),
    )
    total = float(loss_config["fall_weight"]) * fall_loss
    parts = {"fall": float(fall_loss.detach().item())}

    if config["variant"] == "v2":
        start_loss = _masked_bce(
            outputs["start"],
            batch["start_target"].to(outputs["start"].device),
            mask,
            float(loss_config["boundary_positive_weight"]),
        )
        end_loss = _masked_bce(
            outputs["end"],
            batch["end_target"].to(outputs["end"].device),
            mask,
            float(loss_config["boundary_positive_weight"]),
        )
        total = total + float(loss_config["start_weight"]) * start_loss
        total = total + float(loss_config["end_weight"]) * end_loss
        parts["start"] = float(start_loss.detach().item())
        parts["end"] = float(end_loss.detach().item())

    parts["total"] = float(total.detach().item())
    return total, parts
