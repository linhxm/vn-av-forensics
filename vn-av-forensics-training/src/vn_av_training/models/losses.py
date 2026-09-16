"""Masked supervised losses for clip, forgery and mismatch heads."""

import torch
from torch.nn import functional as F


def supervised_loss(output, batch, weights=None, pos_weight=1.0):
    weights = weights or {"clip": 1.0, "forgery": 1.0, "mismatch": 1.0}
    total = output["clip"].sum() * 0
    components = {}
    for name in ("clip", "forgery", "mismatch"):
        if output[name] is None or weights.get(name, 0) <= 0:
            continue
        if name == "clip":
            known = batch["clip_known"].bool() & output["valid"].any(1)
            target = batch["clip_label"]
        else:
            known = batch[name + "_known"].bool() & output["valid"]
            target = batch[name]
        if known.any():
            loss = F.binary_cross_entropy_with_logits(
                output[name][known],
                target[known].float(),
                pos_weight=torch.tensor(
                    pos_weight if name == "clip" else 1.0, device=target.device
                ),
            )
            total = total + weights[name] * loss
            components[name] = float(loss.detach())
    return total, components
