"""Frozen-encoder feature detector: fusion TCN, temporal relation and phoneme ablations."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def correspondence(a, v, amask, vmask, radius):
    """S(t,k)=cos(A(t),V(t+k)); positive k looks later in the video, never wraps."""
    a, v = F.normalize(a, dim=-1), F.normalize(v, dim=-1)
    t = a.shape[1]
    values, masks = [], []
    for k in range(-radius, radius + 1):
        index = torch.arange(t, device=a.device) + k
        inside = (index >= 0) & (index < t)
        index = index.clamp(0, t - 1)
        mask = amask & vmask[:, index] & inside
        values.append((a * v[:, index]).sum(-1).masked_fill(~mask, 0))
        masks.append(mask)
    return torch.stack(values, -1), torch.stack(masks, -1)


class TemporalBlock(nn.Module):
    def __init__(self, hidden, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return (x + self.dropout(F.gelu(self.norm(y)))) * valid[..., None]


class Detector(nn.Module):
    def __init__(
        self,
        audio_dim=768,
        visual_dim=768,
        projection=256,
        hidden=128,
        dilations=(1, 2, 4),
        dropout=0.1,
        modality="av",
        temporal=False,
        radius=5,
        phoneme=False,
    ):
        super().__init__()
        if modality not in ("audio", "visual", "av") or radius < 0:
            raise ValueError("Invalid modality/radius")
        if (temporal or phoneme) and modality != "av":
            raise ValueError("Temporal/phoneme branches require both modalities")
        self.config = dict(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            projection=projection,
            hidden=hidden,
            dilations=list(dilations),
            dropout=dropout,
            modality=modality,
            temporal=temporal,
            radius=radius,
            phoneme=phoneme,
        )
        self.audio = nn.Sequential(nn.Linear(audio_dim, projection), nn.LayerNorm(projection))
        self.visual = nn.Sequential(nn.Linear(visual_dim, projection), nn.LayerNorm(projection))
        width = projection * (2 if modality == "av" else 1)
        width += 2 * (2 * radius + 1) if temporal else 0
        width += 3 if phoneme else 0
        self.input = nn.Linear(width, hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, d, dropout) for d in dilations)
        self.clip = nn.Linear(hidden, 1)
        self.forgery = nn.Linear(hidden, 1) if temporal else None
        self.mismatch = nn.Linear(hidden, 1) if temporal else None

    def forward(self, batch):
        c = self.config
        am = batch["audio_valid"].bool() & batch["padding_valid"]
        vm = batch["visual_valid"].bool() & batch["padding_valid"]
        a = self.audio(batch["audio"].float()) * am[..., None]
        v = self.visual(batch["visual"].float()) * vm[..., None]
        valid = am if c["modality"] == "audio" else vm if c["modality"] == "visual" else am & vm
        features = [a] if c["modality"] == "audio" else [v] if c["modality"] == "visual" else [a, v]
        lag_score = lag_mask = None
        if c["temporal"]:
            lag_score, lag_mask = correspondence(a, v, am, vm, c["radius"])
            features.extend([lag_score, lag_mask.float()])
        if c["phoneme"]:
            pm = batch["phoneme_valid"].bool() & valid
            features.extend([batch["phoneme"].float() * pm[..., None], pm[..., None].float()])
        x = F.gelu(self.input(torch.cat(features, -1))) * valid[..., None]
        for block in self.blocks:
            x = block(x, valid)
        pooled = x.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return {
            "clip": self.clip(pooled).squeeze(-1),
            "valid": valid,
            "forgery": self.forgery(x).squeeze(-1) if self.forgery else None,
            "mismatch": self.mismatch(x).squeeze(-1) if self.mismatch else None,
            "lag_score": lag_score,
            "lag_mask": lag_mask,
        }
