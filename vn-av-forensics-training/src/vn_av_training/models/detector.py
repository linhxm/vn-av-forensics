"""Masked temporal building blocks for the two-head model."""

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
