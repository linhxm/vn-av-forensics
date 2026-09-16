"""Shared trainable relation heads. Outputs are not pretrained predictions.

Lag convention: A(t) corresponds to V(t+k). Positive k means the visible
articulation occurs later (audio leads). The final lag class is no-match.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from vn_av_training.models.detector import TemporalBlock, correspondence

RELATIONS = ("phoneme_viseme", "sequence", "motion_speech", "source")
ACTIVITIES = ("audio_speech", "visual_speech")


class RelationHeads(nn.Module):
    def __init__(
        self,
        audio_dim,
        visual_dim,
        projection=128,
        hidden=128,
        radius=20,
        context=5,
        dropout=0.1,
        lag_confidence=0.5,
    ):
        super().__init__()
        if min(audio_dim, visual_dim, projection, hidden) <= 0:
            raise ValueError("Feature dimensions must be positive")
        if radius < 0 or context < 1 or context % 2 != 1:
            raise ValueError("radius >= 0 and an odd positive context are required")
        if not 0 < lag_confidence <= 1:
            raise ValueError("lag_confidence must be in (0, 1]")
        self.config = {
            "audio_dim": audio_dim,
            "visual_dim": visual_dim,
            "projection": projection,
            "hidden": hidden,
            "radius": radius,
            "context": context,
            "dropout": dropout,
            "lag_confidence": lag_confidence,
        }
        self.audio = nn.Sequential(nn.Linear(audio_dim, projection), nn.LayerNorm(projection))
        self.visual = nn.Sequential(nn.Linear(visual_dim, projection), nn.LayerNorm(projection))
        self.no_match = nn.Sequential(
            nn.Linear(2 * projection, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        # Raw pair + lag-compensated pair + confidence + whether compensation was used.
        self.fusion = nn.Linear(4 * projection + 2, hidden)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, d, dropout) for d in (1, 2, 4))
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in RELATIONS})
        self.audio_speech = nn.Linear(projection, 1)
        self.visual_speech = nn.Linear(projection, 1)

    def forward(self, batch, use_lag_alignment=True):
        audio, visual = batch["audio"].float(), batch["visual"].float()
        if audio.ndim != 3 or visual.ndim != 3 or audio.shape[:2] != visual.shape[:2]:
            raise ValueError("Expected [batch, time, feature] on the same time grid")
        if audio.shape[1] == 0:
            raise ValueError("Empty feature sequence")
        padding = batch["padding_valid"].bool()
        am = batch["audio_valid"].bool() & padding
        vm = batch["visual_valid"].bool() & padding
        if any(mask.shape != audio.shape[:2] for mask in (padding, am, vm)):
            raise ValueError("Invalid feature mask shape")
        if not torch.isfinite(audio[am]).all() or not torch.isfinite(visual[vm]).all():
            raise ValueError("Nonfinite valid features")
        a = self.audio(audio.masked_fill(~am[..., None], 0)).masked_fill(~am[..., None], 0)
        v = self.visual(visual.masked_fill(~vm[..., None], 0)).masked_fill(~vm[..., None], 0)
        valid = am & vm
        radius, context = self.config["radius"], self.config["context"]
        scores, pairs = correspondence(a, v, am, vm, radius)
        counts = F.avg_pool1d(pairs.float().transpose(1, 2), context, 1, context // 2)
        sums = F.avg_pool1d(scores.transpose(1, 2), context, 1, context // 2)
        scores = (sums / counts.clamp_min(1e-6)).transpose(1, 2)
        supported = pairs & (counts.transpose(1, 2) > 0) & valid[..., None]
        logits = (scores * self.log_scale.exp().clamp(max=100)).masked_fill(~supported, -1e4)
        null = self.no_match(torch.cat([a, v], -1))
        lag_logits = torch.cat([logits, null], -1)
        probs = lag_logits.softmax(-1)
        confidence, best = probs[..., :-1].max(-1)
        lag_valid = valid & (confidence >= self.config["lag_confidence"])
        lag_valid &= probs.argmax(-1) != 2 * radius + 1
        if not use_lag_alignment:
            lag_valid = torch.zeros_like(lag_valid)
        steps = best - radius
        index = torch.arange(a.shape[1], device=a.device)[None] + steps
        index = index.clamp(0, a.shape[1] - 1)
        aligned_v = v.gather(1, index[..., None].expand_as(v))
        # Never force a different utterance to match. Keep the raw pair if uncertain.
        aligned_v = torch.where(lag_valid[..., None], aligned_v, v)
        x = F.gelu(
            self.fusion(
                torch.cat(
                    [a, v, a, aligned_v, confidence[..., None], lag_valid[..., None].float()], -1
                )
            )
        ).masked_fill(~valid[..., None], 0)
        for block in self.blocks:
            x = block(x, valid)
        relation_logits = {name: head(x).squeeze(-1) for name, head in self.heads.items()}
        relation_logits["audio_speech"] = self.audio_speech(a).squeeze(-1)
        relation_logits["visual_speech"] = self.visual_speech(v).squeeze(-1)
        return {
            "logits": relation_logits,
            "valid": valid,
            "audio_valid": am,
            "visual_valid": vm,
            "lag_logits": lag_logits,
            "lag_supported": supported,
            "lag_steps": steps,
            "lag_valid": lag_valid,
            "lag_confidence": confidence,
        }


def relation_loss(outputs, targets):
    """Partial supervision: -1 is unknown, never silently a negative.

    lag_class: 0..2*radius for signed offsets, 2*radius+1 for no-match.
    Binary heads: 0/1, with -1 for missing/unobservable annotations.
    A phoneme_viseme head needs its own vetted labels, not generic forgery labels.
    """
    losses = {}
    for name in (*RELATIONS, *ACTIVITIES):
        if name not in targets:
            continue
        y = targets[name]
        if y.shape != outputs["valid"].shape or not torch.isin(y, y.new_tensor([-1, 0, 1])).all():
            raise ValueError(f"Invalid binary labels: {name}")
        mask_name = {"audio_speech": "audio_valid", "visual_speech": "visual_valid"}
        mask = outputs[mask_name.get(name, "valid")] & (y >= 0)
        if mask.any():
            losses[name] = F.binary_cross_entropy_with_logits(
                outputs["logits"][name][mask], y[mask].float()
            )
    if "lag_class" in targets:
        y = targets["lag_class"]
        classes = outputs["lag_logits"].shape[-1]
        if (
            y.shape != outputs["valid"].shape
            or not ((y >= -1) & (y < classes) & (y == y.long())).all()
        ):
            raise ValueError("Invalid lag_class labels")
        mask = outputs["valid"] & (y >= 0)
        supported = torch.cat([outputs["lag_supported"], torch.ones_like(mask[..., None])], -1)
        mask &= supported.gather(-1, y.long().clamp_min(0)[..., None]).squeeze(-1)
        if mask.any():
            losses["timing"] = F.cross_entropy(outputs["lag_logits"][mask], y[mask].long())
    if not losses:
        raise ValueError("No observed relation labels in this batch")
    return torch.stack(list(losses.values())).mean(), losses
