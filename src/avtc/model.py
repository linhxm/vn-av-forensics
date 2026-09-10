"""Local temporal detector with genuinely fine-tunable pretrained SyncNet projections."""

import torch
from torch import nn

from .syncnet import METHOD, WEIGHT_HASHES


def projection():
    return nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1024))


class LocalModel(nn.Module):
    def __init__(self, fine_tune=True, radius=5, hidden=64):
        super().__init__()
        self.config = {"fine_tune": fine_tune, "radius": radius, "hidden": hidden}
        self.radius = radius
        self.audio_projection = projection()
        self.visual_projection = projection()
        self.input = nn.Conv1d(2 * radius + 3, hidden, 1)
        self.temporal = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d) for d in (1, 2, 4)]
        )
        self.output = nn.Conv1d(hidden, 1, 1)
        self.dropout = nn.Dropout(0.1)
        for branch in (self.audio_projection, self.visual_projection):
            for p in branch.parameters():
                p.requires_grad_(fine_tune)
            for layer in branch:
                if isinstance(layer, nn.BatchNorm1d):
                    for p in layer.parameters():
                        p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained BN running statistics fixed with short clips / small batches.
        self.audio_projection.eval()
        self.visual_projection.eval()
        return self

    def initialize(self, path):
        initial = torch.load(path, map_location="cpu", weights_only=True)
        if initial["backbone_sha256"] != WEIGHT_HASHES["data/syncnet_v2.model"]:
            raise ValueError("Pretrained initialization mismatch")
        self.audio_projection.load_state_dict(initial["audio"])
        self.visual_projection.load_state_dict(initial["visual"])

    def forward(self, v, a):
        # One variable-length clip: T x 512 cached CNN features.
        if len(v) != len(a) or len(v) <= 2 * self.radius:
            raise ValueError("Invalid feature length")
        v = self.visual_projection(v)
        a = self.audio_projection(a)
        r = self.radius
        index = torch.arange(r, len(v) - r, device=v.device)
        distance = torch.stack(
            [torch.linalg.vector_norm(v[index] - a[index + k], dim=1) for k in range(-r, r + 1)],
            dim=1,
        )
        centered = distance - distance.mean(dim=1, keepdim=True)
        x = (
            torch.cat(
                [centered, distance.mean(dim=1, keepdim=True), distance.amin(dim=1, keepdim=True)],
                dim=1,
            )
            / 10.0
        )
        x = torch.relu(self.input(x.T.unsqueeze(0)))
        for layer in self.temporal:
            x = x + self.dropout(torch.relu(layer(x)))
        return self.output(x).flatten()


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("method") != METHOD:
        raise ValueError("Use a local.pt checkpoint from the current training notebook")
    model = LocalModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint
