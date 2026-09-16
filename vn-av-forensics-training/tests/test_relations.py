"""Synthetic features verify semantics and gradients, never real model accuracy."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch
from torch import nn

from vn_av_training.features.fate import FATEBackbone, asset_signature
from vn_av_training.models.detector import correspondence
from vn_av_training.models.relations import RELATIONS, RelationHeads, relation_loss
from vn_av_training.serving.relations import RelationPipeline, aggregate_relations


def batch(n=12, d=8):
    return {
        "audio": torch.randn(1, n, d),
        "visual": torch.randn(1, n, d),
        "audio_valid": torch.ones(1, n, dtype=torch.bool),
        "visual_valid": torch.ones(1, n, dtype=torch.bool),
        "padding_valid": torch.ones(1, n, dtype=torch.bool),
    }


def test_lag_sign_and_no_wrap():
    a = torch.eye(10)[None]
    v = torch.zeros_like(a)
    v[:, 2:] = a[:, :-2]
    mask = torch.ones(1, 10, dtype=torch.bool)
    scores, valid = correspondence(a, v, mask, mask, radius=3)
    assert (scores[0, :8].argmax(-1) - 3 == 2).all()
    assert not valid[0, -2:, 5].any()
    assert not valid[0, :3, 0].any()


def test_heads_train_with_partial_labels_and_invalid_cells():
    torch.manual_seed(4)
    model = RelationHeads(8, 8, projection=8, hidden=8, radius=2, context=3)
    data = batch()
    data["visual_valid"][:, -2:] = False
    data["visual"][:, -2:] = float("nan")
    targets = {name: torch.randint(0, 2, (1, 12)) for name in RELATIONS}
    targets["phoneme_viseme"][:] = -1
    targets["lag_class"] = torch.full((1, 12), 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.heads["sequence"].weight.detach().clone()
    output = model(data)
    loss, losses = relation_loss(output, targets)
    assert torch.isfinite(loss)
    assert "phoneme_viseme" not in losses
    assert "timing" in losses
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.heads["sequence"].weight)
    assert model.audio[0].weight.grad is not None
    assert not output["valid"][0, -2:].any()


def test_activity_streams_are_independent():
    model = RelationHeads(8, 8, projection=8, hidden=8, radius=1).eval()
    data = batch()
    first = model(data)
    data["audio"] = torch.randn_like(data["audio"]) * 3
    second = model(data)
    torch.testing.assert_close(first["logits"]["visual_speech"], second["logits"]["visual_speech"])


def test_null_match_and_empty_evidence():
    model = RelationHeads(8, 8, projection=8, hidden=8, radius=1).eval()
    with torch.no_grad():
        model.no_match[-1].weight.zero_()
        model.no_match[-1].bias.fill_(100)
    data = batch()
    output = model(data)
    assert not output["lag_valid"].any()
    assert (output["lag_logits"].argmax(-1) == 3).all()
    data["audio_valid"][:] = False
    output = model(data)
    result = aggregate_relations(
        output, np.arange(12) * 0.04, 0.04, {"timing": 0.5, "sequence": 0.5}, ("timing", "sequence")
    )
    assert result["status"] == "not_assessable"
    assert result["clip_inconsistency_score"] is None
    assert result["suspicious_intervals"] == []
    assert result["global_lag_ms"] is None


def evidence():
    n = 8
    mask = torch.ones(1, n, dtype=torch.bool)
    mask[:, 4] = False
    logits = torch.full((1, n, 6), -10.0)
    logits[..., 4] = 10  # radius 2, +2 steps
    return {
        "valid": mask,
        "lag_valid": mask,
        "lag_steps": torch.full((1, n), 2),
        "lag_confidence": torch.full((1, n), 0.99),
        "lag_logits": logits,
        "logits": {
            name: torch.tensor([[-10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -10.0, -10.0]])
            for name in RELATIONS
        },
    }


def test_intervals_break_at_missing_data_and_untrained_heads_are_excluded():
    result = aggregate_relations(
        evidence(), np.arange(8) * 0.04, 0.04, {"sequence": 0.5}, ("sequence",)
    )
    intervals = result["suspicious_intervals"]
    np.testing.assert_allclose(
        [(s["start"], s["end"]) for s in intervals], [(0.04, 0.16), (0.2, 0.24)]
    )
    assert result["window_scores"][4]["inconsistency_score"] is None
    assert result["global_lag_ms"] is None
    assert result["head_availability"]["phoneme_viseme"] == "untrained"
    assert result["identity_status"] == "not_assessed"
    assert result["status"] == "partial"


def test_global_lag_retains_original_timing_and_local_variation():
    output = evidence()
    args = (np.arange(8) * 0.04, 0.04, {"timing": 0.5}, ("timing",))
    assert aggregate_relations(output, *args)["global_lag_ms"] == pytest.approx(80)
    output["lag_steps"][0, :4] = -2
    assert aggregate_relations(output, *args)["global_lag_ms"] is None


def test_unknown_labels_are_not_negatives_and_bad_timestamps_fail():
    output = RelationHeads(8, 8, projection=8, hidden=8, radius=1)(batch())
    with pytest.raises(ValueError, match="No observed"):
        relation_loss(output, {"sequence": torch.full((1, 12), -1)})
    with pytest.raises(ValueError, match="uniform"):
        aggregate_relations(output, np.arange(12) ** 2, 0.04, {}, ())
    with pytest.raises(ValueError, match="threshold"):
        aggregate_relations(output, np.arange(12) * 0.04, 0.04, {}, ("sequence",))


class FakeUpstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, audio, visual):
        return SimpleNamespace(
            audio_frame_embeds=audio * self.weight, video_frame_embeds=visual * self.weight
        )


def test_three_parts_compose_and_keep_backbone_frozen():
    backbone = FATEBackbone(FakeUpstream(), None, "test-fixture-not-pretrained")
    heads = RelationHeads(8, 8, projection=8, hidden=8, radius=1)
    backbone.train()
    assert not backbone.model.training
    assert not any(p.requires_grad for p in backbone.parameters())
    pipeline = RelationPipeline(backbone, heads)
    data = batch()
    metadata = {key: data[key] for key in ("audio_valid", "visual_valid", "padding_valid")}
    metadata["times_s"] = np.arange(12) * 0.04
    inputs = {key: data[key] for key in ("audio", "visual")}
    with pytest.raises(ValueError, match="Train relation"):
        pipeline.analyze(inputs, metadata, 0.04)
    # Availability here is solely a fixture to exercise integration, not a trained model claim.
    pipeline.trained_heads = ("sequence",)
    pipeline.thresholds = {"sequence": 0.5}
    result = pipeline.analyze(inputs, metadata, 0.04)
    assert len(result["window_scores"]) == 12
    assert result["head_availability"]["timing"] == "untrained"


def test_missing_fate_assets_fail_before_import_or_download():
    # Missing assets need a nonexistent path, not a directory created by pytest.
    missing = Path(__file__).parent / ("missing-fate-" + uuid4().hex)
    with pytest.raises(FileNotFoundError, match="FATE source"):
        asset_signature(missing / "repo", missing / "base", missing / "adapter")
