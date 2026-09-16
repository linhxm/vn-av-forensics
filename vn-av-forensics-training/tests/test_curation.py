import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from vn_av_training.features.fate import temporal_descriptor


def test_descriptor_distinguishes_temporal_order():
    sequence = torch.eye(8)
    a = temporal_descriptor(sequence, 4)
    b = temporal_descriptor(sequence.flip(0), 4)
    assert not np.allclose(a, b)
    np.testing.assert_allclose(
        temporal_descriptor(sequence, 1), temporal_descriptor(sequence.flip(0), 1)
    )
    with pytest.raises(ValueError, match="fewer"):
        temporal_descriptor(sequence, 9)


def test_builtin_demo_without_frontend_build(tmp_path):
    from vn_av_training.serving.api import create_app

    cfg = {
        "pipeline": "relations",
        "jobs_dir": str(tmp_path / "jobs"),
        "frontend_dist": str(tmp_path / "missing"),
    }
    with TestClient(create_app(cfg)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "/api/jobs" in page.text
        assert client.get("/api/health").json()["ready"] is False


def test_pretrained_name_mapping_rejects_missing_weights():
    from vn_av_training.features.fate import base_weight_name

    keys = {"audio_model.audio_encoder.layer.weight", "video_model.video_head.proj.weight"}
    assert (
        base_weight_name("audio_encoder.layer.weight", keys)
        == "audio_model.audio_encoder.layer.weight"
    )
    assert base_weight_name("video_head.proj.weight", keys) == "video_model.video_head.proj.weight"
    with pytest.raises(ValueError, match="missing required"):
        base_weight_name("audio_encoder.unknown.weight", keys)


def test_frozen_vision_reuse_keeps_order_and_resets_for_new_source():
    from types import SimpleNamespace

    from vn_av_training.features.fate import CachedFrameVision

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.count = 0

        def forward(self, pixels):
            self.count += len(pixels)
            return SimpleNamespace(logits=pixels * 3)

    vision = Vision()
    cache = CachedFrameVision(vision)
    pixels = torch.arange(6).float().reshape(3, 2)
    cache.select("one", [0, 1, 2])
    torch.testing.assert_close(cache(pixels).logits, pixels * 3)
    cache.select("one", [2, 0, 2])
    torch.testing.assert_close(cache(pixels[[2, 0, 2]]).logits, pixels[[2, 0, 2]] * 3)
    assert vision.count == 3
    cache.select("two", [0, 1, 2])
    cache(pixels)
    assert vision.count == 6
