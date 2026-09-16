"""Pipeline integration with synthetic features; real weights are never faked as results."""

from pathlib import Path

import numpy as np
import pytest
import torch

from vn_av_training.common.runtime import load_config, save_npz, sha, write_json
from vn_av_training.data.manifest import write_manifest
from vn_av_training.data.relations import apply_variant, labels_for
from vn_av_training.features.fate import window_plan
from vn_av_training.training.relations import fit_relations, load_relations


def test_shift_pcm_and_validity_without_wrap():
    n, sample_rate = 20, 48000
    source = {
        "pcm": np.arange(n * 1920, dtype=np.float32),
        "audio_valid": np.ones(n, bool),
        "frames": np.zeros((n, 2, 2, 3)),
        "visual_valid": np.ones(n, bool),
    }
    result = apply_variant(source, {"kind": "global_lag", "lag_s": 0.08}, sample_rate=sample_rate)
    np.testing.assert_array_equal(result["pcm"][:-3840], source["pcm"][3840:])
    assert not result["audio_valid"][-2:].any()
    assert (result["pcm"][-3840:] == 0).all()
    assert source["pcm"][-1] != 0


def test_local_lag_transition_labels_and_independent_annotations():
    row = {"duration_s": 8.0, "variant": {"kind": "local_lag", "lag_s": 0.4, "span": [2, 6]}}
    times = np.arange(40) * 0.2
    labels = labels_for(row, times, 1.0, 0.2, 4)
    assert labels["lag_class"][16] == 6
    assert labels["lag_class"][5] == 4
    assert labels["lag_class"][9] == -1
    assert (labels["motion_speech"] == -1).all()
    row["relation_annotations"] = {"phoneme_viseme": {"known": [[0, 8]], "positive": [[3, 3.4]]}}
    labels = labels_for(row, times, 1.0, 0.2, 4)
    assert labels["phoneme_viseme"][15] == 1
    assert labels["phoneme_viseme"][14] == 0


def test_window_grid_has_physical_support_and_padding():
    plan = list(window_plan(75, 2.0, 0.2))
    assert len(plan) == 15
    assert plan[0][1][0] < 0
    assert not plan[0][2].all()
    assert plan[7][2].all()
    assert plan[-1][0] == pytest.approx(2.8)
    with pytest.raises(ValueError, match="25 Hz"):
        list(window_plan(75, 2.0, 0.13))


@pytest.fixture
def relation_project(tmp_path):
    cache = tmp_path / "cache"
    rows = []
    rng = np.random.default_rng(5)
    for split in ("train", "validation", "test"):
        for kind in ("clean", "global_lag", "sequence_swap"):
            sid = f"{split}_{kind}"
            variant = {"kind": kind}
            if kind == "global_lag":
                variant["lag_s"] = 0.2
            row = {
                "sample_id": sid,
                "source_id": split,
                "speaker_id": split,
                "dataset": "synthetic-fixture",
                "split": split,
                "video": str(tmp_path / "unused.mp4"),
                "duration_s": 8.0,
                "variant": variant,
            }
            path = cache / (sid + ".npz")
            values = rng.normal(size=(40, 8)).astype(np.float32)
            save_npz(
                path,
                audio=values,
                visual=values.copy()
                if kind == "clean"
                else rng.normal(size=values.shape).astype(np.float32),
                times_s=np.arange(40) * 0.2,
                audio_valid=np.ones(40, bool),
                visual_valid=np.ones(40, bool),
            )
            write_json(
                path.with_suffix(".json"),
                {
                    "format": "fate-window-v1",
                    "feature_signature": "synthetic-only",
                    "feature_sha256": sha(path),
                    "source_fingerprint": "synthetic-only",
                    "step_s": 0.2,
                    "window_s": 0.4,
                    "duration_s": 8.0,
                    "origin_s": 0,
                },
            )
            rows.append(row)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    cfg = {
        "manifest": str(manifest),
        "cache": str(cache),
        "output": str(tmp_path / "run"),
        "checkpoint": str(tmp_path / "run/best.pt"),
        "device": "cpu",
        "seed": 42,
        "encoder": {"window_s": 0.4, "step_s": 0.2},
        "model": {"projection": 8, "hidden": 8, "radius": 1, "context": 3},
        "training": {"epochs": 1, "accumulation": 2, "torch_threads": 2, "patience": 0},
    }
    return cfg, rows


def test_training_resume_evaluation_and_raw_analyzer_contract(relation_project, tmp_path):
    from vn_av_training.common.runtime import read_json
    from vn_av_training.evaluation.relations import evaluate_relations
    from vn_av_training.serving.relations import RelationAnalyzer

    cfg, _rows = relation_project
    result = fit_relations(cfg)
    assert Path(result["checkpoint"]).is_file()
    _model, state = load_relations(cfg["checkpoint"])
    assert "sequence" in state["trained_heads"]
    assert "phoneme_viseme" not in state["trained_heads"]
    cfg["training"]["epochs"] = 2
    fit_relations(cfg, resume=True)
    assert len(read_json(Path(cfg["output"]) / "history.json")) == 2
    metrics = evaluate_relations(cfg)
    assert metrics["samples"] == 3
    assert "sequence" in metrics["relations"]

    class FixtureEncoder:
        signature = "synthetic-only"

        def extract(self, row, output):
            source = Path(cfg["cache"]) / "test_clean.npz"
            Path(output).write_bytes(source.read_bytes())
            meta = read_json(source.with_suffix(".json"))
            write_json(Path(output).with_suffix(".json"), meta)
            return meta

    report = RelationAnalyzer(cfg, encoder=FixtureEncoder()).analyze(
        "fixture.mp4", tmp_path / "analysis"
    )
    assert report["schema_version"] == "av-relations-v1"
    assert report["identity_status"] == "not_assessed"
    assert (tmp_path / "analysis/timeline.csv").is_file()
    assert report["head_availability"]["phoneme_viseme"] == "untrained"
    # Resume must reject a changed architecture rather than silently reuse weights.
    cfg["model"]["hidden"] = 9
    with pytest.raises(ValueError, match="differ"):
        fit_relations(cfg, resume=True)


def test_training_does_not_read_test_cache(relation_project):
    cfg, rows = relation_project
    for row in rows:
        if row["split"] == "test":
            (Path(cfg["cache"]) / (row["sample_id"] + ".npz")).unlink()
    fit_relations(cfg)


def test_cli_config_and_doctor(capsys, tmp_path):
    import json

    from vn_av_training.cli import main

    cfg = load_config(Path(__file__).parents[1] / "configs/relations.yaml")
    assert cfg["pipeline"] == "relations"
    cfg["encoder"]["base"] = str(tmp_path / "missing-base")
    local = tmp_path / "missing-assets.yaml"
    local.write_text(json.dumps(cfg), encoding="utf-8")
    assert (
        main(
            [
                "relations-doctor",
                "--config",
                str(local),
            ]
        )
        == 1
    )
    assert "preprocessing_ready" in capsys.readouterr().out


def test_grid_48k_decode_preserves_stream_offsets(tmp_path):
    from vn_av_training.common.runtime import run
    from vn_av_training.data.media import decode, ffmpeg

    path = tmp_path / "audio-offset.mkv"
    run(
        [
            ffmpeg(),
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=2",
            "-itsoffset",
            "0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.6",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            path,
        ]
    )
    decoded = decode(path, sample_rate=48000)
    assert not decoded["audio_valid"][:8].any()
    assert np.abs(decoded["pcm"][:16000]).max() == 0
    assert len(decoded["pcm"]) == len(decoded["frames"]) * 1920


def test_window_extraction_cache_and_source_invalidation(tmp_path):
    from types import SimpleNamespace

    from vn_av_training.features.fate import FATEBackbone, FATEVideoEncoder

    class UpstreamFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))

        def forward(self, audio, visual):
            return SimpleNamespace(
                audio_frame_embeds=audio * self.anchor, video_frame_embeds=visual * self.anchor
            )

    calls = []

    def processor(**kwargs):
        assert kwargs["sampling_rate"] == 48000
        assert kwargs["videos"][0].shape[0] == 50
        calls.append(1)
        return {"audio": torch.ones(1, 4, 8), "visual": torch.ones(1, 4, 8)}

    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"synthetic-test-source")
    decoded = {
        "frames": np.zeros((75, 4, 4, 3), np.uint8),
        "pcm": np.ones(75 * 1920, np.float32),
        "audio_valid": np.ones(75, bool),
        "visual_valid": np.ones(75, bool),
        "origin_s": 0,
    }
    backbone = FATEBackbone(UpstreamFixture(), processor, "synthetic-test-only")
    encoder = FATEVideoEncoder(
        {"window_s": 2.0, "step_s": 0.2},
        backbone=backbone,
        cropper=lambda frames: (frames, np.ones(len(frames), bool)),
    )
    encoder.decode = lambda _path: decoded
    row = {"video": str(source), "variant": {"kind": "clean"}}
    output = tmp_path / "features.npz"
    meta = encoder.extract(row, output)
    assert len(calls) == 15
    assert meta["duration_s"] == 3
    with np.load(output, allow_pickle=False) as features:
        assert features["audio"].shape == (15, 8)
        assert not features["visual_valid"][0]
        assert features["visual_valid"][7]
    encoder.extract(row, output)
    assert len(calls) == 15
    source.write_bytes(b"changed-test-source")
    encoder.extract(row, output)
    assert len(calls) == 30


@pytest.mark.assets
def test_optional_real_fate_video(tmp_path):
    import os

    from vn_av_training.features.fate import FATEVideoEncoder

    config = os.environ.get("VN_AV_FATE_CONFIG")
    video = os.environ.get("VN_AV_FATE_VIDEO")
    if not config or not video:
        pytest.skip("Set VN_AV_FATE_CONFIG and VN_AV_FATE_VIDEO to check real FATE assets")
    encoder = FATEVideoEncoder(load_config(config)["encoder"])
    output = tmp_path / "features.npz"
    meta = encoder.extract({"video": video}, output)
    assert meta["format"] == "fate-window-v1"
    with np.load(output, allow_pickle=False) as features:
        assert features["audio"].shape == features["visual"].shape
        assert features["visual_valid"].any()
