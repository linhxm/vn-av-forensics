import csv

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vn_av_data.common.runtime import load_config, run, sha
from vn_av_data.data.acquisition import index_sources, youtube_id
from vn_av_data.data.curation import curate_sources, cut_media, media_origin, plan_clips, write_csv
from vn_av_data.data.media import decode, ffmpeg
from vn_av_data.data.vad import audio_blocks, speech_regions
from vn_av_data.serving.review import create_review_app


@pytest.fixture
def delayed_source(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "delayed.mkv"
    run(
        [
            ffmpeg(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=96x96:r=25:d=5",
            "-itsoffset",
            "0.8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4.2",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            path,
        ]
    )
    return path


def test_cut_preserves_relative_stream_delay_and_source_mapping(delayed_source, tmp_path):
    origin, _ = media_origin(delayed_source)
    output = tmp_path / "cut.mp4"
    cut_media(delayed_source, output, 0.4, 4.4, origin)
    media = decode(output, sample_rate=48000)
    # At source time 0.8 audio begins: after cropping at 0.4 it must still be delayed 0.4s.
    power = np.abs(media["pcm"])
    assert power[: int(0.30 * 48000)].max() < 1e-3
    assert power[int(0.45 * 48000) : int(0.6 * 48000)].max() > 0.05
    assert len(media["frames"]) / 25 == pytest.approx(4, abs=0.08)
    with pytest.raises(FileExistsError):
        cut_media(delayed_source, output, 0.4, 4.4, origin)


def test_vad_pcm_keeps_timestamp_gaps(delayed_source):
    origin, _ = media_origin(delayed_source)
    pcm = np.concatenate(list(audio_blocks(delayed_source, origin)))
    assert np.abs(pcm[: int(0.75 * 16000)]).max() == 0
    assert np.abs(pcm[int(0.85 * 16000) : int(0.95 * 16000)]).max() > 0.05
    spans = speech_regions([0] * 20 + [1] * 40 + [0] * 20, 2.56)
    assert spans[0][0] == pytest.approx(0.49)


def test_scene_boundaries_and_duration_limits():
    cfg = {"min_seconds": 3, "max_seconds": 8, "overlap_seconds": 1}
    clips = plan_clips([[0, 25]], [10], 25, cfg)
    assert all(3 <= b - a <= 8 for a, b in clips)
    assert not any(a < 10 < b for a, b in clips)
    assert clips[-1][1] == 25


def test_cut_resume_does_not_overwrite_review(delayed_source, tmp_path):
    manifest = tmp_path / "sources.jsonl"
    index_sources(delayed_source.parent, manifest)
    cfg = load_config("configs/data.yaml")
    # Inject only model predictions; exercise real indexing, FFmpeg, manifests and resume.
    model = tmp_path / "model"
    model.write_bytes(b"test-only")
    cfg["curation"].update(face_model=str(model), vad_model=str(model))
    output = tmp_path / "cut"
    calls = []

    def scan(*_):
        calls.append(1)
        return ([{"time_s": t, "valid": True} for t in np.arange(0, 5, 0.25)], [])

    result = curate_sources(manifest, output, cfg, lambda *_: [[0, 5]], scan)
    assert result["candidates"] == 1
    review = output / "review.csv"
    with review.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["decision"] == "uncertain"
    assert rows[0]["source_sha256"] == sha(delayed_source)
    rows[0]["decision"] = "keep"
    write_csv(review, rows)
    digest = sha(review)
    curate_sources(manifest, output, cfg, lambda *_: [[0, 5]], scan)
    assert sha(review) == digest
    assert len(calls) == 1
    cfg["curation"]["min_face_ratio"] = 0.8
    with pytest.raises(ValueError, match="changed"):
        curate_sources(manifest, output, cfg)


def test_review_saves_decision_and_rejects_traversal(delayed_source, tmp_path):
    path = tmp_path / "review.csv"
    write_csv(path, [{"clip_id": "one", "file_path": "raw/delayed.mkv", "decision": "uncertain"}])
    with TestClient(create_review_app(path, tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/media/0").status_code == 200
        assert client.post("/api/clips/0", json={"decision": "keep"}).status_code == 200
        assert client.post("/api/clips/0", json={"decision": "invalid"}).status_code == 422
        assert client.get("/api/media/1").status_code == 404
    assert "keep" in path.read_text(encoding="utf-8-sig")
    with pytest.raises(ValueError, match="outside root"):
        create_review_app(path, tmp_path / "different")


def test_youtube_url_validation():
    assert youtube_id("https://youtu.be/abcdefghijk?t=2") == "abcdefghijk"
    assert youtube_id("https://www.youtube.com/watch?v=abcdefghijk&list=xyz") == "abcdefghijk"
    for url in (
        "file:///tmp/video",
        "https://youtube.com/playlist?list=xyz",
        "https://evil.test/abcdefghijk",
    ):
        with pytest.raises(ValueError):
            youtube_id(url)


def test_model_download_retries_completed_files_without_network(tmp_path, monkeypatch):
    import io

    from vn_av_data.assets import fetch_file

    requests = []

    def response(*args, **kwargs):
        requests.append(args[0])
        return io.BytesIO(b"fixture-model" * 200)

    monkeypatch.setattr("urllib.request.urlopen", response)
    target = tmp_path / "model.onnx"
    first = fetch_file("https://example.test/model", target)
    assert fetch_file("https://example.test/model", target) == first
    assert len(requests) == 1
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Unverified"):
        fetch_file("https://example.test/model", target)
