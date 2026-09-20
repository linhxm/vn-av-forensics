import json

import pytest

from vn_av_data.data.curation import write_csv
from vn_av_data.data.legacy import import_legacy_review
from vn_av_data.data.source_io import read_rows


def test_import_legacy_review_creates_complete_current_bundle(tmp_path, monkeypatch):
    media = tmp_path / "legacy-media"
    media.mkdir()
    keep_id = "abcdefghijk_s0000010000_e0000014000"
    reject_id = "abcdefghijk_s0000020000_e0000024000"
    (media / f"{keep_id}.mp4").write_bytes(b"kept-video")
    (media / f"{reject_id}.mp4").write_bytes(b"rejected-video")
    manifest = tmp_path / "clips.csv"
    write_csv(
        manifest,
        [
            {
                "clip_id": clip_id,
                "source_video": "abcdefghijk",
                "start_time": start,
                "end_time": end,
                "duration": 4,
                "file_path": f"C:\\old\\media\\{clip_id}.mp4",
                "url": "https://www.youtube.com/watch?v=abcdefghijk",
                "tier": "podcast",
            }
            for clip_id, start, end in (
                (keep_id, 10, 14),
                (reject_id, 20, 24),
            )
        ],
    )
    review = tmp_path / "review.csv"
    write_csv(
        review,
        [
            {
                "clip_id": keep_id,
                "file_path": f"C:\\old\\media\\{keep_id}.mp4",
                "decision": "keep",
                "reason": "",
                "bad_intervals_json": "[]",
                "reviewer_id": "reviewer1",
                "rubric_version": "v1",
                "ts": "2026-01-01T00:00:00Z",
            },
            {
                "clip_id": reject_id,
                "file_path": f"C:\\old\\media\\{reject_id}.mp4",
                "decision": "reject",
                "reason": "bad",
                "bad_intervals_json": "[]",
                "reviewer_id": "reviewer1",
                "rubric_version": "v1",
                "ts": "2026-01-01T00:00:00Z",
            },
        ],
    )
    monkeypatch.setattr("vn_av_data.data.export.probe", lambda _: {"duration_s": 4})

    output = tmp_path / "dataset_v001_legacy"
    result = import_legacy_review(review, manifest, media, output, "dataset_v001_legacy")

    assert result["clips"] == 1
    assert result["excluded_reject"] == 1
    assert result["unknown_speaker_clips"] == 1
    assert (output / "clips" / f"{keep_id}.mp4").read_bytes() == b"kept-video"
    assert not (output / "clips" / f"{reject_id}.mp4").exists()
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["source_id"] == "yt_abcdefghijk"
    assert rows[0]["source_start_s"] == 10
    assert rows[0]["source_end_s"] == 14
    assert rows[0]["speaker_id"] is None
    assert rows[0]["relation_annotations"] == {}
    assert read_rows(output / "sources.csv")[0]["video_id"] == "abcdefghijk"
    assert len(read_rows(output / "review.csv")) == 1
    report = json.loads((output / "import_report.json").read_text())
    assert report["excluded_reject"] == 1
    assert report["speaker_id_available"] is False


def test_import_legacy_review_rejects_disagreeing_source_metadata(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    clip_id = "abcdefghijk_s0000010000_e0000014000"
    (media / f"{clip_id}.mp4").write_bytes(b"video")
    manifest = tmp_path / "clips.csv"
    write_csv(
        manifest,
        [
            {
                "clip_id": clip_id,
                "source_video": "lmnopqrstuv",
                "start_time": 10,
                "end_time": 14,
                "duration": 4,
            }
        ],
    )
    review = tmp_path / "review.csv"
    write_csv(
        review,
        [{"clip_id": clip_id, "file_path": f"{clip_id}.mp4", "decision": "keep"}],
    )
    with pytest.raises(ValueError, match="source_video differs"):
        import_legacy_review(review, manifest, media, tmp_path / "out", "legacy_v001")
