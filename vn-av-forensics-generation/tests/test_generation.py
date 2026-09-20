import shutil
from pathlib import Path

import numpy as np
import pytest

from vn_av_generation.common.runtime import read_json, run, sha, write_json
from vn_av_generation.contract import SCHEMA
from vn_av_generation.data.manifest import read_manifest, write_manifest
from vn_av_generation.data.media import ffmpeg
from vn_av_generation.generate import (
    csv_write,
    edit_media,
    finalize_labels,
    migrate_labels,
    plan_dataset,
    render_dataset,
    split_originals,
    supervision,
)


def test_edits_do_not_wrap_and_mismatch_labels_need_review():
    n = 125
    data = {
        "pcm": np.arange(n * 1920, dtype=np.float32),
        "frames": np.arange(n, dtype=np.uint8)[:, None, None, None]
        * np.ones((n, 2, 2, 3), np.uint8),
    }
    frames, pcm = edit_media(data, {"kind": "global_lag", "lag_s": 0.2})
    np.testing.assert_equal(pcm[:-9600], data["pcm"][9600:])
    assert not pcm[-9600:].any()
    np.testing.assert_equal(frames, data["frames"])
    frames, pcm = edit_media(data, {"kind": "motion_freeze", "span": [1, 4]})
    assert (frames[25:100] == data["frames"][25]).all()
    np.testing.assert_equal(pcm, data["pcm"])
    labels, pending = supervision({"kind": "content_splice", "span": [1, 4]}, 5, 5)
    assert labels["relation_annotations"] == {}
    assert {p["head"] for p in pending} == {"lip_audio_mismatch"}
    with pytest.raises(ValueError, match="Unsupported"):
        supervision({"kind": "source_swap"}, 5, 5)


def test_weighted_split_keeps_large_group_in_train():
    rows = [
        {"clip_id": f"c{i}_{j}", "source_id": f"s{i}", "speaker_id": f"p{i}"}
        for i, n in enumerate([56, 16, 16])
        for j in range(n)
    ]
    split = split_originals(rows, 42)
    assert {r["split"] for r in split if r["source_id"] == "s0"} == {"train"}
    assert len({r["split"] for r in split}) == 3


@pytest.fixture
def clean_bundle(tmp_path):
    root = tmp_path / "clean"
    (root / "clips").mkdir(parents=True)
    rows = []
    for i in range(6):
        path = root / "clips" / f"c{i}.mp4"
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=96x96:r=25:d=5",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={400 + 71 * i}:sample_rate=48000:duration=5",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                path,
            ]
        )
        rows.append(
            {
                "clip_id": f"c{i}",
                "source_id": f"s{i // 2}",
                "speaker_id": f"p{i // 2}",
                "video": f"clips/c{i}.mp4",
                "sha256": sha(path),
                "duration_s": 5,
                "source_start_s": i * 10,
                "source_end_s": i * 10 + 5,
                "review_decision": "keep",
                "sync_status": "reviewed_match",
                "relation_annotations": {},
            }
        )
    write_manifest(root / "manifest.jsonl", rows)
    write_json(
        root / "dataset_info.json",
        {
            "schema_version": SCHEMA,
            "dataset_id": "fixture",
            "clips": len(rows),
            "manifest_sha256": sha(root / "manifest.jsonl"),
        },
    )
    return root


def test_render_resume_review_and_portable_training_handoff(clean_bundle, tmp_path):
    cfg = {
        "seed": 42,
        "step_s": 0.2,
        "radius": 4,
        "window_s": 2.0,
        "shifts_s": [0.2],
        "techniques": [
            "global_lag",
            "local_lag",
            "sequence_swap",
            "motion_freeze",
            "content_splice",
        ],
        "crfs": [18],
        "max_side": 96,
    }
    plan = tmp_path / "plan.json"
    report = plan_dataset(clean_bundle, plan, cfg)
    assert report["variant_counts"]["local_lag"] == 6
    assert "source_swap" not in report["variant_counts"]
    bad_plan = read_json(plan)
    bad_plan["jobs"][0]["original"]["video"] = "../outside.mp4"
    write_json(tmp_path / "bad-plan.json", bad_plan)
    with pytest.raises(ValueError, match="Planned original"):
        render_dataset(clean_bundle, tmp_path / "bad-plan.json", tmp_path / "bad-output")
    out = tmp_path / "generated"
    result = render_dataset(clean_bundle, plan, out)
    assert result["samples"] == 36
    digest = sha(out / "clips" / read_manifest(out / "manifest.jsonl")[0]["video"].split("/")[-1])
    assert render_dataset(clean_bundle, plan, out) == result
    assert (
        sha(out / "clips" / read_manifest(out / "manifest.jsonl")[0]["video"].split("/")[-1])
        == digest
    )
    import csv

    with (out / "review.csv").open(encoding="utf-8-sig", newline="") as f:
        reviews = list(csv.DictReader(f))
    for row in reviews:
        row["decision"] = "positive"
    csv_write(out / "review.csv", reviews)
    finalize_labels(out, out / "review.csv", out / "manifest-reviewed.jsonl")
    moved = tmp_path / "other_machine" / "dataset"
    shutil.copytree(out, moved)
    rows = read_manifest(moved / "manifest-reviewed.jsonl")
    assert any(
        r["relation_annotations"].get("lip_audio_mismatch", {}).get("positive") for r in rows
    )
    # Conversion preserves rendered files and original labels, excludes source-only samples.
    import copy
    legacy_rows = copy.deepcopy(rows)
    for row in legacy_rows:
        row["relation_annotations"] = {"sequence": row["relation_annotations"].get("lip_audio_mismatch", {"known": [], "positive": []})}
    excluded = copy.deepcopy(legacy_rows[0])
    excluded["sample_id"] += "_source"
    excluded["generation"]["edit"]["kind"] = "source_swap"
    legacy_rows.append(excluded)
    legacy_path = moved / "legacy-reviewed.jsonl"
    write_manifest(legacy_path, legacy_rows)
    write_json(legacy_path.with_suffix(".info.json"), {"manifest_sha256": sha(legacy_path), "samples": len(legacy_rows)})
    before = sha(legacy_path)
    migrated = migrate_labels(moved, legacy_path.name, moved / "manifest-two-heads.jsonl")
    assert migrated["excluded_source_swap"] == 1
    assert migrated["samples"] == 36
    assert sha(legacy_path) == before
    training = Path(__file__).resolve().parents[2] / "vn-av-forensics-training/src"
    if not training.is_dir():
        return
    if training.is_dir():
        import sys

        sys.path.insert(0, str(training))
        from vn_av_training.data.generated import import_generated

        result = import_generated(moved, tmp_path / "train.jsonl", "manifest-reviewed.jsonl")
        assert result["samples"] == 36
        assert import_generated(moved, tmp_path / "migrated-train.jsonl", "manifest-two-heads.jsonl")["samples"] == 36
        from vn_av_training.data.labels import labels_for

        rendered = next(r for r in rows if r["generation"]["edit"]["kind"] == "global_lag")
        labels = labels_for(rendered, np.arange(25) * 0.2, 2, 0.2, 4)
        assert (labels["lag_class"] == 5).any()
        assert (labels["lip_audio_mismatch"] == 0).any()
        assert not (labels["lip_audio_mismatch"] == 1).any()
    (moved / rows[0]["video"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        import_generated(moved, tmp_path / "bad.jsonl", "manifest-reviewed.jsonl")
