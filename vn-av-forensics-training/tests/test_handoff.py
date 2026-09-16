"""Real media transfer between independent projects; no model quality claims."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from vn_av_training.common.runtime import run, sha, write_json
from vn_av_training.contract import SCHEMA, validate_bundle
from vn_av_training.data.bundle import import_bundle
from vn_av_training.data.manifest import read_manifest, validate, write_manifest
from vn_av_training.data.media import ffmpeg
from vn_av_training.data.relations import make_controls


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "input_dataset"
    (root / "clips").mkdir(parents=True)
    rows = []
    for i in range(6):
        path = root / "clips" / f"clip{i}.mp4"
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=96x96:r=25:d=4",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={440 + i * 100}:sample_rate=48000:duration=4",
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
                "clip_id": f"clip{i}",
                "video": f"clips/clip{i}.mp4",
                "sha256": sha(path),
                "source_id": f"source{i // 2}",
                "speaker_id": f"speaker{i // 2}",
                "duration_s": 4.0,
                "source_start_s": i * 10,
                "source_end_s": i * 10 + 4,
                "review_decision": "keep",
                "sync_status": "reviewed_match",
                "relation_annotations": {
                    "motion_speech": {"known": [[0, 4]], "positive": [[1, 2]]}
                },
            }
        )
    write_manifest(root / "manifest.jsonl", rows)
    write_json(
        root / "dataset_info.json",
        {
            "schema_version": SCHEMA,
            "dataset_id": "fixture_v001",
            "clips": len(rows),
            "manifest_sha256": sha(root / "manifest.jsonl"),
        },
    )
    return root


def test_relocated_bundle_split_before_controls_and_resume_protection(
    bundle, tmp_path, monkeypatch
):
    working = tmp_path / "kaggle_working"
    working.mkdir()
    moved = working / "datasets" / "test"
    shutil.copytree(bundle, moved)
    monkeypatch.chdir(working)
    output = working / "manifests/clean.jsonl"
    report = import_bundle(moved, output)
    assert report["splits"] == {"train": 2, "validation": 2, "test": 2}
    originals = read_manifest(output)
    assert all(r["video"].startswith("datasets/test/clips/") for r in originals)
    assert originals[0]["relation_annotations"]["motion_speech"]["positive"] == [[1, 2]]
    controls = working / "manifests/controls.jsonl"
    make_controls(output, controls)
    rows = read_manifest(controls)
    validate(rows)
    splits = {r["video"]: r["split"] for r in originals}
    assert len(rows) == 60
    for row in rows:
        assert row["split"] == splits[row["video"]]
        if "donor" in row["variant"]:
            assert splits[row["variant"]["donor"]] == row["split"]
    assert import_bundle(moved, output) == report
    originals[0]["relation_annotations"] = {}
    write_manifest(output, originals)
    with pytest.raises(FileExistsError, match="differ"):
        import_bundle(moved, output)


def test_transitive_speaker_groups_cannot_be_split_to_satisfy_minimum(bundle, tmp_path):
    rows = read_manifest(bundle / "manifest.jsonl")
    for row in rows:
        row["speaker_id"] = "same_person"
    write_manifest(bundle / "manifest.jsonl", rows)
    info = json.loads((bundle / "dataset_info.json").read_text())
    info["manifest_sha256"] = sha(bundle / "manifest.jsonl")
    write_json(bundle / "dataset_info.json", info)
    with pytest.raises(ValueError, match="three independent"):
        import_bundle(bundle, tmp_path / "clean.jsonl")


def test_actual_exporter_to_independent_consumer(bundle, tmp_path, monkeypatch):
    sibling = Path(__file__).resolve().parents[2] / "data_pipeline/src"
    if not sibling.is_dir():
        pytest.skip("Cross-project integration requires both source projects")
    monkeypatch.syspath_prepend(str(sibling))
    from vn_av_data.data.curation import write_csv
    from vn_av_data.data.export import export_dataset

    rows = read_manifest(bundle / "manifest.jsonl")
    review = bundle / "review.csv"
    write_csv(
        review,
        [
            {
                **r,
                "file_path": r["video"],
                "decision": "keep",
                "relation_annotations": json.dumps(r["relation_annotations"]),
            }
            for r in rows
        ],
    )
    export = tmp_path / "exported"
    export_dataset(review, bundle, export, "actual_export")
    relocated = tmp_path / "new_location" / "dataset"
    shutil.copytree(export, relocated)
    report = import_bundle(relocated, tmp_path / "clean.jsonl")
    assert report["samples"] == 6
    data_contract = sibling / "vn_av_data/contract.py"
    training_contract = Path(importlib.util.find_spec("vn_av_training.contract").origin)
    assert data_contract.read_bytes() == training_contract.read_bytes()
    _, receipt = validate_bundle(relocated)
    assert receipt["dataset_id"] == "actual_export"


def test_demo_recovers_source_times_from_manifest_without_sidecar(bundle):
    from vn_av_training.data.bundle import video_provenance

    video = bundle / "clips/clip2.mp4"
    assert not video.with_suffix(".json").exists()
    provenance = video_provenance(video)
    assert provenance["source_id"] == "source1"
    assert provenance["source_start_s"] == 20
    video.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        video_provenance(video)
