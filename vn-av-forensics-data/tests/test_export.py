import json
import shutil

import pytest

from vn_av_data.common.runtime import sha, write_json
from vn_av_data.contract import validate_bundle
from vn_av_data.data.curation import write_csv
from vn_av_data.data.export import export_dataset


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    root = tmp_path / "reviewed"
    root.mkdir()
    rows = []
    for i in range(3):
        path = root / f"clip{i}.mp4"
        path.write_bytes(f"fixture-video-{i}".encode())
        rows.append(
            {
                "clip_id": f"clip{i}",
                "file_path": path.name,
                "decision": "keep",
                "sync_status": "reviewed_match",
                "source_id": f"source{i}",
                "source_start_s": 10,
                "source_end_s": 14,
                "sha256": sha(path),
            }
        )
    rows.append({"clip_id": "deleted", "file_path": "missing.mp4", "decision": "reject"})
    review = root / "review.csv"
    write_csv(review, rows)
    monkeypatch.setattr("vn_av_data.data.export.probe", lambda _: {"duration_s": 4})
    return root, review, rows


def test_export_relocation_labels_and_immutable_output(reviewed, tmp_path):
    root, review, _ = reviewed
    labels = tmp_path / "annotations.jsonl"
    labels.write_text(
        json.dumps(
            {
                "clip_id": "clip0",
                "relation_annotations": {
                    "motion_speech": {"known": [[0, 4]], "positive": [[1, 2]]}
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "export"
    report = export_dataset(review, root, output, "fixture_v001", labels)
    assert report["clips"] == 3
    relocated = tmp_path / "other_machine" / "dataset"
    shutil.copytree(output, relocated)
    rows, _ = validate_bundle(relocated)
    assert all("split" not in row and row["video"].startswith("clips/") for row in rows)
    assert rows[0]["relation_annotations"]["motion_speech"]["positive"] == [[1, 2]]
    assert rows[0]["source_start_s"] == 10
    with pytest.raises(FileExistsError):
        export_dataset(review, root, output, "fixture_v001")
    (relocated / rows[0]["video"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle(relocated)


def test_export_requires_sync_confirmation_and_detects_changed_review(reviewed, tmp_path):
    root, review, rows = reviewed
    rows[0]["sync_status"] = "unverified"
    write_csv(review, rows)
    with pytest.raises(ValueError, match="confirm synchrony"):
        export_dataset(review, root, tmp_path / "out", "test")
    rows[0]["sync_status"] = "reviewed_match"
    write_csv(review, rows)
    (root / "clip0.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="has changed"):
        export_dataset(review, root, tmp_path / "out", "test")


@pytest.mark.parametrize("bad_path", ["../outside.mp4", "/outside.mp4", "C:/outside.mp4"])
def test_consumer_rejects_unsafe_paths_even_with_updated_manifest_hash(
    reviewed, tmp_path, bad_path
):
    root, review, _ = reviewed
    output = tmp_path / "out"
    export_dataset(review, root, output, "test")
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["video"] = bad_path
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    info = json.loads((output / "dataset_info.json").read_text())
    info["manifest_sha256"] = sha(manifest)
    write_json(output / "dataset_info.json", info)
    with pytest.raises(ValueError, match="Unsafe"):
        validate_bundle(output)


def test_dedup_and_invalid_annotation_do_not_silently_lose_labels(reviewed, tmp_path):
    root, review, rows = reviewed
    rows.insert(1, {**rows[0], "clip_id": "duplicate"})
    write_csv(review, rows)
    report = export_dataset(review, root, tmp_path / "dedup", "test")
    assert report["clips"] == 3
    assert report["duplicate_clips_skipped"] == ["duplicate"]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "clip_id": "duplicate",
                "relation_annotations": {"sequence": {"known": [[0, 4]], "positive": [[1, 2]]}},
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate with annotations"):
        export_dataset(review, root, tmp_path / "bad", "test", labels)
