"""Consume the versioned dataset contract; split before generating any controls."""

import random
from pathlib import Path

from vn_av_training.common.runtime import write_json
from vn_av_training.contract import validate_bundle
from vn_av_training.data.manifest import connected_groups, read_manifest, validate, write_manifest
from vn_av_training.data.media import probe


def video_provenance(video):
    """Read only the selected clip's provenance; do not hash an entire training dataset."""
    from vn_av_training.common.runtime import read_json, sha
    from vn_av_training.contract import SCHEMA, bundle_path

    video = Path(video).resolve()
    if video.parent.name != "clips":
        return None
    root = video.parent.parent
    if not (root / "dataset_info.json").is_file():
        return None
    info = read_json(root / "dataset_info.json")
    manifest = root / "manifest.jsonl"
    if info.get("schema_version") != SCHEMA or info.get("manifest_sha256") != sha(manifest):
        raise ValueError("Dataset provenance manifest mismatch")
    for row in read_manifest(manifest):
        if row.get("video") == video.relative_to(root).as_posix():
            if bundle_path(root, row["video"]) != video or row["sha256"] != sha(video):
                raise ValueError("Dataset provenance video hash mismatch")
            return row
    raise ValueError("Video not listed in its dataset manifest")


def import_bundle(dataset, output, seed=42):
    dataset = Path(dataset).resolve()
    original, receipt = validate_bundle(dataset, probe=probe)
    rows = []
    for item in original:
        video = (dataset / item["video"]).resolve()
        stored = (
            video.relative_to(Path.cwd()).as_posix()
            if video.is_relative_to(Path.cwd())
            else str(video)
        )
        rows.append(
            {
                **{k: v for k, v in item.items() if k not in {"clip_id", "video"}},
                "sample_id": item["clip_id"],
                "video": stored,
                "dataset": receipt["dataset_id"],
                "dataset_manifest_sha256": receipt["manifest_sha256"],
                "split": "train",
                "variant": {"kind": "clean"},
                "sync_label_quality": "reviewed",
            }
        )
    groups = connected_groups(rows)
    if len(groups) < 3:
        raise ValueError("Need at least three independent source/speaker groups after validation")
    random.Random(seed).shuffle(groups)
    held_out = max(1, round(len(groups) * 0.15))
    for i, group in enumerate(groups):
        split = "test" if i < held_out else "validation" if i < held_out * 2 else "train"
        for row in group:
            row["split"] = split
    report = validate(rows)
    output = Path(output)
    if output.exists() and read_manifest(output) != rows:
        raise FileExistsError("Imported dataset/splits differ; use a new manifest and run")
    write_manifest(output, rows)
    report.update(
        dataset=receipt,
        seed=seed,
        speaker_note="Unknown speaker IDs guarantee only source-disjoint splits",
    )
    write_json(output.with_suffix(".report.json"), report)
    return report
