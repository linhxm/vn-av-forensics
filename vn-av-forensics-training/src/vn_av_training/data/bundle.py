"""Consume the versioned dataset contract; split before generating any controls."""

from pathlib import Path

from vn_av_training.data.manifest import read_manifest


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
    if info.get("schema_version") not in {
        SCHEMA,
        "vn-av-relations-v1",
        "vn-av-relations-v2",
    } or info.get("manifest_sha256") != sha(manifest):
        raise ValueError("Dataset provenance manifest mismatch")
    for row in read_manifest(manifest):
        if row.get("video") == video.relative_to(root).as_posix():
            if bundle_path(root, row["video"]) != video or row["sha256"] != sha(video):
                raise ValueError("Dataset provenance video hash mismatch")
            return row
    raise ValueError("Video not listed in its dataset manifest")
