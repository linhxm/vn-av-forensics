"""Synchronize a verified speaker assignment through a consolidated dataset version."""

from pathlib import Path

from vn_av_data.common.runtime import read_json, sha, write_json
from vn_av_data.contract import valid_id, validate_bundle
from vn_av_data.data.manifest import read_manifest, write_manifest
from vn_av_data.data.source_io import read_rows, write_rows


def _assign_rows(rows, source_ids, speaker_id, label):
    changed = 0
    for row in rows:
        if row.get("source_id") not in source_ids:
            continue
        current = row.get("speaker_id") or ""
        if current and current != speaker_id:
            raise ValueError(f"{label}: refusing to replace {current} for {row['source_id']}")
        if current != speaker_id:
            row["speaker_id"] = speaker_id
            changed += 1
    return changed


def assign_speaker(data_root, export, dataset_id, speaker_id, availability):
    data_root, export = Path(data_root).resolve(), Path(export).resolve()
    if not valid_id(dataset_id) or not valid_id(speaker_id):
        raise ValueError("dataset_id and speaker_id must be safe identifiers")
    validate_bundle(export)
    sources_dir = data_root / "sources" / dataset_id
    raw_dir = data_root / "raw" / dataset_id / "download_001"
    cut_dir = data_root / "candidates" / dataset_id / "cut_001"
    manifests_dir = data_root / "manifests" / dataset_id
    required = [sources_dir, raw_dir, cut_dir, manifests_dir]
    if any(not path.is_dir() for path in required):
        raise ValueError("Consolidated dataset directories are incomplete")

    files = {
        "videos": sources_dir / "videos.csv",
        "selected": sources_dir / "selected_videos.csv",
        "download_sources": raw_dir / "download_sources.csv",
        "download_results": raw_dir / "download_results.csv",
        "candidates": cut_dir / "candidates.csv",
        "review": cut_dir / "review.csv",
        "clips": manifests_dir / "clips.csv",
    }
    rows = {name: read_rows(path) for name, path in files.items()}
    source_ids = {
        row["source_id"] for row in rows["videos"] if row.get("availability") == availability
    }
    if not source_ids:
        raise ValueError(f"No sources have availability={availability}")
    for name in ("videos", "selected", "download_sources", "download_results"):
        present = {row["source_id"] for row in rows[name]}
        if not source_ids <= present:
            raise ValueError(f"{name}: selected sources are missing")

    changed = {}
    for name, table in rows.items():
        changed[name] = _assign_rows(table, source_ids, speaker_id, name)
    export_rows = read_manifest(export / "manifest.jsonl")
    changed["export_manifest"] = _assign_rows(
        export_rows, source_ids, speaker_id, "export_manifest"
    )
    raw_rows = read_manifest(raw_dir / "sources.jsonl")
    changed["raw_manifest"] = _assign_rows(raw_rows, source_ids, speaker_id, "raw_manifest")

    journals = []
    journal_changes = 0
    for path in sorted((cut_dir / "sources").glob("*.json")):
        value = read_json(path)
        count = _assign_rows(value.get("candidates", []), source_ids, speaker_id, path.name)
        count += _assign_rows(value.get("rejected", []), source_ids, speaker_id, path.name)
        journals.append((path, value))
        journal_changes += count
    changed["source_journals"] = journal_changes

    report = {
        "schema_version": "vn-av-speaker-assignment-v1",
        "dataset_id": dataset_id,
        "speaker_id": speaker_id,
        "selector": {"availability": availability},
        "sources": sorted(source_ids),
        "source_count": len(source_ids),
        "changed_rows": changed,
    }
    for name, table in rows.items():
        write_rows(files[name], table, mutable=True)
    write_manifest(raw_dir / "sources.jsonl", raw_rows)
    for path, value in journals:
        write_json(path, value)
    write_manifest(export / "manifest.jsonl", export_rows)
    write_json(manifests_dir / "speaker_assignment_report.json", report)

    info_path = export / "dataset_info.json"
    info = read_json(info_path)
    info["manifest_sha256"] = sha(export / "manifest.jsonl")
    info["speaker_assignment_sha256"] = sha(
        manifests_dir / "speaker_assignment_report.json"
    )
    write_json(info_path, info)
    quality_path = export / "quality_report.json"
    quality = read_json(quality_path)
    quality["manifest_sha256"] = info["manifest_sha256"]
    quality["speaker_assignment"] = report
    write_json(quality_path, quality)
    _, validation = validate_bundle(export)
    return {
        **validation,
        "speaker_id": speaker_id,
        "assigned_sources": len(source_ids),
        "changed_rows": changed,
        "report": str(manifests_dir / "speaker_assignment_report.json"),
        "zip_requires_rebuild": True,
    }
