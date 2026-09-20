"""Consolidate a native cut run and an imported reviewed bundle into one version."""

import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from vn_av_data.common.runtime import fingerprint, read_json, sha, write_json
from vn_av_data.contract import bundle_path, valid_id, validate_bundle
from vn_av_data.data.export import export_dataset
from vn_av_data.data.manifest import read_manifest, write_manifest
from vn_av_data.data.source_io import read_rows, unique_rows, write_rows


def _copy_verified(source, target, digest=None):
    source, target = Path(source), Path(target)
    if digest and sha(source) != digest:
        raise ValueError(f"Source file hash changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest and sha(target) != digest:
        raise ValueError(f"Copied file hash mismatch: {target}")


def _source_inventory(raw_manifest, imported):
    raw_rows = read_manifest(raw_manifest)
    raw_by_id = {row["source_id"]: row for row in raw_rows}
    if len(raw_by_id) != len(raw_rows):
        raise ValueError("Duplicate source_id in native raw manifest")
    inventory = {}
    for row in raw_rows:
        youtube_id = row["source_id"].removeprefix("yt_")
        inventory[row["source_id"]] = {
            "source_id": row["source_id"],
            "video_id": youtube_id,
            "url": row.get("url", ""),
            "speaker_id": row.get("speaker_id") or "",
            "title": row.get("title", ""),
            "channel": row.get("channel", ""),
            "program_id": row.get("program_id", ""),
            "episode_id": row.get("episode_id", ""),
            "tier": row.get("tier", ""),
            "canonical_source_id": row.get("canonical_source_id", ""),
            "availability": "raw_and_clips",
        }
    for row in read_rows(Path(imported) / "sources.csv"):
        sid = row["source_id"]
        incoming = {
            "source_id": sid,
            "video_id": row["video_id"],
            "url": row["url"],
            "speaker_id": row.get("speaker_id") or "",
            "title": row.get("title") or "",
            "channel": row.get("channel") or "",
            "program_id": row.get("program_id") or "",
            "episode_id": row.get("episode_id") or "",
            "tier": row.get("tier") or "",
            "canonical_source_id": row.get("canonical_source_id") or "",
            "availability": "imported_clips_only",
        }
        if sid not in inventory:
            inventory[sid] = incoming
            continue
        current = inventory[sid]
        if current["video_id"] != incoming["video_id"] or current["url"] != incoming["url"]:
            raise ValueError(f"Conflicting source identity: {sid}")
        for key in (
            "speaker_id",
            "title",
            "channel",
            "program_id",
            "episode_id",
            "tier",
            "canonical_source_id",
        ):
            if current[key] and incoming[key] and current[key] != incoming[key]:
                raise ValueError(f"Conflicting {key}: {sid}")
            current[key] = current[key] or incoming[key]
    return [inventory[key] for key in sorted(inventory)], raw_rows, raw_by_id


def _native_journals(old_cut):
    result = {}
    for path in (Path(old_cut) / "sources").glob("*.json"):
        journal = read_json(path)
        candidates = journal.get("candidates") or []
        rejected = journal.get("rejected") or []
        sample = (candidates or rejected)
        if not sample:
            continue
        result[sample[0]["source_id"]] = journal
    return result


def consolidate_reviewed_version(old_cut, old_raw, imported, data_root, output, dataset_id):
    old_cut, old_raw = Path(old_cut).resolve(), Path(old_raw).resolve()
    imported, data_root = Path(imported).resolve(), Path(data_root).resolve()
    output = Path(output).resolve()
    if not valid_id(dataset_id):
        raise ValueError("Invalid dataset_id")
    if output.exists():
        raise FileExistsError(f"Merged export already exists: {output}")
    imported_rows, _ = validate_bundle(imported)
    imported_reviews = unique_rows(read_rows(imported / "review.csv"))
    native_reviews = read_rows(old_cut / "review.csv")
    unique_rows(native_reviews)
    inventory, raw_sources, raw_by_id = _source_inventory(old_raw / "sources.jsonl", imported)
    source_by_id = {row["source_id"]: row for row in inventory}

    stage = data_root / ("." + dataset_id + ".consolidating")
    targets = {
        "sources": data_root / "sources" / dataset_id,
        "raw": data_root / "raw" / dataset_id / "download_001",
        "candidates": data_root / "candidates" / dataset_id,
        "manifests": data_root / "manifests" / dataset_id,
    }
    if stage.exists() or any(path.exists() for path in targets.values()):
        raise FileExistsError("Consolidation staging/target exists; inspect it before retrying")
    stage_sources = stage / "sources"
    stage_raw = stage / "raw"
    stage_cut = stage / "candidates" / "cut_001"
    stage_manifests = stage / "manifests"
    stage_sources.mkdir(parents=True)
    stage_raw.mkdir(parents=True)
    (stage_cut / "clips").mkdir(parents=True)
    (stage_cut / "sources").mkdir()
    stage_manifests.mkdir(parents=True)

    try:
        write_rows(stage_sources / "videos.csv", inventory)
        write_rows(stage_sources / "selected_videos.csv", inventory)

        raw_results = []
        for source in inventory:
            raw = raw_by_id.get(source["source_id"])
            filename = ""
            if raw:
                source_path = old_raw / raw["video"]
                filename = Path(raw["video"]).name
                _copy_verified(source_path, stage_raw / filename, raw["sha256"])
            raw_results.append(
                {
                    **source,
                    "status": "downloaded" if raw else "imported_clips_only",
                    "filename": filename,
                    "error": "" if raw else "Full source video was not present in the legacy import",
                }
            )
        write_rows(stage_raw / "download_sources.csv", inventory)
        write_rows(stage_raw / "download_results.csv", raw_results, mutable=True)
        write_manifest(stage_raw / "sources.jsonl", raw_sources)
        write_json(stage_raw / "download-errors.json", [])

        combined = []
        ids, hashes = set(), set()
        for row in native_reviews:
            clip_id, digest = row["clip_id"], row["sha256"]
            if clip_id in ids or digest in hashes:
                raise ValueError(f"Duplicate native candidate: {clip_id}")
            source = old_cut / row["file_path"]
            target = stage_cut / "clips" / source.name
            _copy_verified(source, target, digest)
            receipt = source.with_suffix(".json")
            if not receipt.is_file():
                raise ValueError(f"Missing native candidate receipt: {receipt}")
            _copy_verified(receipt, target.with_suffix(".json"))
            ids.add(clip_id)
            hashes.add(digest)
            combined.append({**row, "file_path": f"clips/{target.name}"})

        for row in imported_rows:
            clip_id, digest = row["clip_id"], row["sha256"]
            if clip_id in ids or digest in hashes:
                raise ValueError(f"Duplicate imported candidate: {clip_id}")
            source = bundle_path(imported, row["video"])
            target = stage_cut / "clips" / source.name
            _copy_verified(source, target, digest)
            source_meta = source_by_id[row["source_id"]]
            review = imported_reviews[clip_id]
            candidate = {
                "clip_id": clip_id,
                "source_id": row["source_id"],
                "url": row.get("url") or source_meta["url"],
                "source_sha256": raw_by_id.get(row["source_id"], {}).get("sha256", ""),
                "speaker_id": source_meta["speaker_id"],
                "dataset": "legacy_review_import",
                "source_start_s": row["source_start_s"],
                "source_end_s": row["source_end_s"],
                "face_ratio": "",
                "visual_sample_coverage": "",
                "duration_s": row["duration_s"],
                "file_path": f"clips/{target.name}",
                "sha256": digest,
                "quality": "reviewed_legacy_import",
                "decision": "keep",
                "sync_status": "reviewed_match",
                "reviewer_id": review.get("reviewer_id", ""),
                "rubric_version": review.get("rubric_version", ""),
                "reviewed_at": review.get("reviewed_at", ""),
                "relation_annotations": json.dumps(row.get("relation_annotations", {})),
            }
            for key in ("canonical_source_id", "program_id", "episode_id", "tier"):
                candidate[key] = source_meta.get(key, "")
            write_json(
                target.with_suffix(".json"),
                {
                    "format": "curated-clip-v1",
                    "signature": fingerprint(["legacy_review_import", digest]),
                    "sha256": digest,
                    "source_id": row["source_id"],
                    "source_start_s": row["source_start_s"],
                    "source_end_s": row["source_end_s"],
                },
            )
            ids.add(clip_id)
            hashes.add(digest)
            combined.append(candidate)

        rejected = read_json(old_cut / "rejected.json")
        journals = _native_journals(old_cut)
        candidates_by_source = defaultdict(list)
        rejected_by_source = defaultdict(list)
        for row in combined:
            candidates_by_source[row["source_id"]].append(row)
        for row in rejected:
            rejected_by_source[row["source_id"]].append(row)
        for source in inventory:
            sid = source["source_id"]
            original = journals.get(sid, {})
            write_json(
                stage_cut / "sources" / (fingerprint(sid)[:20] + ".json"),
                {
                    "candidates": candidates_by_source[sid],
                    "rejected": rejected_by_source[sid],
                    "speech_regions": original.get("speech_regions", []),
                    "scene_boundaries": original.get("scene_boundaries", []),
                    "lineage": (
                        "native_and_legacy_import"
                        if sid in raw_by_id and any(
                            row.get("dataset") == "legacy_review_import"
                            for row in candidates_by_source[sid]
                        )
                        else "native_cut"
                        if sid in raw_by_id
                        else "legacy_review_import"
                    ),
                },
            )
        write_rows(stage_cut / "candidates.csv", combined)
        write_rows(stage_cut / "review.csv", combined)
        write_json(stage_cut / "rejected.json", rejected)
        write_json(stage_cut / "errors.json", [])
        native_lock = read_json(old_cut / "cut-run.json")
        write_json(
            stage_cut / "cut-run.json",
            {
                "format": "consolidated-cut-run-v1",
                "signature": fingerprint(
                    {
                        "native": native_lock.get("signature"),
                        "imported": read_json(imported / "dataset_info.json")["manifest_sha256"],
                    }
                ),
                "native_cut_run": native_lock,
                "legacy_import_manifest_sha256": read_json(imported / "dataset_info.json")[
                    "manifest_sha256"
                ],
            },
        )
        decisions = Counter(row.get("decision") for row in combined)
        write_json(
            stage_cut / "summary.json",
            {
                "format": "consolidated-candidates-v1",
                "sources": len(inventory),
                "raw_sources_available": len(raw_sources),
                "imported_clip_only_sources": len(inventory) - len(raw_sources),
                "candidates": len(combined),
                "rejected_during_cut": len(rejected),
                "review_decisions": dict(decisions),
                "errors": 0,
            },
        )

        manifest_rows = [
            {**row, "file_path": "cut_001/" + row["file_path"]} for row in combined
        ]
        write_rows(stage_manifests / "clips.csv", manifest_rows)
        report = {
            "schema_version": "vn-av-consolidation-v1",
            "dataset_id": dataset_id,
            "sources": len(inventory),
            "raw_sources_available": len(raw_sources),
            "imported_clip_only_sources": len(inventory) - len(raw_sources),
            "native_candidates": len(native_reviews),
            "imported_keep_candidates": len(imported_rows),
            "total_candidates": len(combined),
            "review_decisions": dict(decisions),
            "expected_export_keep": decisions["keep"],
            "native_review_sha256": sha(old_cut / "review.csv"),
            "imported_manifest_sha256": sha(imported / "manifest.jsonl"),
        }
        write_json(stage_manifests / "consolidation_report.json", report)

        export_dataset(stage_manifests / "clips.csv", stage / "candidates", output, dataset_id)
        export_report = read_json(output / "quality_report.json")
        export_report["consolidation"] = report
        write_json(output / "quality_report.json", export_report)
        write_json(output / "consolidation_report.json", report)
        validate_bundle(output)

        for key, target in targets.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stage / key
            os.rename(source, target)
        stage.rmdir()
    except Exception:
        raise
    return {
        **report,
        "standardized_paths": {key: str(path) for key, path in targets.items()},
        "merged_export": str(output),
    }
