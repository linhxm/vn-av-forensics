"""Import reviewed clips produced by the legacy VN-AV-DF pipeline."""

import os
import re
from collections import Counter
from pathlib import Path, PureWindowsPath

from vn_av_data.common.runtime import read_json, sha, write_json
from vn_av_data.contract import valid_id, validate_bundle
from vn_av_data.data.collect import video_id
from vn_av_data.data.export import export_dataset
from vn_av_data.data.source_io import read_rows, unique_rows, write_rows

LEGACY_CLIP = re.compile(
    r"^(?P<video_id>[A-Za-z0-9_-]{11})_s(?P<start_ms>\d+)_e(?P<end_ms>\d+)$"
)
DECISIONS = {"keep", "reject", "uncertain"}


def _media_path(root, value):
    """Resolve a legacy absolute path by basename, but never escape the supplied root."""
    root = Path(root).resolve()
    portable = (value or "").replace("\\", "/")
    name = PureWindowsPath(portable).name
    candidate = (root / name).resolve()
    if not name or not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"Legacy review media missing/outside root: {value}")
    return candidate


def _legacy_rows(review, manifest, media_root):
    reviews = read_rows(review)
    if not reviews:
        raise ValueError("Legacy review CSV is empty")
    review_by_id = unique_rows(reviews)
    metadata = unique_rows(read_rows(manifest))
    decisions = Counter((row.get("decision") or "").strip().lower() for row in reviews)
    unknown = set(decisions) - DECISIONS
    if unknown:
        raise ValueError("Unsupported legacy review decision(s): " + ", ".join(sorted(unknown)))

    normalized = []
    sources = {}
    reviewers = Counter()
    for clip_id, review_row in review_by_id.items():
        if (review_row.get("decision") or "").strip().lower() != "keep":
            continue
        if not valid_id(clip_id):
            raise ValueError(f"Invalid legacy clip_id: {clip_id}")
        meta = metadata.get(clip_id)
        if meta is None:
            raise ValueError(f"Legacy manifest has no row for reviewed clip: {clip_id}")
        match = LEGACY_CLIP.fullmatch(clip_id)
        if match is None:
            raise ValueError(f"Legacy clip_id does not encode source times: {clip_id}")

        youtube_id = match.group("video_id")
        if (meta.get("source_video") or "").strip() != youtube_id:
            raise ValueError(f"{clip_id}: source_video differs from the encoded YouTube ID")
        start = int(match.group("start_ms")) / 1000
        end = int(match.group("end_ms")) / 1000
        if end <= start:
            raise ValueError(f"{clip_id}: invalid encoded source interval")
        legacy_start = float(meta.get("start_time") or start)
        legacy_end = float(meta.get("end_time") or end)
        legacy_duration = float(meta.get("duration") or end - start)
        if (
            abs(legacy_start - start) > 0.001
            or abs(legacy_end - end) > 0.001
            or abs(legacy_duration - (end - start)) > 0.002
        ):
            raise ValueError(f"{clip_id}: filename and legacy manifest times disagree")

        path = _media_path(media_root, review_row.get("file_path"))
        if path.stem != clip_id:
            raise ValueError(f"{clip_id}: reviewed filename does not match clip_id")
        url = (meta.get("url") or "").strip() or (
            "https://www.youtube.com/watch?v=" + youtube_id
        )
        if video_id(url) != youtube_id:
            raise ValueError(f"{clip_id}: URL differs from the encoded YouTube ID")
        source_id = "yt_" + youtube_id
        speaker_id = (meta.get("speaker_id") or "").strip()
        reviewer = (review_row.get("reviewer_id") or "").strip()
        if reviewer:
            reviewers[reviewer] += 1
        normalized.append(
            {
                "clip_id": clip_id,
                "file_path": path.name,
                "decision": "keep",
                "reason": (review_row.get("reason") or "").strip(),
                "bad_intervals_json": (review_row.get("bad_intervals_json") or "[]").strip(),
                "reviewer_id": reviewer,
                "rubric_version": (review_row.get("rubric_version") or "").strip(),
                "reviewed_at": (review_row.get("ts") or "").strip(),
                "sync_status": "reviewed_match",
                "source_id": source_id,
                "speaker_id": speaker_id,
                "source_start_s": start,
                "source_end_s": end,
                "url": url,
                "canonical_source_id": (meta.get("canonical_source_id") or "").strip(),
                "program_id": (meta.get("program_id") or "").strip(),
                "episode_id": (meta.get("episode_id") or "").strip(),
                "relation_annotations": "{}",
            }
        )
        source = {
            "source_id": source_id,
            "video_id": youtube_id,
            "url": url,
            "speaker_id": speaker_id,
            "title": (meta.get("title") or "").strip(),
            "channel": (meta.get("channel") or "").strip(),
            "program_id": (meta.get("program_id") or "").strip(),
            "episode_id": (meta.get("episode_id") or "").strip(),
            "tier": (meta.get("tier") or "").strip(),
            "canonical_source_id": (meta.get("canonical_source_id") or "").strip(),
        }
        previous = sources.setdefault(source_id, source)
        if previous != source:
            raise ValueError(f"Conflicting legacy source metadata: {source_id}")

    if not normalized:
        raise ValueError("Legacy review contains no keep clips")
    normalized.sort(key=lambda row: row["clip_id"])
    return normalized, [sources[key] for key in sorted(sources)], decisions, reviewers


def import_legacy_review(review, manifest, root, output, dataset_id):
    """Create a current, portable dataset bundle from one legacy review export."""
    review = Path(review).resolve()
    manifest = Path(manifest).resolve()
    root = Path(root).resolve()
    output = Path(output).resolve()
    if not valid_id(dataset_id):
        raise ValueError("dataset_id must contain letters, digits, underscore or hyphen")
    if output.exists():
        raise FileExistsError(f"Imported dataset already exists: {output}")
    normalized, sources, decisions, reviewers = _legacy_rows(review, manifest, root)

    building = output.with_name(output.name + ".building")
    partial = building.with_name(building.name + ".partial")
    normalized_path = output.with_name(output.name + ".review.partial.csv")
    if building.exists() or partial.exists() or normalized_path.exists():
        raise FileExistsError(
            "Incomplete legacy import exists; inspect/remove it or choose a new output"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_rows(normalized_path, normalized)
        export_dataset(normalized_path, root, building, dataset_id)
        os.replace(normalized_path, building / "review.csv")
        write_rows(building / "sources.csv", sources)

        info_path = building / "dataset_info.json"
        info = read_json(info_path)
        info.update(
            {
                "import_schema": "vn-av-legacy-review-import-v1",
                "legacy_review_sha256": sha(review),
                "legacy_manifest_sha256": sha(manifest),
            }
        )
        write_json(info_path, info)

        report_path = building / "quality_report.json"
        report = read_json(report_path)
        report.update(
            {
                "legacy_review_rows": sum(decisions.values()),
                "legacy_decisions": dict(decisions),
                "excluded_non_keep": decisions["reject"] + decisions["uncertain"],
                "reviewers": dict(reviewers),
                "known_speaker_clips": sum(bool(row["speaker_id"]) for row in normalized),
                "unknown_speaker_clips": sum(not bool(row["speaker_id"]) for row in normalized),
                "relation_annotations": "none; clean reviewed clips are inputs for generation",
            }
        )
        write_json(report_path, report)
        write_json(
            building / "import_report.json",
            {
                "schema_version": "vn-av-legacy-review-import-v1",
                "dataset_id": dataset_id,
                "clips_imported": len(normalized),
                "sources": len(sources),
                "excluded_reject": decisions["reject"],
                "excluded_uncertain": decisions["uncertain"],
                "speaker_id_available": any(row["speaker_id"] for row in normalized),
                "legacy_review_sha256": sha(review),
                "legacy_manifest_sha256": sha(manifest),
            },
        )
        _, validation = validate_bundle(building)
        os.rename(building, output)
    except Exception:
        normalized_path.unlink(missing_ok=True)
        raise
    return {
        **validation,
        "output": str(output),
        "sources_file": str(output / "sources.csv"),
        "review_file": str(output / "review.csv"),
        "excluded_reject": decisions["reject"],
        "excluded_uncertain": decisions["uncertain"],
        "unknown_speaker_clips": sum(not bool(row["speaker_id"]) for row in normalized),
    }
