"""vn-av-dataset-v1 wire contract. Standalone copy in each independent package.

Keep both copies identical; cross-pipeline integration tests verify compatibility.
Only standard-library dependencies are required to inspect a bundle.
"""

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

SCHEMA = "vn-av-dataset-v1"
HEADS = {"phoneme_viseme", "sequence", "motion_speech", "source", "audio_speech", "visual_speech"}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None


def bundle_path(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Bundle paths must use relative POSIX paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts:
        raise ValueError(f"Unsafe bundle path: {relative}")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"Bundle file missing or outside root: {relative}")
    return target


def validate_annotations(annotations, duration):
    if not isinstance(annotations, dict) or set(annotations) - HEADS:
        raise ValueError("Invalid relation_annotations names")
    for name, annotation in annotations.items():
        if not isinstance(annotation, dict) or set(annotation) - {"known", "positive"}:
            raise ValueError(f"Invalid annotation: {name}")
        known, positive = annotation.get("known", []), annotation.get("positive", [])
        if not isinstance(known, list) or not isinstance(positive, list):
            raise ValueError("Annotation spans must be lists")
        for span in known + positive:
            if not isinstance(span, list) or len(span) != 2:
                raise ValueError("Annotation spans require [start, end]")
            a, b = span
            if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in span):
                raise ValueError("Annotation times must be finite numbers")
            if not 0 <= a < b <= duration + 1e-6:
                raise ValueError("Annotation interval outside clip")
        if any(not any(x <= a and b <= y for x, y in known) for a, b in positive):
            raise ValueError("Positive annotations must be inside known regions")


def validate_bundle(root, probe=None):
    root = Path(root).resolve()
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    if info.get("schema_version") != SCHEMA or not valid_id(info.get("dataset_id")):
        raise ValueError("Unsupported dataset schema or invalid dataset_id")
    manifest = bundle_path(root, "manifest.jsonl")
    if file_sha(manifest) != info.get("manifest_sha256"):
        raise ValueError("Dataset manifest hash mismatch; export a new dataset version")
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or info.get("clips") != len(rows):
        raise ValueError("Dataset clip count mismatch or empty dataset")
    ids, hashes, sources = set(), set(), set()
    for row in rows:
        sid = row.get("clip_id")
        if not valid_id(sid) or sid in ids:
            raise ValueError(f"Invalid/duplicate clip_id: {sid}")
        ids.add(sid)
        if "split" in row or "variant" in row:
            raise ValueError("Exported datasets must not contain training splits or variants")
        if not isinstance(row.get("source_id"), str) or not row["source_id"].strip():
            raise ValueError(f"{sid}: source_id required")
        sources.add(row["source_id"])
        if row.get("review_decision") != "keep" or row.get("sync_status") != "reviewed_match":
            raise ValueError(f"{sid}: reviewed matching clip required")
        duration = row.get("duration_s")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{sid}: invalid duration")
        start, end = row.get("source_start_s"), row.get("source_end_s")
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)):
            raise ValueError(f"{sid}: finite source times required")
        if not 0 <= start < end or abs(end - start - duration) > 0.25:
            raise ValueError(f"{sid}: invalid source interval")
        path = bundle_path(root, row.get("video"))
        if PurePosixPath(row["video"]).parts[0] != "clips":
            raise ValueError("Videos must be stored under clips/")
        digest = file_sha(path)
        if digest != row.get("sha256"):
            raise ValueError(f"{sid}: video hash mismatch")
        if digest in hashes:
            raise ValueError(f"{sid}: duplicate video content")
        hashes.add(digest)
        validate_annotations(row.get("relation_annotations", {}), duration)
        if probe is not None:
            actual = probe(path).get("duration_s")
            if actual is None or not math.isfinite(actual) or abs(actual - duration) > 0.25:
                raise ValueError(f"{sid}: media duration mismatch")
    return rows, {
        "schema_version": SCHEMA,
        "dataset_id": info["dataset_id"],
        "manifest_sha256": info["manifest_sha256"],
        "clips": len(rows),
        "sources": len(sources),
        "annotated_clips": dict(
            Counter(name for row in rows for name in row.get("relation_annotations", {}))
        ),
    }
