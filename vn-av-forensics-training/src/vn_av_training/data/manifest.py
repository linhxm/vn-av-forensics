"""Manifest import, connected provenance groups, split and label validation."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vn_av_training.common.runtime import atomic_bytes, fingerprint, read_json, sha, write_json

SPLITS = {"train", "validation", "test"}


def identity_list(value):
    """CSV uses semicolon-separated IDs; JSONL may use a list."""
    if value is None:
        return []
    values = value.split(";") if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        raise ValueError("Identity fields must be a list or semicolon-separated string")
    return sorted(
        {
            str(x).strip()
            for x in values
            if str(x).strip().lower() not in ("", "unknown", "nan", "none", "-1")
        }
    )


def identity_nodes(row):
    """Local IDs are dataset-scoped; explicitly canonical IDs join datasets."""
    namespace = str(row.get("dataset", ""))
    nodes = [(namespace, "source", str(row["source_id"]))]
    nodes.extend((namespace, "source", x) for x in identity_list(row.get("parent_ids")))
    speakers = identity_list(row.get("speaker_id")) + identity_list(row.get("speaker_ids"))
    nodes.extend((namespace, "speaker", x) for x in speakers)
    nodes.extend(("global", "speaker", x) for x in identity_list(row.get("global_speaker_ids")))
    if row.get("canonical_source_id"):
        nodes.append(("global", "source", str(row["canonical_source_id"])))
    if row.get("program_id") and row.get("episode_id"):
        nodes.append((namespace, "episode", str(row["program_id"]), str(row["episode_id"])))
    if row.get("group_id"):
        nodes.append((namespace, "locked_group", str(row["group_id"])))
    return nodes


def connected_groups(rows):
    """Resolve transitive source, donor, speaker, episode and repost connections."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while node != root:
            parent[node], node = root, parent[node]
        return root

    for row in rows:
        nodes = identity_nodes(row)
        for node in nodes[1:]:
            parent[find(node)] = find(nodes[0])
    members = defaultdict(list)
    for row in rows:
        members[find(identity_nodes(row)[0])].append(row)
    return [
        sorted(group, key=lambda r: r["sample_id"])
        for group in sorted(members.values(), key=lambda group: min(r["sample_id"] for r in group))
    ]


def read_manifest(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_manifest(path, rows):
    atomic_bytes(
        path,
        (
            "\n".join(json.dumps(x, ensure_ascii=False, allow_nan=False) for x in rows) + "\n"
        ).encode(),
    )


def validate(rows, check_files=True, hash_files=False):
    ids, groups, hashes = set(), defaultdict(set), defaultdict(set)
    for r in rows:
        sid = r["sample_id"]
        if (
            not sid
            or sid in ids
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in sid
            )
        ):
            raise ValueError(f"Invalid/duplicate sample_id: {sid}")
        ids.add(sid)
        if r["split"] not in SPLITS:
            raise ValueError(f"{sid}: invalid split")
        if not r.get("source_id"):
            raise ValueError(f"{sid}: source_id required")
        if r.get("clip_label") not in (0, 1, None):
            raise ValueError(f"{sid}: clip_label must be 0, 1 or null")
        duration = float(r["duration_s"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"{sid}: invalid duration")
        for key in ("forgery_intervals", "mismatch_intervals"):
            spans = r.get(key)
            if spans is not None:
                for a, b in spans:
                    if not (np.isfinite(a) and np.isfinite(b) and 0 <= a < b <= duration + 0.05):
                        raise ValueError(f"{sid}: invalid {key}: {[a, b]}")
        if r.get("clip_label") == 0 and r.get("forgery_intervals"):
            raise ValueError(f"{sid}: real clip cannot have forgery intervals")
        if r.get("clip_label") == 1 and r.get("forgery_intervals") == []:
            raise ValueError(
                f"{sid}: fake clip has explicitly empty forgery intervals; use null if unknown"
            )
        for node in identity_nodes(r):
            groups[node].add(r["split"])
        if check_files and not Path(r["video"]).is_file():
            raise FileNotFoundError(r["video"])
        if hash_files:
            hashes[sha(r["video"])].add(r["split"])
    leaks = [str(k) for k, v in groups.items() if len(v) > 1]
    if leaks or any(len(v) > 1 for v in hashes.values()):
        raise ValueError(f"Cross-split source/speaker/content leakage: {leaks[:5]}")
    if not rows:
        raise ValueError("Manifest is empty")
    return {
        "samples": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "labels": dict(Counter(str(r.get("clip_label")) for r in rows)),
        "unknown_speakers": sum(
            not (
                identity_list(r.get("speaker_id"))
                or identity_list(r.get("speaker_ids"))
                or identity_list(r.get("global_speaker_ids"))
            )
            for r in rows
        ),
        "independent_groups": len(connected_groups(rows)),
    }


def import_dataset(kind, metadata, root, output, seed=42):
    root = Path(root).resolve()
    if kind == "csv":
        with Path(metadata).open(encoding="utf-8-sig", newline="") as f:
            original = list(csv.DictReader(f))
    else:
        original = read_json(metadata)
        if isinstance(original, dict):
            original = original.get("metadata", original.get("data", list(original.values())))
    rows = []
    for i, r in enumerate(original):
        file = r.get("file", r.get("video", r.get("path")))
        if not file:
            raise ValueError(f"Record {i}: missing file/video/path")
        source = str(r.get("original") or r.get("source_id") or file)
        import hashlib

        sid = r.get("sample_id") or hashlib.sha256((kind + ":" + file).encode()).hexdigest()[:20]
        label = r.get("clip_label", r.get("label"))
        spans = r.get("fake_periods", r.get("forgery_intervals"))
        if isinstance(spans, str):
            spans = json.loads(spans) if spans else None
        if kind == "lavdf":
            label = int(r["n_fakes"] > 0)
        elif label in ("", None):
            label = int(bool(spans)) if spans is not None else None
        else:
            label = int(label)
        if label == 0 and spans is None:
            spans = []
        split = {"dev": "validation", "val": "validation"}.get(r.get("split"), r.get("split"))
        duration = float(r.get("duration_s") or r.get("duration") or 0)
        if duration <= 0:
            from vn_av_training.data.media import probe

            duration = probe(root / file)["duration_s"]
            if not duration:
                raise ValueError(f"{file}: unknown duration; provide duration_s")
        rows.append(
            {
                "sample_id": sid,
                "source_id": source,
                "speaker_id": r.get("speaker_id") or None,
                "video": str((root / file).resolve()),
                "dataset": r.get("dataset") or kind,
                "language": r.get("language", "vi" if kind == "csv" else "unknown"),
                "split": split,
                "clip_label": label,
                "forgery_intervals": spans,
                "mismatch_intervals": None,
                "duration_s": duration,
                "generator": r.get("generator", "unknown"),
                "parent_ids": identity_list(r.get("parent_ids")),
                "speaker_ids": identity_list(r.get("speaker_ids")),
                "global_speaker_ids": identity_list(r.get("global_speaker_ids")),
                "canonical_source_id": r.get("canonical_source_id") or None,
                "program_id": r.get("program_id") or None,
                "episode_id": r.get("episode_id") or None,
                "group_id": r.get("group_id") or None,
            }
        )
    if any(not r["split"] for r in rows):
        if not all(not r["split"] for r in rows):
            raise ValueError("Provide all splits or leave all empty")
        rows = group_split(rows, seed)
    validate(rows)
    write_manifest(output, rows)
    return validate(rows)


def group_split(rows, seed=42):
    groups = connected_groups(rows)
    if len(groups) < 3:
        raise ValueError("Need at least 3 independent source/speaker groups")
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    names, ratios = ("train", "validation", "test"), (0.7, 0.15, 0.15)
    targets, filled, assignments = [len(rows) * r for r in ratios], [0, 0, 0], {}
    for i, group in enumerate(groups):
        empty = [j for j in range(3) if filled[j] == 0]
        choices = empty if len(groups) - i == len(empty) else range(3)
        chosen = max(choices, key=lambda j: (targets[j] - filled[j]) / targets[j])
        filled[chosen] += len(group)
        gid = (
            "group_"
            + fingerprint(sorted({node for row in group for node in identity_nodes(row)}))[:16]
        )
        for row in group:
            assignments[row["sample_id"]] = (names[chosen], gid)
    return [
        dict(row, split=assignments[row["sample_id"]][0], group_id=assignments[row["sample_id"]][1])
        for row in rows
    ]


def subset(manifest, output, groups_per_split=20, seed=42):
    rows = read_manifest(manifest)
    validate(rows)
    groups = connected_groups(rows)
    result = []
    rng = random.Random(seed)
    for split in sorted(SPLITS):
        selected = [group for group in groups if group[0]["split"] == split]
        rng.shuffle(selected)
        for group in selected[:groups_per_split]:
            result.extend(group)
    validate(result)
    write_manifest(output, result)
    return validate(result)


def filter_excluded(manifest, exclusions, output):
    rows = read_manifest(manifest)
    errors = read_json(exclusions)
    excluded = {r["sample_id"] for r in errors}
    unknown = excluded - {r["sample_id"] for r in rows}
    if unknown:
        raise ValueError(f"Exclusion IDs not in manifest: {sorted(unknown)[:5]}")
    selected = [r for r in rows if r["sample_id"] not in excluded]
    validate(selected)
    write_manifest(output, selected)
    report = {"before": len(rows), "after": len(selected), "excluded": errors}
    write_json(str(output) + ".audit.json", report)
    return report
