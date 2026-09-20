"""Manifest import, connected provenance groups, split and label validation."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vn_av_generation.common.runtime import atomic_bytes, fingerprint, read_json, sha, write_json

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
