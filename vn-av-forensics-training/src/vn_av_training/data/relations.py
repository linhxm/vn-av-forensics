"""Reviewed positives, virtual media controls, and partial relation annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vn_av_training.data.manifest import read_manifest, validate, write_manifest
from vn_av_training.models.relations import ACTIVITIES, RELATIONS


def make_controls(manifest, output, step_s=0.2, radius=4):
    if step_s <= 0 or radius < 1:
        raise ValueError("Positive step_s and radius are required")
    originals = read_manifest(manifest)
    validate(originals)
    if any(row.get("variant", {}).get("kind", "clean") != "clean" for row in originals):
        raise ValueError("Controls must be created from original clean rows")
    rows = []
    # Use multiple magnitudes, retaining the exact signed offset in seconds.
    shifts = sorted({-radius, -1, 1, radius})
    for row in originals:
        rows.append({**row, "variant": {"kind": "clean"}})
        for index, shift in enumerate(shifts):
            for local in (False, True):
                variant = {"kind": "local_lag" if local else "global_lag", "lag_s": shift * step_s}
                if local:
                    variant["span"] = [row["duration_s"] * 0.25, row["duration_s"] * 0.75]
                rows.append(
                    {
                        **row,
                        "sample_id": row["sample_id"] + f"_lag{index}_{int(local)}",
                        "parent_ids": [row["source_id"]],
                        "variant": variant,
                        "relation_annotations": {},
                    }
                )
        donors = [
            r
            for r in originals
            if r["split"] == row["split"]
            and r["sample_id"] != row["sample_id"]
            and r["source_id"] == row["source_id"]
        ]
        if donors:
            donor = donors[0]
            rows.append(
                {
                    **row,
                    "sample_id": row["sample_id"] + "_sequence",
                    "parent_ids": [row["source_id"], donor["source_id"]],
                    "variant": {
                        "kind": "sequence_swap",
                        "donor": donor["video"],
                        "label_quality": "synthetic_proxy_requires_audit",
                    },
                    "relation_annotations": {},
                }
            )
    validate(rows)
    if Path(output).exists() and read_manifest(output) != rows:
        raise FileExistsError("Controls differ; use a new output path")
    write_manifest(output, rows)
    return {
        "originals": len(originals),
        "samples": len(rows),
        "note": "Sequence swap is proxy supervision; phoneme/activity/source positives need annotations",
    }


def apply_variant(decoded, variant, donor=None, sample_rate=48000):
    """Edit raw audio before feature extraction; never wrap shifted samples."""
    result = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in decoded.items()
    }
    kind = variant.get("kind", "clean")
    if kind == "clean":
        return result
    if kind == "sequence_swap":
        if donor is None:
            raise ValueError("Sequence swap requires decoded donor media")
        length = min(len(result["pcm"]), len(donor["pcm"]))
        result["pcm"][:] = 0
        result["pcm"][:length] = donor["pcm"][:length]
        result["audio_valid"][:] = False
        n = min(len(result["audio_valid"]), len(donor["audio_valid"]))
        result["audio_valid"][:n] = donor["audio_valid"][:n]
        return result
    if kind not in ("global_lag", "local_lag"):
        raise ValueError(f"Unsupported raw-media variant: {kind}")
    shift = round(float(variant["lag_s"]) * sample_rate)
    n = len(result["pcm"])
    index = np.arange(n) + shift
    inside = (index >= 0) & (index < n)
    index = index.clip(0, n - 1)
    changed = np.ones(n, dtype=bool)
    if kind == "local_lag":
        start, end = variant["span"]
        changed = (np.arange(n) / sample_rate >= start) & (np.arange(n) / sample_rate < end)
    original_valid = np.repeat(decoded["audio_valid"], sample_rate // 25)
    result["pcm"][changed] = np.where(inside, decoded["pcm"][index], 0)[changed]
    valid = original_valid.copy()
    valid[changed] = (inside & original_valid[index])[changed]
    result["audio_valid"] = valid.reshape(-1, sample_rate // 25).mean(-1) >= 0.95
    return result


def labels_for(row, times, window_s, step_s, radius):
    """Only fully contained contexts receive control labels; transitions stay unknown.

    relation_annotations[name] = {known: [[start,end]], positive: [[start,end]]}.
    Annotation positive/known spans describe the input clip after any edits.
    """
    times = np.asarray(times)
    left = times + step_s / 2 - window_s / 2
    right = left + window_s
    interior = (left >= 0) & (right <= row["duration_s"] + 1e-6)
    targets = {name: np.full(len(times), -1, np.int64) for name in (*RELATIONS, *ACTIVITIES)}
    lag = np.full(len(times), -1, np.int64)
    kind = row.get("variant", {}).get("kind", "clean")
    variant = row.get("variant", {})
    if kind == "clean":
        lag[interior] = radius
        for name in RELATIONS:
            targets[name][interior] = 0
    elif kind in ("global_lag", "local_lag"):
        offset = float(variant["lag_s"]) / step_s
        if abs(offset - round(offset)) > 1e-5 or abs(offset) > radius:
            raise ValueError("Control offset must be on the configured lag grid")
        lag[interior] = radius + round(offset)
        for name in ("phoneme_viseme", "sequence", "source"):
            targets[name][interior] = 0
        if kind == "local_lag":
            start, end = variant["span"]
            outside = (right <= start) | (left >= end)
            within = (left >= start) & (right <= end)
            lag[interior & outside] = radius
            lag[~(outside | within)] = -1
            for target in targets.values():
                target[~(outside | within)] = -1
    elif kind == "sequence_swap":
        lag[interior] = 2 * radius + 1
        targets["sequence"][interior] = 1
    else:
        raise ValueError(f"Unknown variant: {kind}")
    for name, annotation in row.get("relation_annotations", {}).items():
        if name not in targets:
            raise ValueError(f"Unknown annotated relation: {name}")
        known = annotation.get("known", [])
        positive = annotation.get("positive", [])
        for a, b in known + positive:
            if not (0 <= a < b <= row["duration_s"] + 1e-6):
                raise ValueError("Annotation interval outside clip")
        for a, b in positive:
            if not any(x <= a and b <= y for x, y in known):
                raise ValueError("Positive annotations must be inside known regions")
        targets[name][:] = -1
        center = times + step_s / 2
        for a, b in known:
            targets[name][(center >= a) & (center < b)] = 0
        for a, b in positive:
            targets[name][(center >= a) & (center < b)] = 1
    targets["lag_class"] = lag
    return targets
