"""Render reproducible AV controls; mechanical edits are not phoneme ground truth."""

import csv
import io
import math
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from vn_av_generation.common.runtime import (
    atomic_bytes,
    fingerprint,
    read_json,
    run,
    sha,
    write_json,
)
from vn_av_generation.contract import bundle_path, mismatch_annotation, validate_bundle
from vn_av_generation.data.manifest import (
    connected_groups,
    identity_list,
    read_manifest,
    write_manifest,
)
from vn_av_generation.data.media import decode, ffmpeg, probe

SCHEMA = "vn-av-relations-v2"
LABEL_SCHEMA = "timing-lip-audio-v1"
RELATIONS = ("lip_audio_mismatch",)


def csv_write(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode("utf-8-sig"))


def split_originals(rows, seed):
    groups = connected_groups([{**r, "sample_id": r["clip_id"]} for r in rows])
    if len(groups) < 3:
        raise ValueError("Need at least three independent source/speaker groups")
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    target = dict(train=len(rows) * 0.7, validation=len(rows) * 0.15, test=len(rows) * 0.15)
    counts, assigned = Counter(), Counter()
    result = []
    for i, group in enumerate(groups):
        empty = [s for s in target if not assigned[s]]
        choices = empty if len(groups) - i == len(empty) else list(target)
        choice = min(
            choices,
            key=lambda s: sum(
                (counts[t] + (len(group) if t == s else 0) - target[t]) ** 2 for t in target
            ),
        )
        counts[choice] += len(group)
        assigned[choice] += 1
        result.extend({**r, "split": choice} for r in group)
    return result


def span_for(duration, context):
    # Make the local positive event wider than the extraction window when possible.
    start, end = round(duration * 0.15 * 25) / 25, round(duration * 0.85 * 25) / 25
    return [start, end] if end - start >= context + 0.4 else None


def plan_dataset(dataset, output, cfg):
    root, output = Path(dataset).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Plan already exists; use a new plan version")
    rows, info = validate_bundle(root, probe=probe)
    rows = split_originals(rows, int(cfg.get("seed", 42)))
    techniques = cfg.get("techniques", [])
    allowed = {
        "global_lag",
        "local_lag",
        "sequence_swap",
        "motion_freeze",
        "content_splice",
    }
    if set(techniques) - allowed:
        raise ValueError("Unknown generation technique")
    shifts = cfg.get("shifts_s", [])
    for lag in shifts:
        steps = lag / cfg["step_s"]
        if (
            not math.isfinite(lag)
            or not lag
            or abs(steps - round(steps)) > 1e-6
            or abs(steps) > cfg["radius"]
        ):
            raise ValueError("Shifts must lie on the configured timing grid")
    jobs, skipped = [], []
    for row in rows:
        same = [
            r
            for r in rows
            if r["split"] == row["split"]
            and r["clip_id"] != row["clip_id"]
            and (
                r["source_id"] == row["source_id"]
                or (
                    identity_list(row.get("speaker_id"))
                    and row["speaker_id"] == r.get("speaker_id")
                )
            )
            and (
                r["source_id"] != row["source_id"]
                or r["source_end_s"] <= row["source_start_s"]
                or row["source_end_s"] <= r["source_start_s"]
            )
            and (
                not row.get("speaker_id")
                or not r.get("speaker_id")
                or row["speaker_id"] == r["speaker_id"]
            )
        ]
        variants = [{"kind": "clean"}]
        span = span_for(row["duration_s"], cfg["window_s"])
        for kind in techniques:
            if kind in {"global_lag", "local_lag"}:
                if kind == "local_lag" and span is None:
                    skipped.append(
                        {
                            "clip_id": row["clip_id"],
                            "kind": kind,
                            "reason": "event shorter than context window",
                        }
                    )
                    continue
                variants.extend(
                    {"kind": kind, "lag_s": lag, **({"span": span} if kind == "local_lag" else {})}
                    for lag in shifts
                )
            elif kind == "motion_freeze":
                if span:
                    variants.append({"kind": kind, "span": span})
            else:
                donors = same
                if not donors:
                    skipped.append(
                        {
                            "clip_id": row["clip_id"],
                            "kind": kind,
                            "reason": "no eligible non-overlapping donor in same split",
                        }
                    )
                    continue
                donor = sorted(donors, key=lambda r: r["clip_id"])[0]
                variant = {
                    "kind": kind,
                    "donor_id": donor["clip_id"],
                    "donor": donor["video"],
                    "donor_sha256": donor["sha256"],
                    "donor_source_id": donor["source_id"],
                    "donor_speaker_id": donor.get("speaker_id"),
                }
                if kind == "content_splice":
                    if not span:
                        continue
                    variant.update(span=span, donor_start_s=0)
                variants.append(variant)
        for variant in variants:
            for crf in cfg.get("crfs", [18]):
                if not isinstance(crf, int) or not 0 <= crf <= 51:
                    raise ValueError("Invalid CRF")
                key = fingerprint(
                    [
                        info["manifest_sha256"],
                        row["clip_id"],
                        variant,
                        crf,
                        int(cfg.get("seed", 42)),
                    ]
                )[:24]
                jobs.append(
                    {"sample_id": "av_" + key, "original": row, "edit": variant, "crf": crf}
                )
    plan = {
        "format": "vn-av-generation-plan-v2",
        "dataset_id": info["dataset_id"],
        "input_manifest_sha256": info["manifest_sha256"],
        "config": cfg,
        "jobs": jobs,
        "skipped": skipped,
        "original_splits": dict(Counter(r["split"] for r in rows)),
        "variant_counts": dict(Counter(j["edit"]["kind"] for j in jobs)),
    }
    write_json(output, plan)
    return {k: v for k, v in plan.items() if k not in {"jobs", "config"}}


def edit_media(decoded, edit, donor=None, sr=48000):
    frames = decoded["frames"].copy()
    pcm = decoded["pcm"].copy()
    duration = len(pcm) / sr
    kind = edit["kind"]
    if kind in {"global_lag", "local_lag"}:
        shift = round(edit["lag_s"] * sr)
        index = np.arange(len(pcm)) + shift
        valid = (index >= 0) & (index < len(pcm))
        changed = np.ones(len(pcm), bool)
        if kind == "local_lag":
            a, b = edit["span"]
            changed = (np.arange(len(pcm)) / sr >= a) & (np.arange(len(pcm)) / sr < b)
        pcm[changed] = np.where(valid, pcm[index.clip(0, len(pcm) - 1)], 0)[changed]
    elif kind in {"sequence_swap", "content_splice"}:
        if donor is None:
            raise ValueError("Donor required")
        a, b = edit.get("span", [0, duration])
        start, end = round(a * sr), min(len(pcm), round(b * sr))
        ds = round(edit.get("donor_start_s", 0) * sr)
        available = min(end - start, len(donor["pcm"]) - ds)
        if available <= 0:
            raise ValueError("Donor interval unavailable")
        pcm[start:end] = 0
        pcm[start : start + available] = donor["pcm"][ds : ds + available]
        # 10ms crossfade reduces splice clicks; uncertain boundary windows are not timing labels.
        if kind == "content_splice":
            width = min(round(0.01 * sr), available // 2)
            ramp = np.linspace(0, 1, width)
            pcm[start : start + width] = (
                decoded["pcm"][start : start + width] * (1 - ramp)
                + pcm[start : start + width] * ramp
            )
            stop = start + available
            pcm[stop - width : stop] = (
                pcm[stop - width : stop] * (1 - ramp) + decoded["pcm"][stop - width : stop] * ramp
            )
    elif kind == "motion_freeze":
        a, b = edit["span"]
        start, end = round(a * 25), min(len(frames), round(b * 25))
        frames[start:end] = frames[start]
    elif kind != "clean":
        raise ValueError(f"Unsupported edit: {kind}")
    return frames, pcm


def encode(frames, pcm, output, crf):
    import cv2
    from scipy.io import wavfile

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(tmp / "video.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 25, (w, h)
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot create lossless intermediate")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        wavfile.write(
            tmp / "audio.wav", 48000, np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
        )
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-i",
                tmp / "video.avi",
                "-i",
                tmp / "audio.wav",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "1",
                "-shortest",
                tmp / "result.mp4",
            ]
        )
        os.replace(tmp / "result.mp4", output)


def supervision(edit, duration, donor_duration=None):
    kind = edit["kind"]
    timing = []
    labels = {}
    pending = []

    def clean_labels(end):
        return {name: {"known": [[0, end]], "positive": []} for name in RELATIONS}

    if kind == "clean":
        timing = [{"start": 0, "end": duration, "lag_s": 0}]
        labels = clean_labels(duration)
    elif kind in {"global_lag", "local_lag"}:
        lag = edit["lag_s"]
        lo = max(0, -lag)
        hi = min(duration, duration - lag)
        if kind == "global_lag":
            timing = [{"start": lo, "end": hi, "lag_s": lag}]
        else:
            a, b = edit["span"]
            timing = [
                {"start": 0, "end": a, "lag_s": 0},
                {"start": max(a, lo), "end": min(b, hi), "lag_s": lag},
                {"start": b, "end": duration, "lag_s": 0},
            ]
        # Residual mismatch is negative only in fully supported timing contexts (labels_for).
    elif kind == "sequence_swap":
        end = min(duration, donor_duration)
        # A different audio source does not prove the absence of a timing match.
        # Keep timing unknown until independently annotated.
        head = "lip_audio_mismatch"
        pending = [
            {"head": head, "start": 0, "end": end, "reason": "synthetic proxy requires review"}
        ]
    elif kind == "motion_freeze":
        a, b = edit["span"]
        pending = [
            {
                "head": "lip_audio_mismatch",
                "start": a,
                "end": b,
                "reason": "verify audible speech and frozen mouth",
            }
        ]
        timing = [{"start": 0, "end": a, "lag_s": 0}, {"start": b, "end": duration, "lag_s": 0}]
    elif kind == "content_splice":
        a, b = edit["span"]
        b = min(b, a + donor_duration)
        pending = [
            {
                "head": "lip_audio_mismatch",
                "start": a,
                "end": b,
                "reason": "verify incompatibility with visible mouth motion after allowing plausible timing shift",
            }
        ]
    else:
        raise ValueError(f"Unsupported generator: {kind}")

    return {
        "timing": [r for r in timing if r["end"] > r["start"]],
        "relation_annotations": labels,
    }, pending


def render_dataset(dataset, plan_path, output):
    root = Path(dataset).resolve()
    output = Path(output).resolve()
    plan = read_json(plan_path)
    if plan.get("format") != "vn-av-generation-plan-v2":
        raise ValueError(
            "Create a new two-head generation plan; old plans are not rerendered implicitly"
        )
    originals, info = validate_bundle(root, probe=probe)
    if info["manifest_sha256"] != plan["input_manifest_sha256"]:
        raise ValueError("Input dataset differs from locked plan")
    byid = {
        r["clip_id"]: r for r in split_originals(originals, int(plan["config"].get("seed", 42)))
    }
    seen = set()
    for job in plan["jobs"]:
        sid, original, edit = job["sample_id"], job["original"], job["edit"]
        if not re.fullmatch(r"av_[0-9a-f]{24}", sid) or sid in seen:
            raise ValueError("Invalid or duplicate planned sample ID")
        seen.add(sid)
        if original != byid.get(original["clip_id"]):
            raise ValueError("Planned original or split differs from input dataset")
        if "donor" in edit:
            donor = byid.get(edit.get("donor_id"))
            if (
                donor is None
                or donor["split"] != original["split"]
                or any(
                    edit.get(key) != donor.get(field)
                    for key, field in (
                        ("donor", "video"),
                        ("donor_sha256", "sha256"),
                        ("donor_source_id", "source_id"),
                        ("donor_speaker_id", "speaker_id"),
                    )
                )
            ):
                raise ValueError("Planned donor differs from input dataset or crosses splits")
    signature = fingerprint({"plan": plan, "implementation": sha(__file__)})
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "generation.json"
    if not lock.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty and has no generation lock; use a new directory")
    if lock.exists() and read_json(lock)["signature"] != signature:
        raise ValueError("Run differs; use a new output")
    write_json(lock, {"signature": signature, "plan": plan, "schema_version": SCHEMA})
    rows = []
    audits = []
    for i, job in enumerate(plan["jobs"]):
        print(f"Render {i + 1}/{len(plan['jobs'])} {job['edit']['kind']}", flush=True)
        sid = job["sample_id"]
        target = output / "clips" / f"{sid}.mp4"
        record = output / "records" / f"{sid}.json"
        if record.exists() != target.exists():
            raise ValueError(f"Incomplete sample {sid}; move its orphan file out before retrying")
        if record.exists() and target.exists():
            saved = read_json(record)
            if saved["row"]["sha256"] != sha(target):
                raise ValueError(f"Generated clip changed: {sid}")
            rows.append(saved["row"])
            audits.extend(saved["audits"])
            continue
        original = job["original"]
        edit = job["edit"]
        source = (root / original["video"]).resolve()
        decoded = decode(source, max_side=plan["config"].get("max_side", 640), sample_rate=48000)
        donor = (
            decode(
                root / edit["donor"],
                max_side=plan["config"].get("max_side", 640),
                sample_rate=48000,
            )
            if "donor" in edit
            else None
        )
        if donor is not None and sha(root / edit["donor"]) != edit["donor_sha256"]:
            raise ValueError("Donor hash mismatch")
        frames, pcm = edit_media(decoded, edit, donor)
        encode(frames, pcm, target, job["crf"])
        duration = len(frames) / 25
        if abs(probe(target)["duration_s"] - duration) > 0.15:
            raise ValueError("Rendered duration differs")
        labels, pending = supervision(
            edit, duration, len(donor["pcm"]) / 48000 if donor is not None else None
        )
        if edit["kind"] == "clean":
            labels["relation_annotations"] = {
                "lip_audio_mismatch": mismatch_annotation(
                    original.get("relation_annotations", {}), duration, assume_clean=True
                )
            }
        row = {
            **{
                key: original[key]
                for key in (
                    "url",
                    "source_sha256",
                    "canonical_source_id",
                    "program_id",
                    "episode_id",
                    "global_speaker_ids",
                )
                if original.get(key)
            },
            "sample_id": sid,
            "video": f"clips/{sid}.mp4",
            "sha256": sha(target),
            "duration_s": duration,
            "dataset": info["dataset_id"],
            "source_id": original["source_id"],
            "speaker_id": original.get("speaker_id"),
            "parent_ids": list(
                dict.fromkeys(
                    [original["source_id"], edit.get("donor_source_id", original["source_id"])]
                )
            ),
            "speaker_ids": list(
                filter(
                    None, dict.fromkeys([original.get("speaker_id"), edit.get("donor_speaker_id")])
                )
            ),
            "split": original["split"],
            "variant": {"kind": "clean"},
            "materialized": True,
            "generation": {
                "edit": edit,
                "crf": job["crf"],
                "source_clip_id": original["clip_id"],
                "source_sha256": original["sha256"],
            },
            "source_start_s": original["source_start_s"],
            "source_end_s": original["source_start_s"] + duration,
            "supervision": labels,
            "relation_annotations": labels["relation_annotations"],
        }
        audit = [
            {
                "sample_id": sid,
                "file_path": row["video"],
                "head": p["head"],
                "start": p["start"],
                "end": p["end"],
                "decision": "pending",
                "reason": p["reason"],
            }
            for p in pending
        ]
        write_json(record, {"row": row, "audits": audit})
        rows.append(row)
        audits.extend(audit)
    write_manifest(output / "manifest.jsonl", rows)
    if audits and not (output / "review.csv").exists():
        csv_write(output / "review.csv", audits)
    write_json(
        output / "dataset_info.json",
        {
            "schema_version": SCHEMA,
            "label_schema": LABEL_SCHEMA,
            "dataset_id": info["dataset_id"] + "-generated",
            "manifest_sha256": sha(output / "manifest.jsonl"),
            "samples": len(rows),
            "generation_signature": signature,
            "step_s": plan["config"]["step_s"],
            "radius": plan["config"]["radius"],
            "window_s": plan["config"]["window_s"],
        },
    )
    result = {
        "samples": len(rows),
        "pending_annotations": len(audits),
        "splits": dict(Counter(r["split"] for r in rows)),
        "techniques": dict(Counter(r["generation"]["edit"]["kind"] for r in rows)),
    }
    write_json(output / "summary.json", result)
    return result


def finalize_labels(dataset, review, output):
    """Publish labels as a new manifest version without copying/re-encoding media."""
    root = Path(dataset).resolve()
    output = Path(output).resolve()
    if output.parent != root or output.exists():
        raise ValueError("Write a new manifest filename directly inside the generated dataset")
    info = read_json(root / "dataset_info.json")
    if info.get("schema_version") != SCHEMA or info.get("manifest_sha256") != sha(
        root / "manifest.jsonl"
    ):
        raise ValueError(
            "Expected an unchanged two-head dataset; use migrate for old reviewed manifests"
        )
    rows = read_manifest(root / "manifest.jsonl")
    byid = {r["sample_id"]: r for r in rows}
    with Path(review).open(encoding="utf-8-sig", newline="") as stream:
        decisions = list(csv.DictReader(stream))
    seen = set()
    for item in decisions:
        key = (item["sample_id"], item["head"])
        if key in seen:
            raise ValueError("Duplicate head review for sample")
        seen.add(key)
        row = byid[item["sample_id"]]
        if sha(root / row["video"]) != row["sha256"]:
            raise ValueError("Reviewed video changed")
        if item["decision"] == "pending":
            continue
        if item["decision"] not in {"positive", "negative", "uncertain"}:
            raise ValueError("Invalid review decision")
        if item["head"] not in RELATIONS:
            raise ValueError("Invalid reviewed head")
        if item["decision"] == "uncertain":
            continue
        a, b = float(item["start"]), float(item["end"])
        if not 0 <= a < b <= row["duration_s"] + 0.01:
            raise ValueError("Review interval outside clip")
        row["relation_annotations"][item["head"]] = {
            "known": [[a, b]],
            "positive": [[a, b]] if item["decision"] == "positive" else [],
        }
        row["supervision"]["relation_annotations"] = row["relation_annotations"]
        row["annotation_quality"] = "human_reviewed_generated_event"
    write_manifest(output, rows)
    report = {
        "schema_version": SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "manifest_sha256": sha(output),
        "review_sha256": sha(review),
        "samples": len(rows),
    }
    write_json(output.with_suffix(".info.json"), report)
    return report


def migrate_labels(dataset, manifest, output):
    """Publish a new two-head manifest; never edit original media or labels."""
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output.parent != root or output.exists() or output.with_suffix(".info.json").exists():
        raise ValueError("Use a new manifest filename inside the existing dataset")
    path = bundle_path(root, manifest)
    info = read_json(root / "dataset_info.json")
    receipt = info if manifest == "manifest.jsonl" else read_json(path.with_suffix(".info.json"))
    if receipt.get("manifest_sha256") != sha(path):
        raise ValueError("Input manifest hash mismatch")
    rows = read_manifest(path)
    if len(rows) != receipt.get("samples"):
        raise ValueError("Input manifest count mismatch")
    converted, excluded = [], 0
    for row in rows:
        if not row.get("materialized") or row.get("variant") != {"kind": "clean"}:
            raise ValueError("Migration requires rendered media")
        if sha(bundle_path(root, row["video"])) != row["sha256"]:
            raise ValueError("Input video hash mismatch")
        kind = row.get("generation", {}).get("edit", {}).get("kind")
        if kind == "source_swap":
            excluded += 1
            continue
        old = row.get("relation_annotations", {})
        row["legacy_relation_annotations"] = old
        row["relation_annotations"] = {
            "lip_audio_mismatch": mismatch_annotation(
                old, row["duration_s"], assume_clean=kind == "clean"
            )
        }
        row.setdefault("supervision", {})["relation_annotations"] = row["relation_annotations"]
        # Legacy synthetic no-match labels on replacements are not measured offsets.
        if kind in {"sequence_swap", "content_splice", "motion_freeze"}:
            row["supervision"]["timing"] = [
                s for s in row["supervision"].get("timing", []) if not s.get("no_match")
            ]
        converted.append(row)
    if not converted:
        raise ValueError("No in-scope samples remain")
    write_manifest(output, converted)
    result = {
        "schema_version": SCHEMA,
        "label_schema": LABEL_SCHEMA,
        "manifest_sha256": sha(output),
        "input_manifest_sha256": sha(path),
        "samples": len(converted),
        "excluded_source_swap": excluded,
    }
    write_json(output.with_suffix(".info.json"), result)
    return result
