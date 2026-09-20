"""Read materialized relation datasets without generating or editing media."""

from pathlib import Path

import numpy as np

from vn_av_training.common.runtime import read_json, sha, write_json
from vn_av_training.contract import bundle_path, validate_annotations
from vn_av_training.data.labels import labels_for
from vn_av_training.data.manifest import read_manifest, validate, write_manifest
from vn_av_training.data.media import probe


def validate_generated(dataset, manifest="manifest.jsonl", cfg=None):
    root = Path(dataset).resolve()
    path = bundle_path(root, manifest)
    info = read_json(root / "dataset_info.json")
    if info.get("schema_version") not in {"vn-av-relations-v1", "vn-av-relations-v2"}:
        raise ValueError("Expected materialized relation dataset")
    receipt = info if manifest == "manifest.jsonl" else read_json(path.with_suffix(".info.json"))
    if receipt.get("label_schema") != "timing-lip-audio-v1":
        raise ValueError(
            "Migrate the reviewed manifest to timing-lip-audio-v1 in the generation project"
        )
    if receipt.get("manifest_sha256") != sha(path):
        raise ValueError("Generated manifest hash mismatch")
    if cfg and (
        info["step_s"] != cfg["encoder"]["step_s"] or info["radius"] != cfg["model"]["radius"]
    ):
        raise ValueError("Generated timing grid differs from model config")
    rows = read_manifest(path)
    if len(rows) != receipt.get("samples"):
        raise ValueError("Generated sample count mismatch")
    for row in rows:
        if not row.get("materialized") or row.get("variant") != {"kind": "clean"}:
            raise ValueError("Training consumes rendered media only; do not apply edits twice")
        if row.get("generation", {}).get("edit", {}).get("kind") == "source_swap":
            raise ValueError("source_swap is outside the two-head task; migrate the dataset")
        if set(row.get("relation_annotations", {})) - {"lip_audio_mismatch"}:
            raise ValueError("Migrate legacy relation annotations before training")
        video = bundle_path(root, row["video"])
        if sha(video) != row["sha256"]:
            raise ValueError("Generated video hash mismatch")
        if abs(probe(video)["duration_s"] - row["duration_s"]) > 0.2:
            raise ValueError("Media duration differs")
        validate_annotations(row.get("relation_annotations", {}), row["duration_s"])
        for span in row.get("supervision", {}).get("timing", []):
            if not 0 <= span["start"] < span["end"] <= row["duration_s"] + 0.001:
                raise ValueError("Timing interval outside clip")
            if not span.get("no_match") and not np.isfinite(span.get("lag_s", np.nan)):
                raise ValueError("Invalid timing offset")
        row["video"] = str(video)
    report = validate(rows, hash_files=True)
    return rows, {**report, "manifest_sha256": sha(path), "dataset_id": info["dataset_id"]}


def import_generated(dataset, output, manifest="manifest.jsonl", cfg=None):
    rows, report = validate_generated(dataset, manifest, cfg)
    for row in rows:
        video = Path(row["video"])
        if video.is_relative_to(Path.cwd()):
            row["video"] = video.relative_to(Path.cwd()).as_posix()
    path = Path(output)
    if path.exists() and read_manifest(path) != rows:
        raise FileExistsError("Manifest differs; use new dataset/run paths")
    write_manifest(path, rows)
    write_json(path.with_suffix(".report.json"), report)
    return report


def supervision_report(cfg, enforce=False):
    rows = read_manifest(cfg["manifest"])
    validate(rows, check_files=False)
    names = ("timing", "lip_audio_mismatch")
    report = {
        split: {name: {"negative": 0, "positive": 0} for name in names}
        for split in ("train", "validation", "test")
    }
    ecfg = cfg["encoder"]
    radius = cfg["model"]["radius"]
    for row in rows:
        times = np.arange(0, row["duration_s"], ecfg["step_s"])
        targets = labels_for(row, times, ecfg["window_s"], ecfg["step_s"], radius)
        for head in names:
            y = targets["lag_class" if head == "timing" else head]
            if head == "timing":
                neg = y == radius
                pos = (y >= 0) & (y <= 2 * radius) & (y != radius)
            else:
                neg = y == 0
                pos = y == 1
            report[row["split"]][head]["negative"] += int(neg.sum())
            report[row["split"]][head]["positive"] += int(pos.sum())
    missing = [
        head
        for head in cfg.get("training", {}).get("required_heads", [])
        if any(not all(report[split][head].values()) for split in ("train", "validation"))
    ]
    result = {
        "windows_before_visibility_masks": report,
        "missing_required_heads": missing,
        "note": "FATE/face visibility can reduce usable labels further",
    }
    if enforce and missing:
        raise ValueError(
            "Dataset not ready for full task; missing positive/negative labels: "
            + ", ".join(missing)
            + ". Review generated events or explicitly choose a limited pilot in training.required_heads."
        )
    return result
