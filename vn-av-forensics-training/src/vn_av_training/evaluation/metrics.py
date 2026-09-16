"""Clip and temporal evaluation; thresholds learned exclusively from validation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from vn_av_training.common.runtime import write_json
from vn_av_training.data.dataset import FeatureDataset, collate, to_device
from vn_av_training.data.manifest import connected_groups


def binary_metrics(labels, scores, threshold=0.5):
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(y):
        return {"n": 0, "auc": None, "ap": None, "f1": None, "fpr": None}
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "n": len(y),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "ap": float(average_precision_score(y, p)) if y.sum() else None,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "fpr": fp / (fp + tn) if fp + tn else None,
        "accuracy": float((pred == y).mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_for(labels, scores):
    if len(set(labels)) != 2:
        raise ValueError("Threshold calibration needs both classes in validation")
    candidates = np.unique(np.r_[0.0, np.quantile(scores, np.linspace(0, 1, 101)), 1.0])
    return float(max(candidates, key=lambda t: (binary_metrics(labels, scores, t)["f1"], t)))


def segments(times, scores, valid, threshold, step=0.04):
    spans, start, peak, previous = [], None, 0.0, None
    for t, p, ok in zip(times, scores, valid):
        active = bool(ok and p >= threshold)
        if start is not None and (not active or t - previous > 1.5 * step):
            spans.append(
                {"start_s": float(start), "end_s": float(previous + step), "score": float(peak)}
            )
            start = None
        if active:
            if start is None:
                start, peak = t, p
            peak = max(peak, p)
        previous = t
    if start is not None:
        spans.append(
            {"start_s": float(start), "end_s": float(previous + step), "score": float(peak)}
        )
    return spans


def iou(a, b):
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return intersection / max(1e-12, max(a[1], b[1]) - min(a[0], b[0]))


def temporal_ap(records, head, threshold=0.5):
    targets, preds = {}, []
    for r in records:
        gold = r.get(head + "_intervals")
        if gold is None:
            continue
        targets[r["sample_id"]] = gold
        for p in r.get(head + "_segments", []):
            preds.append((p["score"], r["sample_id"], [p["start_s"], p["end_s"]]))
    total = sum(map(len, targets.values()))
    if total == 0:
        return None
    used, tp, fp = defaultdict(set), [], []
    for _, sid, span in sorted(preds, reverse=True):
        matches = [(iou(span, g), j) for j, g in enumerate(targets[sid]) if j not in used[sid]]
        best, j = max(matches, default=(0.0, -1))
        hit = best >= threshold
        if hit:
            used[sid].add(j)
        tp.append(int(hit))
        fp.append(int(not hit))
    if not tp:
        return 0.0
    precision = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    recall = np.cumsum(tp) / total
    # Interpolated precision envelope, including missed targets.
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[0.0, precision, 0.0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ix = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[ix + 1] - mrec[ix]) * mpre[ix + 1]).sum())


@torch.inference_mode()
def predict(model, dataset, device="cpu", batch_size=4, thresholds=None, heads=None):
    thresholds = thresholds or {"clip": 0.5, "forgery": 0.5, "mismatch": 0.5}
    heads = heads or {"clip": True, "forgery": False, "mismatch": False}
    model.eval()
    records = []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate):
        output = model(to_device(batch, device))
        for i, row in enumerate(batch["rows"]):
            n = int(batch["padding_valid"][i].sum())
            valid = output["valid"][i, :n].cpu().numpy()
            t = batch["times_s"][i, :n].numpy()
            record = {
                k: row.get(k)
                for k in (
                    "sample_id",
                    "source_id",
                    "speaker_id",
                    "speaker_ids",
                    "parent_ids",
                    "global_speaker_ids",
                    "canonical_source_id",
                    "program_id",
                    "episode_id",
                    "group_id",
                    "compression_parent_id",
                    "compression_crf",
                    "dataset",
                    "split",
                    "generator",
                    "clip_label",
                    "forgery_intervals",
                    "mismatch_intervals",
                )
            }
            record.update(
                clip_score=float(output["clip"][i].sigmoid()) if valid.any() else None,
                coverage=float(valid.mean()),
                times_s=t.tolist(),
                valid=valid.tolist(),
            )
            for head in ("forgery", "mismatch"):
                if heads.get(head) and output[head] is not None:
                    p = output[head][i, :n].sigmoid().cpu().numpy()
                    record[head + "_scores"] = p.tolist()
                    record[head + "_segments"] = segments(t, p, valid, thresholds.get(head, 0.5))
            records.append(record)
    return records


def summarize(records, thresholds, bootstrap=0, seed=42):
    eligible = [
        r for r in records if r.get("clip_label") is not None and r.get("clip_score") is not None
    ]

    def metric(rs):
        return binary_metrics(
            [r["clip_label"] for r in rs], [r["clip_score"] for r in rs], thresholds["clip"]
        )

    report = {
        "clip": metric(eligible),
        "total": len(records),
        "evaluated": len(eligible),
        "mean_coverage": float(np.mean([r["coverage"] for r in records])) if records else 0.0,
        "thresholds": thresholds,
        "by_dataset": {},
        "by_generator": {},
        "by_compression_crf": {},
    }
    for key in ("dataset", "generator", "compression_crf"):
        for value in sorted({str(r.get(key)) for r in eligible}):
            report["by_" + key][value] = metric([r for r in eligible if str(r.get(key)) == value])
    for head in ("forgery", "mismatch"):
        y, p = [], []
        from vn_av_training.data.dataset import interval_targets

        for r in records:
            if head + "_scores" not in r:
                continue
            target, known = interval_targets(np.asarray(r["times_s"]), r.get(head + "_intervals"))
            mask = known & np.asarray(r["valid"])
            y.extend(target[mask].tolist())
            p.extend(np.asarray(r[head + "_scores"])[mask].tolist())
        if p:
            report[head] = {
                "window": binary_metrics(y, p, thresholds.get(head, 0.5)),
                "segment_ap": {str(t): temporal_ap(records, head, t) for t in (0.3, 0.5, 0.75)},
            }
    if bootstrap and eligible:
        eligible_ids = {r["sample_id"] for r in eligible}
        # Keep unlabeled/unusable bridge rows when resolving provenance connections.
        groups = [
            [r for r in group if r["sample_id"] in eligible_ids]
            for group in connected_groups(records)
        ]
        groups = [group for group in groups if group]
        report["bootstrap_independent_groups"] = len(groups)
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(bootstrap):
            sample = [r for j in rng.integers(len(groups), size=len(groups)) for r in groups[j]]
            auc = metric(sample)["auc"]
            if auc is not None:
                values.append(auc)
        report["auc_group_bootstrap_95ci"] = (
            np.quantile(values, [0.025, 0.975]).tolist() if values else None
        )
    return report


def evaluate(
    checkpoint, manifest, cache, output, split="test", device="cpu", annotations=None, bootstrap=200
):
    from vn_av_training.data.manifest import read_manifest, validate
    from vn_av_training.training.trainer import load_checkpoint, validate_cache

    model, info = load_checkpoint(checkpoint, device)
    all_rows = read_manifest(manifest)
    validate(all_rows, check_files=False)
    rows = [r for r in all_rows if r["split"] == split]
    if not rows:
        raise ValueError(f"No {split} rows")
    signature = validate_cache(rows, cache)
    if signature != info["feature_signature"]:
        raise ValueError("Evaluation encoder/preprocessing differs from training")
    if info["model_config"]["phoneme"]:
        if not annotations:
            raise ValueError("P3 evaluation requires annotations matching its training mode")
        from vn_av_training.common.runtime import read_json

        events = read_json(annotations)
        if info.get("phoneme_mode") == "automatic" and any(
            e.get("source") != "automatic_mfa" for r in rows for e in events.get(r["sample_id"], [])
        ):
            raise ValueError("Manual annotations cannot be used to evaluate automatic P3")
        if info.get("phoneme_mode") == "automatic" and any(
            e.get("pipeline_signature") != info.get("phoneme_signature")
            for r in rows
            for e in events.get(r["sample_id"], [])
        ):
            raise ValueError("Evaluation alignment assets differ from training")
    records = predict(
        model,
        FeatureDataset(rows, cache, annotations),
        device,
        thresholds=info["thresholds"],
        heads=info["heads"],
    )
    metrics = summarize(records, info["thresholds"], bootstrap)
    metrics.update(
        split=split, checkpoint=str(checkpoint), training_manifest_hash=info["manifest_hash"]
    )
    write_json(Path(output) / "predictions.json", records)
    write_json(Path(output) / "metrics.json", metrics)
    return metrics
