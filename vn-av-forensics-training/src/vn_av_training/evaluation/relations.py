"""Held-out relation metrics. Test labels never select thresholds."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from vn_av_training.common.runtime import write_json
from vn_av_training.data.manifest import read_manifest, validate
from vn_av_training.serving.relations import aggregate_relations
from vn_av_training.training.relations import load_relations, load_sample, validate_source


def spans(mask, times, step):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [
        (float(times[a]), float(times[b - 1] + step))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def event_counts(predicted, truth, threshold=0.5):
    candidates = []
    for i, (a, b) in enumerate(predicted):
        for j, (x, y) in enumerate(truth):
            intersection = max(0, min(b, y) - max(a, x))
            iou = intersection / (b - a + y - x - intersection)
            if iou >= threshold:
                candidates.append((iou, i, j))
    used_pred, used_true = set(), set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_pred and j not in used_true:
            used_pred.add(i)
            used_true.add(j)
    return np.array([len(used_pred), len(predicted) - len(used_pred), len(truth) - len(used_true)])


@torch.no_grad()
def evaluate_relations(cfg, split="test"):
    model, state = load_relations(cfg["checkpoint"], cfg.get("device", "cpu"))
    rows = read_manifest(cfg["manifest"])
    validate(rows, check_files=False)
    rows = [row for row in rows if row["split"] == split]
    if not rows:
        raise ValueError(f"No {split} rows")
    buckets = {
        name: {"labels": [], "scores": [], "events": np.zeros(3, int), "known": 0, "assessed": 0}
        for name in state["trained_heads"]
    }
    lag_errors, results = [], []
    for row in rows:
        batch, targets, meta, times = load_sample(
            row, cfg["cache"], model.config["radius"], cfg.get("device", "cpu")
        )
        if meta["feature_signature"] != state["feature_signature"]:
            raise ValueError("Evaluation cache differs from checkpoint encoder")
        validate_source(row, meta)
        output = model(batch, use_lag_alignment="timing" in state["active_heads"])
        report = aggregate_relations(
            output,
            times,
            meta["step_s"],
            state["thresholds"],
            state["trained_heads"],
            state.get("lag_tolerance_steps", 0),
        )
        results.append({"sample_id": row["sample_id"], **report})
        for name, bucket in buckets.items():
            if name == "timing":
                target = targets["lag_class"][0].cpu().numpy()
                known = (target >= 0) & (target <= 2 * model.config["radius"])
                labels = (target != model.config["radius"]).astype(int)
                lag_mask = known & output["lag_valid"][0].cpu().numpy()
                error = (
                    np.abs(output["lag_steps"][0].cpu().numpy() - (target - model.config["radius"]))
                    * meta["step_s"]
                    * 1000
                )
                lag_errors.extend(error[lag_mask].tolist())
            else:
                labels = targets[name][0].cpu().numpy()
                known = labels >= 0
            values = [window["relation_scores"].get(name) for window in report["window_scores"]]
            measured = np.asarray([value is not None for value in values])
            score = np.asarray([value if value is not None else 0 for value in values])
            mask = known & measured
            bucket["known"] += int(known.sum())
            bucket["assessed"] += int(mask.sum())
            bucket["labels"].extend(labels[mask].tolist())
            bucket["scores"].extend(score[mask].tolist())
            bucket["events"] += event_counts(
                spans(mask & (score >= state["thresholds"][name]), times, meta["step_s"]),
                spans(known & (labels == 1), times, meta["step_s"]),
            )
    summary = {}
    for name, bucket in buckets.items():
        labels, scores = np.asarray(bucket["labels"]), np.asarray(bucket["scores"])
        if len(labels):
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, scores >= state["thresholds"][name], average="binary", zero_division=0
            )
            negatives = labels == 0
            false_alarm = (
                float(np.mean(scores[negatives] >= state["thresholds"][name]))
                if negatives.any()
                else None
            )
        else:
            precision = recall = f1 = false_alarm = None
        tp, fp, fn = bucket["events"].tolist()
        summary[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": float(average_precision_score(labels, scores))
            if set(labels.tolist()) == {0, 1}
            else None,
            "false_alarm_rate": false_alarm,
            "coverage": bucket["assessed"] / max(1, bucket["known"]),
            "event_f1_at_iou_0_5": 2 * tp / max(1, 2 * tp + fp + fn),
        }
    report = {
        "split": split,
        "samples": len(rows),
        "relations": summary,
        "lag_mae_ms": float(np.mean(lag_errors)) if lag_errors else None,
        "identity_status": "not_assessed",
        "label_note": "Generated sequence swaps are proxy labels; audit before research claims",
    }
    out = Path(cfg["output"]) / ("evaluation-" + split)
    write_json(out / "metrics.json", report)
    write_json(out / "predictions.json", results)
    return report
