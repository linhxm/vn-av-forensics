"""Validation-only operating points and explicit deployment eligibility."""

import numpy as np
from sklearn.metrics import roc_auc_score


def calibrate(labels, scores, policy=None, threshold=None):
    policy = policy or {}
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        return {
            "status": "uncalibrated",
            "reason": "need both classes and finite validation scores",
        }
    negatives = int((labels == 0).sum())
    positives = int((labels == 1).sum())
    candidates = np.unique(np.r_[0.5, scores, np.nextafter(scores, 1)]).clip(1e-6, 1 - 1e-6)
    if threshold is not None:
        candidates = [float(threshold)]
    results = []
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int((predicted & (labels == 1)).sum())
        fp = int((predicted & (labels == 0)).sum())
        recall = tp / positives
        precision = tp / max(1, tp + fp)
        far = fp / negatives
        results.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "false_alarm_rate": far,
                "f1": 2 * tp / max(1, 2 * tp + fp + positives - tp),
            }
        )
    feasible = [
        r for r in results if r["false_alarm_rate"] <= policy.get("max_false_alarm_rate", 0.05)
    ]
    choice = max(
        feasible or results, key=lambda r: (r["f1"], -r["false_alarm_rate"], r["threshold"])
    )
    auc = float(roc_auc_score(labels, scores))
    reasons = []
    if min(negatives, positives) < policy.get("min_windows_per_class", 20):
        reasons.append("too few labeled validation windows")
    if choice["false_alarm_rate"] > policy.get("max_false_alarm_rate", 0.05):
        reasons.append("false alarms exceed limit")
    if choice["recall"] < policy.get("min_recall", 0.1):
        reasons.append("recall below minimum")
    if choice["precision"] < policy.get("min_precision", 0.5):
        reasons.append("precision below minimum")
    if auc < policy.get("min_auroc", 0.6):
        reasons.append("AUROC below minimum")
    return {
        **choice,
        "auroc": auc,
        "positive_windows": positives,
        "negative_windows": negatives,
        "status": "quality_failed" if reasons else "ready",
        "reasons": reasons,
    }


def head_status(state):
    from vn_av_training.models.relations import RELATIONS

    reports = state.get("validation_report", {})
    return {
        head: (
            reports[head]["status"]
            if head in reports
            else "uncalibrated"
            if head in state.get("active_heads", [])
            else "untrained"
        )
        for head in ("timing", *RELATIONS)
    }


def assessment_masks(output, state):
    """No reliable lag is not proof of mismatch; require validated unaligned behavior."""
    q = state.get("validation_report", {}).get("lip_audio_mismatch", {})
    return {
        **output,
        "mismatch_valid": output["valid"]
        & (output["lag_valid"] | bool(q.get("unaligned_ready", False))),
    }
