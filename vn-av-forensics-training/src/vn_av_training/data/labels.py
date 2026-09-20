"""Two-head partial supervision on rendered media; no edits during training."""

import numpy as np

from vn_av_training.contract import mismatch_annotation


def labels_for(row, times, window_s, step_s, radius):
    times = np.asarray(times)
    left = times + step_s / 2 - window_s / 2
    right = left + window_s
    interior = (left >= 0) & (right <= row["duration_s"] + 1e-6)
    lag = np.full(len(times), -1, np.int64)
    mismatch = np.full(len(times), -1, np.int64)
    if row.get("variant", {"kind": "clean"}) != {"kind": "clean"}:
        raise ValueError("Virtual controls are no longer supported; render or migrate the dataset")
    kind = row.get("generation", {}).get("edit", {}).get("kind")
    for span in row.get("supervision", {}).get("timing", []):
        mask = interior & (left >= span["start"]) & (right <= span["end"] + 1e-6)
        if span.get("no_match"):
            lag[mask] = 2 * radius + 1
        else:
            offset = float(span["lag_s"]) / step_s
            if abs(offset - round(offset)) > 1e-5 or abs(offset) > radius:
                raise ValueError("Timing labels differ from configured grid")
            lag[mask] = radius + round(offset)
            # Shifted clean speech is a hard negative for residual mismatch.
            if kind in {"global_lag", "local_lag"}:
                mismatch[mask] = 0
    annotation = mismatch_annotation(row.get("relation_annotations", {}), row["duration_s"])
    center = times + step_s / 2
    for a, b in annotation["known"]:
        mismatch[(center >= a) & (center < b) & interior] = 0
    for a, b in annotation["positive"]:
        mismatch[(center >= a) & (center < b) & interior] = 1
    return {"lag_class": lag, "lip_audio_mismatch": mismatch}
