"""Cached-feature batches, temporal targets and padding masks."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vn_av_training.common.runtime import read_json


def interval_targets(times, intervals):
    if intervals is None:
        return np.zeros(len(times), np.float32), np.zeros(len(times), bool)
    y = np.zeros(len(times), np.float32)
    for start, end in intervals:
        y[(times >= start) & (times < end)] = 1
    return y, np.ones(len(times), bool)


class FeatureDataset(Dataset):
    def __init__(self, rows, cache, annotation_path=None):
        self.rows, self.cache = rows, Path(cache)
        self.annotations = read_json(annotation_path) if annotation_path else {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with np.load(self.cache / (r["sample_id"] + ".npz"), allow_pickle=False) as z:
            x = {
                k: z[k].copy()
                for k in ("audio", "visual", "times_s", "audio_valid", "visual_valid")
            }
            if "mouth_open" in z:
                x["mouth_open"] = z["mouth_open"].copy()
        t = x["times_s"]
        if (
            not len(t)
            or np.any(np.diff(t) <= 0)
            or not all(np.isfinite(x[k]).all() for k in ("audio", "visual", "times_s"))
        ):
            raise ValueError(f"Invalid cached feature: {r['sample_id']}")
        for name, key in [("forgery", "forgery_intervals"), ("mismatch", "mismatch_intervals")]:
            x[name], x[name + "_known"] = interval_targets(t, r.get(key))
        from vn_av_training.features.phonemes import event_features

        x["phoneme"], x["phoneme_valid"] = event_features(
            t, x.get("mouth_open"), self.annotations.get(r["sample_id"], [])
        )
        x["clip_label"] = float(r.get("clip_label") or 0)
        x["clip_known"] = r.get("clip_label") is not None
        x["row"] = r
        return x


def collate(samples):
    n = max(len(s["times_s"]) for s in samples)
    result = {"rows": [s["row"] for s in samples]}
    keys = [
        "audio",
        "visual",
        "times_s",
        "audio_valid",
        "visual_valid",
        "forgery",
        "forgery_known",
        "mismatch",
        "mismatch_known",
        "phoneme",
        "phoneme_valid",
    ]
    for key in keys:
        tensors = []
        for s in samples:
            v = s[key]
            pad = [(0, n - len(v)), *[(0, 0)] * (v.ndim - 1)]
            tensors.append(torch.from_numpy(np.pad(v, pad)))
        result[key] = torch.stack(tensors)
    result["padding_valid"] = (
        torch.arange(n)[None, :] < torch.tensor([len(s["times_s"]) for s in samples])[:, None]
    )
    for key in ("clip_label", "clip_known"):
        result[key] = torch.tensor([s[key] for s in samples])
    return result


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
