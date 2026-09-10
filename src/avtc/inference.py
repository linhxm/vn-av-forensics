"""Upload-video inference using the trained local checkpoint."""

from pathlib import Path

import pandas as pd
import torch

from .model import load_model
from .syncnet import (
    WEIGHT_HASHES,
    SyncNet,
    decode_crop,
    encode,
    intervals,
    normalize_source,
    prepare_track,
    write_json,
)


def analyze(video, checkpoint, output, assets="checkpoints/syncnet", device="cpu"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model, info = load_model(checkpoint, device)
    if info["backbone_sha256"] != WEIGHT_HASHES["data/syncnet_v2.model"]:
        raise ValueError("CNN backbone mismatch")
    extractor = SyncNet(assets, device)
    crop, meta = prepare_track(video, out / "preprocess", assets)
    frames, pcm = decode_crop(crop, assets)
    frames, pcm, trim, source_lag = normalize_source(frames, pcm, extractor)
    v, a, t, valid = encode(frames, pcm, extractor)
    with torch.inference_mode():
        score = torch.sigmoid(model(torch.tensor(v, device=device), torch.tensor(a, device=device)))
    r = model.radius
    t = t[r:-r] + meta["start_s"] + trim
    valid = valid[r:-r]
    scores = score.cpu().numpy()
    threshold = info["threshold"]
    spans = intervals(t, valid & (scores > threshold)) if valid.sum() >= 3 else []
    table = pd.DataFrame({"time_s": t, "score": scores, "valid": valid})
    result = {
        "status": "insufficient_evidence"
        if valid.sum() < 3
        else "inconsistency_detected"
        if spans
        else "no_clear_inconsistency",
        "intervals": spans,
        "threshold": threshold,
        "source_lag_corrected_ms": source_lag,
        "valid_windows": int(valid.sum()),
        "total_windows": len(valid),
        "notice": "Local audio-mouth inconsistency; not a deepfake/real verdict.",
    }
    write_json(out / "result.json", result)
    table.to_csv(out / "timeline.csv", index=False)
    return result, table
