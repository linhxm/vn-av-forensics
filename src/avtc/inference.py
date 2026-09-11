"""Upload-video inference using the trained local checkpoint."""

from pathlib import Path

import numpy as np
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


def analyze(video, checkpoint, output, assets="checkpoints/syncnet", device="cpu", on_progress=None):
    """Analyze one video; ``on_progress(stage, message)`` is optional UI telemetry."""

    def report(stage, message):
        if on_progress is not None:
            on_progress(stage, message)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    report("checkpoint", "Đang nạp local checkpoint.")
    model, info = load_model(checkpoint, device)
    if info["backbone_sha256"] != WEIGHT_HASHES["data/syncnet_v2.model"]:
        raise ValueError("CNN backbone mismatch")
    report("face_tracking", "Đang tìm và theo dõi face track.")
    extractor = SyncNet(assets, device)
    crop, meta = prepare_track(video, out / "preprocess", assets)
    report("decode", "Đang decode face frames và PCM audio.")
    frames, pcm = decode_crop(crop, assets)
    report("global_lag", "Đang ước lượng và bù global lag.")
    frames, pcm, trim, source_lag = normalize_source(frames, pcm, extractor)
    report("feature_extraction", "Đang tạo frozen SyncNet features.")
    v, a, t, valid = encode(frames, pcm, extractor)
    energy = np.array(
        [np.sqrt(np.mean(pcm[i * 640 : (i + 5) * 640].astype(float) ** 2)) for i in range(len(t))]
    )
    report("local_scoring", "Đang tính local offset pattern và score timeline.")
    with torch.inference_mode():
        score = torch.sigmoid(model(torch.tensor(v, device=device), torch.tensor(a, device=device)))
    r = model.radius
    t = t[r:-r] + meta["start_s"] + trim
    valid = valid[r:-r]
    energy = energy[r:-r]
    scores = score.cpu().numpy()
    threshold = info["threshold"]
    spans = intervals(t, valid & (scores > threshold)) if valid.sum() >= 3 else []
    table = pd.DataFrame({"time_s": t, "score": scores, "valid": valid, "audio_energy": energy})
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
    report("output", "Đang lưu timeline và evidence artifacts.")
    write_json(out / "result.json", result)
    table.to_csv(out / "timeline.csv", index=False)
    return result, table
