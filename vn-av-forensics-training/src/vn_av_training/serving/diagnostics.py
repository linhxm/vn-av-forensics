"""Measured preprocessing evidence, distinct from model explanations or labels."""

from pathlib import Path

import numpy as np

from vn_av_training.common.runtime import read_json, sha
from vn_av_training.data.media import decode
from vn_av_training.training.quality import head_status


def checkpoint_evidence(state, checkpoint):
    status = head_status(state)
    metrics_path = Path(checkpoint).parent / "evaluation-test/metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else None
    checkpoint_hash = sha(checkpoint)
    if metrics and metrics.get("checkpoint_sha256") not in (None, checkpoint_hash):
        metrics = None
    # Test is reported, never used to choose thresholds.
    return {
        "active_heads": state.get("active_heads", []),
        "calibrated_heads": state.get("trained_heads", []),
        "deployable_heads": [name for name, value in status.items() if value == "ready"],
        "head_status": status,
        "validation_report": state.get("validation_report", {}),
        "test_report": metrics,
        "test_report_provenance": (
            "checkpoint SHA256 verified"
            if metrics.get("checkpoint_sha256")
            else "historical adjacent report without checkpoint hash; association not verified"
        )
        if metrics
        else None,
        "label_coverage": state.get("label_coverage", {}),
        "checkpoint_sha256": checkpoint_hash,
    }


def preprocessing_evidence(video, folder, cfg):
    import cv2

    from vn_av_training.features.face import FaceTracker

    decoded = decode(
        video,
        max_duration=cfg.get("max_duration", 60),
        max_side=cfg.get("max_side", 640),
        sample_rate=48000,
    )
    frames = decoded["frames"]
    folder = Path(folder) / "evidence"
    folder.mkdir(exist_ok=True)
    tracker = FaceTracker(cfg["face_model"]) if cfg.get("face_model") else None
    samples = []
    for i, index in enumerate(np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)):
        frame = frames[index]
        target = folder / f"frame-{i}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Evidence image encoding failed")
        target.write_bytes(encoded.tobytes())
        if tracker:
            tracker.previous = None  # Sparse previews are not adjacent tracking frames.
        geometry = tracker.inspect(frame) if tracker else None
        samples.append(
            {
                "time_s": float(index / 25),
                "image": f"evidence/{target.name}",
                "face_check": geometry,
            }
        )
    pcm = decoded["pcm"]
    bins = []
    for start in range(0, len(pcm), 9600):
        chunk = pcm[start : start + 9600]
        bins.append({"start": start / 48000, "rms": float(np.sqrt(np.mean(chunk**2)))})
    return {
        "sample_rate_hz": 48000,
        "video_grid_fps": 25,
        "decoded_frames": len(frames),
        "origin_s": decoded["origin_s"],
        "audio_rms": bins,
        "frames": samples,
        "note": "RMS is signal energy; sampled face geometry is not a phoneme or active-speaker decision.",
    }
