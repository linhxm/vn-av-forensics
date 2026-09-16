import numpy as np
import pytest

from vn_av_training.common.runtime import save_npz, sha, write_json
from vn_av_training.data.manifest import write_manifest


@pytest.fixture
def feature_project(tmp_path):
    """Synthetic activations exercise training plumbing, never detector accuracy claims."""
    cache = tmp_path / "cache"
    rows = []
    rng = np.random.default_rng(123)
    for split, count in [("train", 8), ("validation", 4), ("test", 4)]:
        for i in range(count):
            sid = f"{split}_{i}"
            label = i % 2
            n = 32 + i
            a = rng.normal(size=(n, 6)).astype(np.float32)
            v = rng.normal(size=(n, 6)).astype(np.float32)
            a[:, 0] += label * 3
            v[:, 0] += label * 3
            row = {
                "sample_id": sid,
                "source_id": sid,
                "speaker_id": sid,
                "dataset": "synthetic-test-fixture",
                "split": split,
                "video": str(tmp_path / (sid + ".mp4")),
                "duration_s": n / 25,
                "clip_label": label,
                "forgery_intervals": [[0.4, 0.8]] if label else [],
                "mismatch_intervals": [[0.4, 0.8]] if label else [],
                "generator": "fixture",
            }
            path = cache / (sid + ".npz")
            save_npz(
                path,
                audio=a,
                visual=v,
                times_s=np.arange(n) / 25,
                audio_valid=np.ones(n, bool),
                visual_valid=np.ones(n, bool),
                mouth_open=np.ones(n, np.float32) * 0.2,
            )
            write_json(
                path.with_suffix(".json"),
                {
                    "feature_sha256": sha(path),
                    "source_sha256": "test-only",
                    "feature_signature": "synthetic-fixture-not-an-encoder",
                },
            )
            rows.append(row)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, rows)
    cfg = {
        "manifest": str(manifest),
        "cache": str(cache),
        "output": str(tmp_path / "run"),
        "seed": 7,
        "device": "cpu",
        "model": {"projection": 8, "hidden": 8, "dilations": [1, 2], "dropout": 0.1},
        "training": {"epochs": 2, "batch_size": 4, "lr": 0.01, "patience": 0},
    }
    return cfg, rows
