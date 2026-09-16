"""Explicit, resumable downloads. Record exact revisions and hashes for reproducibility."""

import os
import shutil
import urllib.request
from pathlib import Path

from vn_av_training.common.runtime import read_json, run, sha, write_json

MEDIA = {
    "face": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}
FATE_REVISION = "beae95aeb6f72cf1751d06d1428931016a7a1867"
HF_REVISIONS = {
    "facebook/pe-av-small": "dd050762bb9704ae9cd996ca45532a98f81d817e",
    "Guan123/fate": "8463ab93a644a22bd85db91e7e77d99ebe1ec5e0",
}


def fetch_file(url, destination):
    destination = Path(destination)
    record = destination.with_suffix(destination.suffix + ".download.json")
    if destination.exists():
        if record.exists() and read_json(record)["sha256"] == sha(destination):
            return read_json(record)
        raise ValueError(f"Unverified existing asset: {destination}; use a new asset path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if partial.stat().st_size < 1024:
            raise ValueError(f"Invalid model download: {url}")
        os.replace(partial, destination)
        result = {"url": url, "sha256": sha(destination), "bytes": destination.stat().st_size}
        write_json(record, result)
        return result
    finally:
        partial.unlink(missing_ok=True)


def setup_assets(cfg, only="all"):
    result = {}
    if only in ("all", "media"):
        target = cfg["encoder"]["face_model"]
        result["face"] = fetch_file(MEDIA["face"], target)
    if only in ("all", "fate", "source"):
        encoder = cfg["encoder"]
        repo = Path(encoder["repo"])
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--depth", "1", "https://github.com/guankaisi/FATE.git", repo])
            run(["git", "fetch", "--depth", "1", "origin", FATE_REVISION], cwd=repo)
            run(["git", "checkout", "--detach", FATE_REVISION], cwd=repo)
        if not (repo / "models/pe_av/modeling_pe_audio_video.py").is_file():
            raise ValueError(f"Incomplete FATE source: {repo}")
        commit = run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
        if commit != FATE_REVISION:
            raise ValueError(
                "FATE source differs from the tested revision; use a new encoder.repo path"
            )
        lock = repo / "vn_av_revision.json"
        if lock.exists() and read_json(lock)["commit"] != commit:
            raise ValueError("FATE revision changed; use a separate source/cache directory")
        write_json(lock, {"commit": commit, "repository": "https://github.com/guankaisi/FATE"})
        result["fate_source"] = commit
        if only != "source":
            from huggingface_hub import snapshot_download

            for key, name in (("base", "facebook/pe-av-small"), ("adapter", "Guan123/fate")):
                target = Path(encoder[key])
                revision_file = target / "vn_av_revision.json"
                revision = (
                    read_json(revision_file)["revision"]
                    if revision_file.exists()
                    else HF_REVISIONS[name]
                )
                # Save the revision before downloading so interrupted runs resume the same snapshot.
                write_json(revision_file, {"repo_id": name, "revision": revision})
                print(f"Downloading {name}@{revision}", flush=True)
                try:
                    snapshot_download(
                        name,
                        revision=revision,
                        local_dir=target,
                        max_workers=2,
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Download {name} failed. Rerun relations-setup --only fate "
                        f"on a working Internet connection; partial files retained. {exc}"
                    ) from exc
                result[key] = {"repo_id": name, "revision": revision}
    return result
