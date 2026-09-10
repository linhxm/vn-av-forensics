"""Pinned SyncNet backbone, official face crops and CNN feature extraction."""

import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import urllib.request
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from python_speech_features import mfcc

COMMIT = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
RATE, FPS = 16000, 25
METHOD = "local-consistency-v1"
WEIGHT_HASHES = {
    "data/syncnet_v2.model": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
    "detectors/s3fd/weights/sfd_face.pth": "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def download(url, path):
    """Atomic download; bounded HTTP ranges avoid Oxford large-response stalls."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    if path.suffix in (".model", ".pth"):
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=30) as r:
            content_range = r.headers.get("Content-Range", "")
            if r.status != 206 or not content_range:
                raise ValueError("Weights server must support byte ranges; attach assets manually")
            total = int(content_range.split("/")[-1])
            r.read()
        chunk = 256 * 1024

        def fetch(start):
            stop = min(total, start + chunk) - 1
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop}"})
                    with urllib.request.urlopen(req, timeout=45) as r:
                        data = r.read()
                        if (
                            r.status != 206
                            or r.headers.get("Content-Range") != f"bytes {start}-{stop}/{total}"
                            or len(data) != stop - start + 1
                        ):
                            raise ValueError("Unexpected partial model response")
                        return start, data
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)

        with part.open("wb") as f, ThreadPoolExecutor(max_workers=8) as pool:
            for i, (start, data) in enumerate(pool.map(fetch, range(0, total, chunk))):
                f.seek(start)
                f.write(data)
                if i % 40 == 0:
                    print(f"{path.name}: {min(total, start + len(data))}/{total} bytes", flush=True)
        if part.stat().st_size != total:
            raise ValueError("Incomplete model download")
    else:
        with urllib.request.urlopen(url, timeout=90) as r, part.open("wb") as f:
            size = r.headers.get("Content-Length")
            shutil.copyfileobj(r, f)
        if size and part.stat().st_size != int(size):
            raise ValueError("Incomplete archive download; rerun")
    part.replace(path)
    return path


def setup(root):
    root = Path(root).resolve()
    repo = root / ("syncnet_python-" + COMMIT)
    if not (repo / "run_pipeline.py").is_file():
        archive = download(
            "https://codeload.github.com/joonson/syncnet_python/zip/" + COMMIT,
            root / "upstream.zip",
        )
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if not (root / n).resolve().is_relative_to(root):
                    raise ValueError("Unsafe upstream archive")
            for name in z.namelist():
                parts = Path(name).parts
                if len(parts) > 1 and parts[1] in (
                    "SyncNetModel.py",
                    "run_pipeline.py",
                    "LICENSE.md",
                    "detectors",
                ):
                    z.extract(name, root)
        archive.unlink()
    urls = {
        "data/syncnet_v2.model": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
        "detectors/s3fd/weights/sfd_face.pth": "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth",
    }
    hashes = {}
    for name, url in urls.items():
        target = download(url, repo / name)
        hashes[name] = sha(target)
        if hashes[name] != WEIGHT_HASHES[name]:
            raise ValueError("Checkpoint hash mismatch: " + str(target))
    write_json(
        root / "provenance.json",
        {
            "upstream": "https://github.com/joonson/syncnet_python",
            "commit": COMMIT,
            "assets": urls,
            "sha256": hashes,
            "note": "Hashes record downloaded assets, not independent upstream signatures.",
        },
    )
    return repo


def ffmpeg_env(root):
    import imageio_ffmpeg

    root = Path(root).resolve()
    binary = Path(imageio_ffmpeg.get_ffmpeg_exe())
    folder = root / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists():
        shutil.copy2(binary, target)
        target.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = str(folder) + os.pathsep + env.get("PATH", "")
    env["OMP_NUM_THREADS"] = "2"
    return str(target), env


def run(args, env=None, log=None):
    if log:
        with Path(log).open("w", encoding="utf-8") as f:
            subprocess.run(
                [str(a) for a in args], env=env, stdout=f, stderr=subprocess.STDOUT, check=True
            )
    else:
        subprocess.run(
            [str(a) for a in args],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )


def prepare_track(video, out, assets):
    """Official S3FD crop. Reject multi-track/partial coverage; original timeline retained."""
    out, video = Path(out).resolve(), Path(video).resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = setup(assets)
    _ff, env = ffmpeg_env(assets)
    source_hash = sha(video)
    meta_file = out / "track.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text())
        if meta["source_sha256"] != source_hash or meta["commit"] != COMMIT:
            raise ValueError("Cached source changed; use a new output directory")
        return Path(meta["crop"]), meta
    reference = "track-" + uuid.uuid4().hex[:10]
    command = [
        sys.executable,
        repo / "run_pipeline.py",
        "--videofile",
        video,
        "--data_dir",
        out,
        "--reference",
        reference,
        "--min_track",
        "20",
        "--min_face_size",
        "30",
        "--facedet_scale",
        "1.0",
    ]
    # Upstream resolves detector weights relative to its working directory.
    with (out / "preprocess.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(a) for a in command],
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    # Read only metadata just produced by the pinned subprocess, never an uploaded pickle.
    with (out / "pywork" / reference / "tracks.pckl").open("rb") as f:
        tracks = pickle.load(f)
    if len(tracks) != 1:
        raise ValueError("Need exactly one continuous face track; see preprocess.log")
    frames = tracks[0]["track"]["frame"]
    with (out / "pywork" / reference / "faces.pckl").open("rb") as f:
        detections = pickle.load(f)
    # Tracking mutates detections internally, but the saved file precedes that mutation.
    ambiguous = np.mean([len(x) > 1 for x in detections])
    coverage = len(frames) / max(1, len(detections))
    if coverage < 0.9 or ambiguous > 0.05:
        raise ValueError("Insufficient single-face coverage or ambiguous speaker")
    crop = out / "pycrop" / reference / "00000.avi"
    meta = {
        "source_sha256": source_hash,
        "commit": COMMIT,
        "crop": str(crop),
        "start_s": float(frames[0] / FPS),
        "coverage": float(coverage),
        "duration_s": float(len(frames) / FPS),
    }
    write_json(meta_file, meta)
    return crop, meta


def decode_crop(path, assets):
    ff, env = ffmpeg_env(assets)
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    r = subprocess.run(
        [
            ff,
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-f",
            "s16le",
            "-",
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    audio = np.frombuffer(r.stdout, dtype="<i2").copy()
    n = min(len(frames), len(audio) // 640)
    if n < 40:
        raise ValueError("Clip too short")
    return np.stack(frames[:n]), audio[: n * 640]


class SyncNet:
    def __init__(self, assets, device=None):
        self.repo = setup(assets)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(2)
        spec = importlib.util.spec_from_file_location(
            "syncnet_upstream_model", self.repo / "SyncNetModel.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.model = mod.S(num_layers_in_fc_layers=1024).to(self.device).eval()
        self.model.load_state_dict(
            torch.load(self.repo / "data/syncnet_v2.model", map_location="cpu", weights_only=True),
            strict=True,
        )

    @torch.inference_mode()
    def visual(self, frames, batch=16, mid=False):
        outputs = []
        for start in range(0, len(frames) - 5, batch):
            ix = range(start, min(len(frames) - 5, start + batch))
            x = np.stack([frames[i : i + 5].transpose(3, 0, 1, 2) for i in ix])
            outputs.append(
                (
                    self.model.forward_lipfeat(torch.from_numpy(x).float().to(self.device))
                    if mid
                    else self.model.forward_lip(torch.from_numpy(x).float().to(self.device))
                )
                .cpu()
                .numpy()
            )
        return np.concatenate(outputs)

    @torch.inference_mode()
    def audio(self, audio, nframes, batch=16, mid=False):
        # Same PCM scale, MFCC defaults, 20-column windows and BGR pixels as official demo.
        x = mfcc(audio, RATE).T.astype(np.float32)
        outputs = []
        for start in range(0, nframes - 5, batch):
            ix = range(start, min(nframes - 5, start + batch))
            b = np.stack([x[:, i * 4 : i * 4 + 20] for i in ix])[:, None]
            outputs.append(
                (
                    self.model.netcnnaud(torch.from_numpy(b).to(self.device)).flatten(1)
                    if mid
                    else self.model.forward_aud(torch.from_numpy(b).to(self.device))
                )
                .cpu()
                .numpy()
            )
        return np.concatenate(outputs)


def normalize_source(frames, audio, extractor):
    """Estimate constant timing nuisance before local analysis; never a fake/real detector."""
    v = extractor.visual(frames)
    a = extractor.audio(audio, len(frames))
    n, search = min(len(v), len(a)), 10
    if n < 40:
        raise ValueError("Video too short for reliable temporal context")
    ix = np.arange(search, n - search)
    rms = np.array(
        [np.sqrt(np.mean(audio[i * 640 : (i + 5) * 640].astype(float) ** 2)) for i in ix]
    )
    voiced = rms > max(100.0, float(rms.max()) * 0.05)
    if voiced.sum() < 10:
        raise ValueError("Insufficient speech")
    distances = np.stack(
        [np.linalg.norm(v[ix] - a[ix + k], axis=1) for k in range(-search, search + 1)], axis=1
    )
    lag = int(np.argmin(np.median(distances[voiced], axis=0))) - search
    if abs(lag) == search:
        raise ValueError("Source timing outside search range; review video")
    left, right = max(0, -lag), min(len(frames), len(audio) // 640 - lag)
    return frames[left:right], audio[(left + lag) * 640 : (right + lag) * 640], left / FPS, lag * 40


def encode(frames, audio, extractor):
    v = extractor.visual(frames, mid=True)
    a = extractor.audio(audio, len(frames), mid=True)
    n = min(len(v), len(a))
    energy = np.array(
        [np.sqrt(np.mean(audio[i * 640 : (i + 5) * 640].astype(float) ** 2)) for i in range(n)]
    )
    valid = energy > max(100.0, float(energy.max()) * 0.05)
    return v[:n], a[:n], (np.arange(n) + 2.5) / FPS, valid


def intervals(times, active):
    result, start, previous = [], None, None
    for t, flag in zip(times, active):
        if flag and start is None:
            start = float(t - 0.02)
        if not flag and start is not None:
            result.append([start, float(previous + 0.02)])
            start = None
        previous = t
    if start is not None:
        result.append([start, float(previous + 0.02)])
    return result
