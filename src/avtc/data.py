"""Real corpus acquisition, identity splits and controlled local perturbations."""

import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch

from .syncnet import (
    COMMIT,
    FPS,
    METHOD,
    RATE,
    WEIGHT_HASHES,
    SyncNet,
    decode_crop,
    download,
    encode,
    ffmpeg_env,
    normalize_source,
    prepare_track,
    run,
    sha,
    write_json,
)


def download_grid(root, speakers=(1, 2, 3, 4, 5, 6), clips=40):
    """Download official speaker archives; only extract the requested paired utterances."""
    root = Path(root).resolve()
    if len(set(speakers)) < 3 or len(set(speakers)) != len(speakers) or clips < 1:
        raise ValueError("Select >=3 distinct speakers and a positive clip count")
    rows = []
    for speaker in speakers:
        sid = f"s{speaker}"
        if speaker == 21:
            raise ValueError("GRID speaker 21 has no video")
        base = f"https://spandh.dcs.shef.ac.uk/gridcorpus/{sid}"
        print(f"Downloading {sid}: video + RAW 50 kHz audio (large archives)", flush=True)
        videozip = download(base + f"/video/{sid}.mpg_vcd.zip", root / "archives" / f"{sid}.zip")
        audiotar = download(base + f"/audio/{sid}_50kHz.tar", root / "archives" / f"{sid}.tar")
        target = root / "selected" / sid
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(videozip) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".mpg"))[:clips]
            wanted = {Path(n).stem for n in names}
            for name in names:
                (target / Path(name).name).write_bytes(z.read(name))
        with tarfile.open(audiotar) as tar:
            for member in tar:
                if (
                    member.isfile()
                    and Path(member.name).suffix.lower() == ".wav"
                    and Path(member.name).stem in wanted
                ):
                    f = tar.extractfile(member)
                    with (target / Path(member.name).name).open("wb") as out:
                        import shutil

                        shutil.copyfileobj(f, out)
        for stem in sorted(wanted):
            video, audio = target / (stem + ".mpg"), target / (stem + ".wav")
            if not audio.is_file():
                raise ValueError(f"Missing original audio: {audio}")
            rows.append(
                {
                    "id": sid + "_" + stem,
                    "speaker": sid,
                    "video": str(video),
                    "audio": str(audio),
                    "alignment_basis": "nominal_GRID_raw_audio",
                }
            )

    # Split identities before creating any local variants.
    train_speakers = {f"s{s}" for s in speakers[:-2]}
    for row in rows:
        row["split"] = (
            "train"
            if row["speaker"] in train_speakers
            else "validation"
            if row["speaker"] == f"s{speakers[-2]}"
            else "test"
        )
    write_json(root / "manifest.json", rows)
    return root / "manifest.json"


def temporal_map(times, start, end, delta):
    # Piecewise-linear monotonic mapping with identity endpoints and a shifted plateau.
    ramp = max(0.24, 2.5 * abs(delta))
    if end - start < 2 * ramp + 0.16:
        raise ValueError("Warp interval too short for monotonic transition")
    g = np.minimum(np.clip((times - start) / ramp, 0, 1), np.clip((end - times) / ramp, 0, 1))
    mapped = times - delta * g
    if np.any(np.diff(mapped) <= 0):
        raise ValueError("Time map must preserve speech order")
    return mapped


def variants(frames, audio, delta=0.20, seed=42):
    duration = len(frames) / FPS
    width = 2 * max(0.24, 2.5 * delta) + 0.24
    if duration < width + 1.0:
        raise ValueError("Need more context around local warp")
    rng = np.random.default_rng(seed)
    start = float(rng.uniform(0.45, duration - width - 0.45))
    end = start + width
    ta, tv = np.arange(len(audio)) / RATE, (np.arange(len(frames)) + 0.5) / FPS
    yield "original", frames, audio, []
    for sign in (-1, 1):
        d = sign * delta
        source_times = temporal_map(ta, start, end, d)
        a = np.interp(source_times, ta, audio).round().clip(-32768, 32767).astype(np.int16)
        yield f"local_{sign:+d}", frames, a, [[start, end]]
        vi = np.rint(temporal_map(tv, start, end, d) * FPS - 0.5).astype(int)
        v = frames[vi.clip(0, len(frames) - 1)]
        yield f"synchronous_{sign:+d}", v, a, []


def prepare_dataset(manifest, output, assets="checkpoints/syncnet", device=None, seed=42):
    """Cache frozen CNN activations; trained projection layers remain outside this cache."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Manifest empty or duplicate source IDs")
    identities, video_hashes, audio_hashes = {}, {}, {}
    for row in rows:
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Split must be train/validation/test")
        if Path(row["id"]).name != row["id"] or row["id"] in (".", ".."):
            raise ValueError("Invalid source ID")
        identities.setdefault(row["speaker"], set()).add(row["split"])
        video_hashes.setdefault(sha(row["video"]), set()).add(row["split"])
        if row.get("audio"):
            audio_hashes.setdefault(sha(row["audio"]), set()).add(row["split"])
    if any(
        len(s) > 1 for groups in (identities, video_hashes, audio_hashes) for s in groups.values()
    ):
        raise ValueError("Speaker/source bytes overlap between splits")
    config = {"method": METHOD, "manifest_sha256": sha(manifest), "seed": seed, "commit": COMMIT}
    config_path = root / "prepare_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Preparation inputs changed; choose a new cache directory")
    write_json(config_path, config)
    extractor = SyncNet(assets, device)
    torch.save(
        {
            "audio": {k: v.cpu() for k, v in extractor.model.netfcaud.state_dict().items()},
            "visual": {k: v.cpu() for k, v in extractor.model.netfclip.state_dict().items()},
            "backbone_sha256": WEIGHT_HASHES["data/syncnet_v2.model"],
        },
        root / "projection_init.pt",
    )
    features, exclusions = [], []
    for number, row in enumerate(rows):
        print(f"Preparing {number + 1}/{len(rows)}: {row['id']} ({row['split']})", flush=True)
        folder = root / "sources" / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        fingerprint = {
            "video": sha(row["video"]),
            "audio": sha(row["audio"]) if row.get("audio") else None,
        }
        record = folder / "source.json"
        if record.exists():
            cached = json.loads(record.read_text())
            if cached["fingerprint"] != fingerprint:
                raise ValueError("Source content changed")
            if all((root / r["feature"]).is_file() for r in cached["rows"]):
                features.extend(cached["rows"])
                continue
        try:
            video = Path(row["video"])
            if row.get("audio"):
                video = folder / "paired.mkv"
                ff, env = ffmpeg_env(assets)
                run(
                    [
                        ff,
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        row["video"],
                        "-i",
                        row["audio"],
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "pcm_s16le",
                        video,
                    ],
                    env,
                )
            crop, meta = prepare_track(video, folder / "preprocess", assets)
            frames, pcm = decode_crop(crop, assets)
            frames, pcm, trim, lag = normalize_source(frames, pcm, extractor)
            # Different intervals/offsets across sources; fixed seed makes preparation reproducible.
            import hashlib

            local_seed = seed + int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)
            delta = float(np.random.default_rng(local_seed).choice([0.12, 0.16, 0.20]))
            family = []
            cached_visual = None
            for name, v, a, gold in variants(frames, pcm, delta, local_seed):
                if name.startswith("synchronous"):
                    vf, af, times, valid = encode(v, a, extractor)
                else:
                    if cached_visual is None:
                        cached_visual = extractor.visual(v, mid=True)
                    vf = cached_visual
                    af = extractor.audio(a, len(v), mid=True)
                    n = min(len(vf), len(af))
                    vf = vf[:n]
                    af = af[:n]
                    times = (np.arange(n) + 2.5) / FPS
                    energy = np.array(
                        [
                            np.sqrt(np.mean(a[i * 640 : (i + 5) * 640].astype(float) ** 2))
                            for i in range(n)
                        ]
                    )
                    valid = energy > max(100.0, float(energy.max()) * 0.05)
                label = np.zeros(len(times), dtype=np.float32)
                for start, end in gold:
                    label[(times >= start) & (times < end)] = 1
                if valid[5:-5].sum() < 3:
                    raise ValueError("Insufficient valid context in source family")
                relative = Path("features") / (row["id"] + "_" + name + ".npz")
                (root / relative).parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    root / relative,
                    visual=vf.astype(np.float32),
                    audio=af.astype(np.float32),
                    label=label,
                    valid=valid,
                    times=times + meta["start_s"] + trim,
                )
                family.append(
                    {
                        "source_id": row["id"],
                        "speaker": row["speaker"],
                        "split": row["split"],
                        "variant": name,
                        "feature": relative.as_posix(),
                        "gold_intervals": [
                            [s + meta["start_s"] + trim, e + meta["start_s"] + trim]
                            for s, e in gold
                        ],
                    }
                )
            write_json(
                record,
                {
                    "fingerprint": fingerprint,
                    "rows": family,
                    "delta_s": delta,
                    "estimated_source_lag_ms": lag,
                    "sync_basis": "Model-normalized nominal source; manual review needed",
                },
            )
            features.extend(family)
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            exclusions.append({"source_id": row["id"], "split": row["split"], "reason": str(exc)})
            print("Excluded entire source family:", exc, flush=True)
        write_json(root / "excluded.json", exclusions)
    write_json(root / "excluded.json", exclusions)
    for split in ("train", "validation", "test"):
        subset = [r for r in features if r["split"] == split]
        if not subset or not any(r["variant"].startswith("local") for r in subset):
            raise ValueError(f"No usable {split} data; inspect excluded.json")
    write_json(
        root / "index.json",
        {
            "method": METHOD,
            "config": config,
            "rows": features,
            "backbone_sha256": WEIGHT_HASHES["data/syncnet_v2.model"],
            "feature_type": "SyncNet frozen CNN outputs, 512-D before pretrained FC",
        },
    )
    return root
