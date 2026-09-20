"""VAD + scene boundaries + single-face quality, inspired by the Capstone cutter.

Quality acceptance does not assert audio-visual synchrony. Every clip retains source times.
"""

import csv
import io
import math
from itertools import pairwise
from pathlib import Path

from vn_av_data.common.runtime import atomic_bytes, fingerprint, read_json, run, sha, write_json
from vn_av_data.data.acquisition import source_path
from vn_av_data.data.manifest import read_manifest
from vn_av_data.data.media import ffmpeg, probe


def write_csv(path, rows, fields=None):
    stream = io.StringIO(newline="")
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode("utf-8-sig"))


def media_origin(path):
    import av

    info = probe(path)
    with av.open(str(path)) as container:
        starts = [info[k] for k in ("video_start_s", "audio_start_s") if info[k] is not None]
        origin = (
            float(container.start_time / av.time_base)
            if container.start_time is not None
            else min(starts, default=0.0)
        )
    return origin, info


def scan_visual(path, cfg, origin, tracker=None):
    import av
    import cv2

    from vn_av_data.features.face import FaceTracker

    tracker = tracker or FaceTracker(cfg["face_model"], min_size=cfg["min_face_size"])
    samples, scenes, previous_hist, next_sample = [], [], None, 0.0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Video frame without timestamp")
            time = float(frame.pts * frame.time_base) - origin
            image = frame.to_ndarray(format="bgr24")
            tiny = cv2.resize(image, (96, 54))
            hist = cv2.calcHist([tiny], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            cv2.normalize(hist, hist)
            if (
                previous_hist is not None
                and cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL)
                < cfg["scene_correlation"]
            ):
                scenes.append(max(0, time))
                tracker.previous = None
            previous_hist = hist
            if time + 1e-6 < next_sample:
                continue
            scale = min(1, cfg["scan_max_side"] / max(image.shape[:2]))
            image = cv2.resize(
                image, (round(image.shape[1] * scale), round(image.shape[0] * scale))
            )
            info = tracker.inspect(image)
            samples.append({"time_s": max(0, time), **info})
            next_sample = time + 1 / cfg["sample_fps"]
    return samples, scenes


def plan_clips(speech, scenes, duration, cfg):
    minimum, maximum, overlap = cfg["min_seconds"], cfg["max_seconds"], cfg["overlap_seconds"]
    if not (0 < minimum <= maximum <= 60 and 0 <= overlap < minimum):
        raise ValueError("Need 0 <= overlap < min <= max <= 60 seconds")
    boundaries = sorted({0.0, duration, *(s for s in scenes if 0 < s < duration)})
    result = []
    for start, end in speech:
        for left, right in pairwise(boundaries):
            a, b = max(start, left), min(end, right)
            a, b = math.ceil(a * 25 - 1e-6) / 25, math.floor(b * 25 + 1e-6) / 25
            while b - a >= minimum - 1e-6:
                stop = min(b, a + maximum)
                # Absorb a short tail by moving the final full window backwards.
                if 0 < b - stop < minimum - overlap:
                    stop = b
                    a = max(a, stop - maximum)
                result.append((round(a, 6), round(stop, 6)))
                if stop == b:
                    break
                a = stop - overlap
    return sorted(set(result))


def cut_media(source, output, start, end, origin):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mp4")
    # Both streams subtract the SAME timestamp. Never independently reset STARTPTS.
    absolute = origin + start
    stop = origin + end
    filters = (
        f"[0:v:0]trim=start={absolute:.6f}:end={stop:.6f},"
        f"setpts=PTS-({absolute:.6f})/TB,scale=trunc(iw/2)*2:trunc(ih/2)*2[v];"
        f"[0:a:0]atrim=start={absolute:.6f}:end={stop:.6f},"
        f"asetpts=PTS-({absolute:.6f})/TB[a]"
    )
    try:
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-copyts",
                "-ss",
                str(start),
                "-i",
                str(source),
                "-filter_complex",
                filters,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "vfr",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-avoid_negative_ts",
                "disabled",
                "-movflags",
                "+faststart",
                partial,
            ]
        )
        info = probe(partial)
        if not info["duration_s"] or abs(info["duration_s"] - (end - start)) > 0.2:
            raise ValueError("Encoded clip duration disagrees with requested source interval")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def curate_sources(manifest, output, cfg, speech_detector=None, visual_scanner=None):
    from vn_av_data.data.source_io import read_rows

    progress = Path(manifest).resolve().parent / "download_results.csv"
    if progress.exists() and any(r.get("status") != "downloaded" for r in read_rows(progress)):
        raise ValueError(
            "Download batch still contains pending/failed sources; finish step 02 first"
        )
    from vn_av_data.data.vad import detect_speech

    cfg = dict(cfg["curation"])
    if not 0 < cfg["sample_fps"] <= 25 or not 0 <= cfg["min_face_ratio"] <= 1:
        raise ValueError("Invalid visual sampling/gate configuration")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    if not rows or len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Need unique source IDs")
    signature = fingerprint(
        {
            "config": cfg,
            "sources": rows,
            "implementation": [
                sha(__file__),
                sha(Path(__file__).with_name("vad.py")),
                sha(Path(__file__).parents[1] / "features/face.py"),
            ],
            "models": {key: sha(cfg[key]) for key in ("face_model", "vad_model")},
        }
    )
    lock = output / "cut-run.json"
    if lock.exists() and read_json(lock)["signature"] != signature:
        raise ValueError("Inputs/settings changed; use a new cut output directory")
    write_json(lock, {"signature": signature, "config": cfg})
    candidates, rejected, errors = [], [], []
    for source in rows:
        sid = source["source_id"]
        journal = output / "sources" / (fingerprint(sid)[:20] + ".json")
        path = source_path(source, manifest)
        if sha(path) != source["sha256"]:
            raise ValueError(f"Source changed: {path}")
        if journal.exists():
            done = read_json(journal)
            for clip in done["candidates"]:
                target = output / clip["file_path"]
                if not target.is_file() or sha(target) != clip["sha256"]:
                    raise ValueError(f"Published clip missing/changed: {target}; use a new cut run")
            candidates.extend(done["candidates"])
            rejected.extend(done["rejected"])
            continue
        try:
            print(f"Cut source {sid}", flush=True)
            origin, info = media_origin(path)
            duration = info["duration_s"]
            if not duration or not 0 < duration <= cfg["max_source_hours"] * 3600:
                raise ValueError("Source duration missing or exceeds configured limit")
            speech = (speech_detector or detect_speech)(path, cfg["vad_model"], origin, duration)
            samples, scenes = (visual_scanner or scan_visual)(path, cfg, origin)
            accepted_source, rejected_source = [], []
            for start, end in plan_clips(speech, scenes, duration, cfg):
                selected = [s for s in samples if start <= s["time_s"] < end]
                ratio = sum(s["valid"] for s in selected) / max(1, len(selected))
                coverage = len(selected) / max(1, (end - start) * cfg["sample_fps"])
                clip_id = "clip_" + fingerprint([sid, start, end])[:24]
                base = {
                    "clip_id": clip_id,
                    "source_id": sid,
                    "url": source.get("url", ""),
                    "source_sha256": source["sha256"],
                    "speaker_id": source.get("speaker_id") or "",
                    "dataset": source.get("dataset", "youtube"),
                    **{
                        key: source.get(key, "")
                        for key in (
                            "channel",
                            "program_id",
                            "episode_id",
                            "canonical_source_id",
                            "tier",
                        )
                    },
                    "source_start_s": start,
                    "source_end_s": end,
                    "face_ratio": ratio,
                    "visual_sample_coverage": min(1.0, coverage),
                }
                if ratio < cfg["min_face_ratio"] or coverage < 0.8:
                    rejected_source.append({**base, "reason": "face_or_coverage"})
                    continue
                target = output / "clips" / (clip_id + ".mp4")
                receipt = target.with_suffix(".json")
                expected = fingerprint([signature, sid, start, end])
                if target.exists():
                    if (
                        not receipt.exists()
                        or read_json(receipt).get("signature") != expected
                        or read_json(receipt).get("sha256") != sha(target)
                    ):
                        raise ValueError(f"Unverified existing clip: {target}; use a new run")
                else:
                    cut_media(path, target, start, end, origin)
                    write_json(
                        receipt,
                        {
                            "format": "curated-clip-v1",
                            "signature": expected,
                            "sha256": sha(target),
                            "source_id": sid,
                            "source_start_s": start,
                            "source_end_s": end,
                        },
                    )
                accepted_source.append(
                    {
                        **base,
                        "duration_s": end - start,
                        "file_path": target.relative_to(output).as_posix(),
                        "sha256": sha(target),
                        "quality": "pass",
                        "decision": "uncertain",
                        "sync_status": "unverified",
                    }
                )
            write_json(
                journal,
                {
                    "candidates": accepted_source,
                    "rejected": rejected_source,
                    "speech_regions": speech,
                    "scene_boundaries": scenes,
                },
            )
            candidates.extend(accepted_source)
            rejected.extend(rejected_source)
        except Exception as exc:  # noqa: BLE001 -- record a source failure and continue the batch
            errors.append({"source_id": sid, "error": str(exc)})
        write_json(output / "errors.json", errors)
    # Never overwrite the review CSV: it may already contain human decisions.
    if candidates:
        write_csv(output / "candidates.csv", candidates)
        if not (output / "review.csv").exists():
            write_csv(output / "review.csv", candidates)
    write_json(output / "rejected.json", rejected)
    summary = {
        "sources": len(rows),
        "candidates": len(candidates),
        "rejected": len(rejected),
        "errors": len(errors),
        "output": str(output),
        "note": "Quality pass is not verified synchrony; review before importing clean labels",
    }
    write_json(output / "summary.json", summary)
    if errors:
        raise RuntimeError(
            f"{len(errors)} source failures; see {output / 'errors.json'}; rerun to resume"
        )
    return summary
