"""PTS-aware decoding. The two streams retain their relative start times and gaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vn_av_data.common.runtime import require_file, run


def ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path):
    import av

    with av.open(str(path)) as container:
        if not container.streams.video or not container.streams.audio:
            raise ValueError("Video requires both image and audio streams")
        v = container.streams.video[0]
        duration = (
            float(container.duration / av.time_base)
            if container.duration
            else float(v.duration * v.time_base)
            if v.duration
            else None
        )
        return {
            "duration_s": duration,
            "width": v.width,
            "height": v.height,
            "video_start_s": float(v.start_time * v.time_base)
            if v.start_time is not None
            else None,
            "audio_start_s": float(
                container.streams.audio[0].start_time * container.streams.audio[0].time_base
            )
            if container.streams.audio[0].start_time is not None
            else None,
        }


def decode(path, max_duration=60.0, max_side=384, sample_rate=16000):
    import av
    import cv2

    require_file(path, "input video")
    info = probe(path)
    if info["duration_s"] and info["duration_s"] > max_duration + 0.1:
        raise ValueError(f"Video longer than {max_duration}s; use data segment first")
    frames, selected_pts = [], []
    with av.open(str(path)) as container:
        starts = [x for x in (info["video_start_s"], info["audio_start_s"]) if x is not None]
        origin = (
            float(container.start_time / av.time_base)
            if container.start_time is not None
            else min(starts, default=0.0)
        )
        previous, previous_pts, target = None, None, 0.0
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Video frame without timestamp")
            t = float(frame.pts * frame.time_base) - origin
            if t > max_duration + 0.1:
                raise ValueError("Decoded video exceeds duration limit")
            image = frame.to_ndarray(format="bgr24")
            scale = min(1.0, max_side / max(image.shape[:2]))
            if scale < 1.0:
                image = cv2.resize(
                    image,
                    (max(2, round(image.shape[1] * scale)), max(2, round(image.shape[0] * scale))),
                )
            while target < t - 1e-7:
                frames.append(previous.copy() if previous is not None else np.zeros_like(image))
                selected_pts.append(previous_pts if previous_pts is not None else -1.0)
                target += 0.04
            previous, previous_pts = image, t
        if previous is None:
            raise ValueError("No decoded video frames")
        video_end = min(max_duration, previous_pts + 0.04)
        while target < video_end - 1e-7:
            frames.append(previous.copy())
            selected_pts.append(previous_pts)
            target += 0.04
    n = len(frames)
    if n < 10:
        raise ValueError("Need at least 0.4s of video")
    times = np.arange(n, dtype=np.float64) / 25
    visual_valid = (np.asarray(selected_pts) >= 0) & (times - np.asarray(selected_pts) <= 0.12)
    if sample_rate <= 0 or sample_rate % 25:
        raise ValueError("sample_rate must be positive and divisible by 25")
    samples_per_frame = sample_rate // 25
    pcm = np.zeros(n * samples_per_frame, np.float32)
    audio_known = np.zeros(len(pcm), bool)
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)

        def insert(f):
            if f.pts is None:
                raise ValueError("Resampled audio without timestamp")
            start = round((float(f.pts * f.time_base) - origin) * sample_rate)
            samples = f.to_ndarray().reshape(-1)
            lo, hi = max(0, start), min(len(pcm), start + len(samples))
            if hi > lo:
                pcm[lo:hi] = samples[lo - start : hi - start]
                audio_known[lo:hi] = True

        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                insert(f)
        for f in resampler.resample(None):
            insert(f)
    audio_valid = audio_known.reshape(n, samples_per_frame).mean(1) >= 0.95
    return {
        "frames": np.stack(frames),
        "pcm": pcm,
        "times_s": times,
        "audio_valid": audio_valid,
        "visual_valid": visual_valid,
        "selected_pts_s": np.asarray(selected_pts),
        "origin_s": origin,
    }


def encode(frames, pcm, output):
    """Encode generated controls identically, retaining a 25Hz/16kHz common timeline."""
    import tempfile

    import cv2
    from scipy.io import wavfile

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        video = Path(tmp) / "video.avi"
        audio = Path(tmp) / "audio.wav"
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 25, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Cannot open FFV1 video writer")
        for frame in frames:
            writer.write(frame)
        writer.release()
        wavfile.write(audio, 16000, np.clip(pcm * 32767, -32768, 32767).astype(np.int16))
        temp = Path(tmp) / "result.mp4"
        run(
            [
                ffmpeg(),
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                video,
                "-i",
                audio,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                temp,
            ]
        )
        temp.replace(output)
