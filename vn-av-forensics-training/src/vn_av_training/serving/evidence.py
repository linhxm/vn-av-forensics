"""Export bounded evidence clips without changing source speed or global lag."""

from __future__ import annotations

import math
import subprocess
import uuid
from pathlib import Path

CONTEXT_OPTIONS = (0.0, 0.75, 1.5)


def export_clip(source: Path, output_dir: Path, index: int, interval, context: float):
    if context not in CONTEXT_OPTIONS:
        raise ValueError("Unsupported context duration")
    start, end = map(float, interval)
    if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
        raise ValueError("Invalid evidence interval")
    start = max(0.0, start - context)
    end += context
    destination = output_dir / f"evidence-{index + 1}-{int(context * 1000)}ms.mp4"
    if destination.is_file():
        return destination

    import imageio_ffmpeg

    # Unique temporary outputs keep simultaneous downloads from exposing partial files.
    temporary = output_dir / f"evidence-{uuid.uuid4().hex}.tmp.mp4"
    try:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(source),
                "-t",
                f"{end - start:.6f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-y",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty clip")
        # Another request may already have finished exporting the same clip.
        if not destination.is_file():
            temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)
