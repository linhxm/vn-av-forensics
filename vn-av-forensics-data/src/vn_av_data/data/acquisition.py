"""Download only user-selected YouTube URLs, or index already downloaded source videos."""

import csv
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vn_av_data.common.runtime import read_json, sha, write_json
from vn_av_data.data.manifest import read_manifest, write_manifest
from vn_av_data.data.media import ffmpeg, probe

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpg"}


def source_path(row, manifest):
    path = Path(row["video"])
    return path if path.is_absolute() else Path(manifest).resolve().parent / path


def youtube_id(url):
    import re

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        value = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        value = (
            parse_qs(parsed.query).get("v", [""])[0]
            if parsed.path == "/watch"
            else parsed.path.split("/")[-1]
            if parsed.path.startswith(("/shorts/", "/live/"))
            else ""
        )
    else:
        value = ""
    if parsed.scheme not in {"https", "http"} or not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError(f"Use a single YouTube video URL: {url}")
    return value


def download_sources(selection, output, cookies_from_browser=None, force_ipv4=False):
    import shutil

    import yt_dlp

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with Path(selection).open(encoding="utf-8-sig", newline="") as handle:
        selected = list(csv.DictReader(handle))
    ids = [youtube_id(row["url"].strip()) for row in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Source CSV must contain unique YouTube URLs")
    manifest = output / "sources.jsonl"
    existing = (
        {row["source_id"]: row for row in read_manifest(manifest)} if manifest.exists() else {}
    )
    errors = []
    for row, video_id in zip(selected, ids):
        sid = "yt_" + video_id
        if sid in existing:
            path = source_path(existing[sid], manifest)
            if path.is_file() and sha(path) == existing[sid]["sha256"]:
                continue
        try:
            options = {
                "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/best[height<=720]",
                "outtmpl": str(output / (video_id + ".%(ext)s")),
                "noplaylist": True,
                "merge_output_format": "mp4",
                "ffmpeg_location": ffmpeg(),
                "continuedl": True,
                "overwrites": False,
                "socket_timeout": 30,
            }
            if cookies_from_browser:
                options["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
            if force_ipv4:
                options["source_address"] = "0.0.0.0"
            if shutil.which("node"):
                options["js_runtimes"] = {"node": {}}
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(
                    "https://www.youtube.com/watch?v=" + video_id, download=True
                )
            paths = [p for p in output.glob(video_id + ".*") if p.suffix in VIDEO_SUFFIXES]
            if len(paths) != 1:
                raise ValueError("Expected one merged video after download")
            path = paths[0]
            metadata = probe(path)
            existing[sid] = {
                "source_id": sid,
                "video": path.name,
                "sha256": sha(path),
                "url": "https://www.youtube.com/watch?v=" + video_id,
                "title": info.get("title", ""),
                "speaker_id": row.get("speaker_id") or None,
                "dataset": "youtube",
                **metadata,
            }
            write_manifest(manifest, list(existing.values()))
        except Exception as exc:  # noqa: BLE001 -- persist per-source failures for resume
            errors.append({"source_id": sid, "error": str(exc)})
        write_json(output / "download-errors.json", errors)
    if errors:
        raise RuntimeError(
            f"{len(errors)} downloads failed; see {output / 'download-errors.json'}; rerun to resume"
        )
    return {"sources": len(existing), "manifest": str(manifest)}


def index_sources(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    rows, hashes = [], set()
    if output.exists():
        raise FileExistsError(f"Use a new source manifest: {output}")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        digest = sha(path)
        if digest in hashes:
            continue
        hashes.add(digest)
        metadata_path = path.with_suffix(".source.json")
        meta = read_json(metadata_path) if metadata_path.exists() else {}
        import os

        rows.append(
            {
                **probe(path),
                "source_id": meta.get("source_id", "local_" + digest[:20]),
                "sha256": digest,
                "video": os.path.relpath(path, output.parent),
                "speaker_id": meta.get("speaker_id"),
                "dataset": meta.get("dataset", "local"),
            }
        )
    if not rows or len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Need source videos with unique source IDs")
    write_manifest(output, rows)
    return {"sources": len(rows), "manifest": str(output)}
