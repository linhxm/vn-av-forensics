"""Download only user-selected YouTube URLs, or index already downloaded source videos."""

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


def download_sources(
    selection, output, cookies_from_browser=None, force_ipv4=False, limit=0, dry_run=False
):
    import shutil

    import yt_dlp

    from vn_av_data.data.collect import normalize_sources
    from vn_av_data.data.source_io import read_rows, write_rows

    selected = normalize_sources(read_rows(selection))
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if dry_run:
        return {"selected": len(selected), "dry_run": True}
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js must be on PATH for the YouTube downloader")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "download_sources.csv"
    write_rows(snapshot, selected)
    manifest = output / "sources.jsonl"
    existing = (
        {row["source_id"]: row for row in read_manifest(manifest)} if manifest.exists() else {}
    )
    ids = {"yt_" + row["video_id"] for row in selected}
    if set(existing) - ids:
        raise ValueError("Run contains other sources; use a new download run")
    results = [{**row, "status": "pending", "filename": "", "error": ""} for row in selected]
    errors = []
    attempted = 0
    for result in results:
        video_id = result["video_id"]
        sid = "yt_" + video_id
        old = existing.get(sid)
        if old:
            path = source_path(old, manifest)
            if path.is_file() and sha(path) == old["sha256"]:
                probe(path)
                result.update(status="downloaded", filename=path.name)
                write_rows(output / "download_results.csv", results, mutable=True)
                continue
        if limit and attempted >= limit:
            continue
        attempted += 1
        try:
            options = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                "outtmpl": str(output / (video_id + ".%(ext)s")),
                "noplaylist": True,
                "merge_output_format": "mp4",
                "ffmpeg_location": ffmpeg(),
                "continuedl": True,
                "overwrites": False,
                "socket_timeout": 30,
                "js_runtimes": {"node": {"path": node}},
            }
            if cookies_from_browser:
                options["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
            if force_ipv4:
                options["source_address"] = "0.0.0.0"
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(result["url"], download=True)
            path = output / (video_id + ".mp4")
            metadata = probe(path)
            existing[sid] = {
                **{k: v for k, v in result.items() if k not in {"status", "filename", "error"}},
                "source_id": sid,
                "video": path.name,
                "sha256": sha(path),
                "title": result.get("title") or info.get("title", ""),
                "channel": result.get("channel") or info.get("channel", ""),
                "speaker_id": result.get("speaker_id") or None,
                "dataset": "youtube",
                **metadata,
            }
            write_manifest(manifest, list(existing.values()))
            result.update(status="downloaded", filename=path.name)
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            errors.append({"source_id": sid, "error": str(exc)})
        write_rows(output / "download_results.csv", results, mutable=True)
        write_json(output / "download-errors.json", errors)
    write_rows(output / "download_results.csv", results, mutable=True)
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; rerun same batch to resume")
    return {
        "sources": len(existing),
        "pending": sum(r["status"] == "pending" for r in results),
        "manifest": str(manifest),
    }


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
