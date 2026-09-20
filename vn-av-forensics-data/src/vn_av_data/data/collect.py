# Adapted from the user-authorized VN-AV-DF-Capstone collect workflow.
"""Normalize selected YouTube URLs/IDs; expand only playlists supplied by the user."""

import argparse
import re
from urllib.parse import parse_qs, urlparse

from vn_av_data.data.source_io import read_rows, unique_rows, write_rows


def video_id(value):
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Expected a YouTube URL or video ID: {value}")
    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.strip("/")
    elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        parts = parsed.path.strip("/").split("/")
        candidate = parse_qs(parsed.query).get("v", [""])[0]
        if len(parts) == 2 and parts[0] in ("shorts", "embed", "live"):
            candidate = parts[1]
    else:
        raise ValueError(f"Expected a YouTube URL: {value}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        raise ValueError(f"No valid video ID: {value}")
    return candidate


def normalize_sources(rows):
    output = []
    for row in rows:
        row = {key: (value or "").strip() for key, value in row.items() if key is not None}
        if not any(row.values()):
            continue
        vid = video_id(row.get("video_id") or row.get("url", ""))
        if row.get("url") and video_id(row["url"]) != vid:
            raise ValueError(f"video_id and URL disagree: {vid}")
        output.append(dict(row, video_id=vid, url="https://www.youtube.com/watch?v=" + vid))
    if not output:
        raise ValueError("Fill the url column in videos.csv before running this step")
    return list(unique_rows(output, "video_id").values())


def collect_sources(rows, playlist_loader=None):
    expanded = []
    for row in rows:
        url = (row.get("url") or "").strip()
        parsed = urlparse(url)
        is_playlist = (
            parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com")
            and parsed.path.rstrip("/") == "/playlist"
            and parse_qs(parsed.query).get("list")
        )
        if not is_playlist:
            expanded.append(row)
            continue
        if row.get("video_id", "").strip():
            raise ValueError("Leave video_id empty for a playlist row")
        if playlist_loader is None:
            import yt_dlp

            with yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "skip_download": True}) as client:
                info = client.extract_info(url, download=False)
        else:
            info = playlist_loader(url)
        entries = list((info or {}).get("entries") or [])
        if not entries:
            raise ValueError(f"Playlist has no readable entries: {url}")
        for item in entries:
            if not item or not item.get("id"):
                raise ValueError(f"Unreadable playlist entry: {url}")
            expanded.append(
                dict(
                    row,
                    video_id=item["id"],
                    url="https://www.youtube.com/watch?v=" + item["id"],
                    title=item.get("title") or "",
                    channel=item.get("channel") or row.get("channel") or "",
                    playlist_url=url,
                )
            )
    normalized = []
    by_id = {}
    # Overlapping playlists may contain the same video. Keep it once, but do not
    # silently discard conflicting manually supplied provenance.
    for row in expanded:
        for item in normalize_sources([row]):
            previous = by_id.get(item["video_id"])
            if previous:
                for key in (
                    "program_id",
                    "episode_id",
                    "canonical_source_id",
                    "tier",
                    "speaker_id",
                ):
                    if previous.get(key) and item.get(key) and previous[key] != item[key]:
                        raise ValueError(f"Conflicting {key} for {item['video_id']}")
                    if item.get(key) and not previous.get(key):
                        previous[key] = item[key]
                continue
            by_id[item["video_id"]] = item
            normalized.append(item)
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = collect_sources(read_rows(args.input))
    write_rows(args.out, rows)
    print(f"{len(rows)} selected videos -> {args.out}; no media downloaded")


if __name__ == "__main__":
    main()
