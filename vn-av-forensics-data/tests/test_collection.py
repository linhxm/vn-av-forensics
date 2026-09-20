from pathlib import Path

import pytest

from vn_av_data.data.acquisition import download_sources
from vn_av_data.data.collect import collect_sources
from vn_av_data.data.curation import curate_sources
from vn_av_data.data.source_io import read_rows, write_rows


def test_playlist_expansion_dedup_and_source_conflicts():
    rows = [
        {"url": "https://youtube.com/playlist?list=one", "speaker_id": "person1"},
        {"url": "https://youtu.be/abcdefghijk", "speaker_id": "person1"},
    ]
    result = collect_sources(rows, lambda _: {"entries": [{"id": "abcdefghijk", "title": "A"}]})
    assert len(result) == 1
    assert result[0]["speaker_id"] == "person1"
    rows[-1]["speaker_id"] = "person2"
    with pytest.raises(ValueError, match="Conflicting speaker_id"):
        collect_sources(rows, lambda _: {"entries": [{"id": "abcdefghijk"}]})
    one = collect_sources(
        [{"url": "https://youtube.com/watch?v=abcdefghijk&list=one"}],
        lambda _: pytest.fail("Must not expand watch URL"),
    )
    assert len(one) == 1


def test_download_snapshot_resume_pending_and_cut_gate(tmp_path, monkeypatch):
    from vn_av_data.data import acquisition

    selected = tmp_path / "selected.csv"
    out = tmp_path / "raw"
    rows = [
        {"url": "https://youtu.be/abcdefghijk", "speaker_id": "one"},
        {"url": "https://youtu.be/lmnopqrstuv", "speaker_id": "two"},
    ]
    write_rows(selected, rows)
    assert download_sources(selected, out, dry_run=True)["selected"] == 2
    assert not out.exists()
    monkeypatch.setattr("shutil.which", lambda _: "node")
    monkeypatch.setattr(acquisition, "probe", lambda _: {"duration_s": 5})
    requests = []

    class Downloader:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def extract_info(self, url, download=True):
            requests.append(url)
            Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(url.encode())
            assert self.opts["js_runtimes"]["node"]["path"] == "node"
            return {"title": "fixture"}

    monkeypatch.setattr("yt_dlp.YoutubeDL", Downloader)
    report = download_sources(selected, out, limit=1)
    assert report["pending"] == 1
    with pytest.raises(ValueError, match="pending/failed"):
        curate_sources(out / "sources.jsonl", tmp_path / "cut", {})
    download_sources(selected, out)
    assert len(requests) == 2
    assert all(r["status"] == "downloaded" for r in read_rows(out / "download_results.csv"))
    download_sources(selected, out)
    assert len(requests) == 2
    rows[0]["speaker_id"] = "changed"
    write_rows(selected, rows, mutable=True)
    with pytest.raises(FileExistsError):
        download_sources(selected, out)
