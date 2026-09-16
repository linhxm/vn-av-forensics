"""Local review of candidate clips. Decisions persist to the explicit review CSV."""

import csv
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from vn_av_data.data.curation import write_csv

PAGE = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Duyệt clip</title>
<style>body{max-width:950px;margin:32px auto;font:17px system-ui;background:#101b28;color:#edf4ff}
video{width:100%;max-height:60vh;background:black}button{padding:12px;margin:8px;border-radius:8px}
select{padding:8px}p{line-height:1.5}</style><h1>Duyệt clip môi–tiếng</h1>
<p>Giữ = một người, miệng thấy rõ, âm thanh thuộc người trong hình và khớp thời gian.
Nếu chưa chắc, chọn Chưa rõ. Nguồn YouTube thật chưa tự bảo đảm đồng bộ.</p>
<p id="position"></p><video id="video" controls></video><p id="info"></p>
<button onclick="move(-1)">← Trước</button><button onclick="save('keep')">Giữ</button>
<button onclick="save('reject')">Loại</button><button onclick="save('uncertain')">Chưa rõ</button>
<button onclick="move(1)">Sau →</button><p id="status"></p>
<script>let rows=[],i=0;const $=id=>document.getElementById(id);
async function load(){const r=await fetch('/api/clips');rows=await r.json();show()}
function show(){if(!rows.length){$('info').textContent='Không có clip';return}
let r=rows[i];$('position').textContent=`${i+1}/${rows.length} — ${r.clip_id}`;
$('info').textContent=`Nguồn: ${r.source_id||''}; ${r.source_start_s||0}–${r.source_end_s||'?'} giây; nhãn: ${r.decision}`;
$('video').src='/api/media/'+i}
function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}
async function save(decision){if(!rows.length)return;let r=await fetch('/api/clips/'+i,{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify({decision})});
if(!r.ok){$('status').textContent=await r.text();return}rows[i].decision=decision;
$('status').textContent='Đã lưu CSV';move(1)}load().catch(e=>$('status').textContent=e.message);</script></html>"""


class Decision(BaseModel):
    decision: str


def create_review_app(review, root):
    review, root = Path(review).resolve(), Path(root).resolve()
    with review.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths = []
    for row in rows:
        path = root / row["file_path"].replace("\\", "/")
        if not path.is_file():
            path = root / row["file_path"].replace("\\", "/").rsplit("/", 1)[-1]
        path = path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Review media missing/outside root: {path}")
        paths.append(path)
    app, lock = FastAPI(), threading.Lock()

    def check(index):
        if not 0 <= index < len(rows):
            raise HTTPException(404, "Clip not found")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/api/clips")
    def clips():
        return rows

    @app.get("/api/media/{index}")
    def media(index: int):
        check(index)
        return FileResponse(paths[index])

    @app.post("/api/clips/{index}")
    def update(index: int, value: Decision):
        check(index)
        if value.decision not in {"keep", "reject", "uncertain"}:
            raise HTTPException(422, "Invalid decision")
        with lock:
            rows[index]["decision"] = value.decision
            if "sync_status" in rows[index]:
                rows[index]["sync_status"] = (
                    "reviewed_match" if value.decision == "keep" else "unverified"
                )
            write_csv(review, rows)
        return {"saved": True}

    return app
