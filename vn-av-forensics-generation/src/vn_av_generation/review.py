"""Audit generated events; save labels separately from immutable media."""

import csv
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from vn_av_generation.data.manifest import read_manifest
from vn_av_generation.generate import csv_write

PAGE = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Duyệt bất nhất</title>
<style>body{max-width:1000px;margin:24px auto;font:16px system-ui}video{width:100%;max-height:55vh}button,input{padding:8px;margin:5px}pre{white-space:pre-wrap}</style>
<h1>Duyệt bất nhất môi–âm thanh</h1><p>positive = audio không tương thích với chuyển động môi, không giải thích được chỉ bằng dịch thời gian hợp lý; negative = đã kiểm tra và tương thích; uncertain = không đủ bằng chứng. Không gán nhãn chỉ dựa vào tên generator. Xem/nghe rồi sửa khoảng quan sát được trước khi lưu. Không xác minh danh tính hay nhận dạng âm vị cụ thể.</p>
<p id="info"></p><video id="v" controls></video><pre id="meta"></pre>
<label>Từ <input id="start" type="number" step="0.01"></label><label>Đến <input id="end" type="number" step="0.01"></label>
<p><button onclick="move(-1)">Trước</button><button onclick="save('positive')">Có bất nhất</button><button onclick="save('negative')">Khớp</button><button onclick="save('uncertain')">Không chắc</button><button onclick="move(1)">Sau</button></p><p id="status"></p>
<script>let rows=[],i=0;const $=id=>document.getElementById(id);function show(){if(!rows.length){$('info').textContent='Không có sự kiện cần duyệt';return}let r=rows[i];$('info').textContent=`${i+1}/${rows.length} · ${r.head} · ${r.decision}`;$('v').src='/media/'+i;$('start').value=r.start;$('end').value=r.end;$('meta').textContent=JSON.stringify(r,null,2)}function move(d){i=Math.max(0,Math.min(rows.length-1,i+d));show()}async function save(decision){if(!rows.length)return;let r=await fetch('/review/'+i,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision,start:Number($('start').value),end:Number($('end').value)})});if(!r.ok){$('status').textContent=await r.text();return}rows[i]={...rows[i],...await r.json()};$('status').textContent='Đã lưu';move(1)}fetch('/reviews').then(r=>r.json()).then(r=>{rows=r;show()}).catch(e=>$('status').textContent=e.message)</script></html>"""


class Decision(BaseModel):
    decision: str
    start: float
    end: float


def create_app(dataset):
    root = Path(dataset).resolve()
    path = root / "review.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    media = {r["sample_id"]: r for r in read_manifest(root / "manifest.jsonl")}
    app = FastAPI()
    lock = threading.Lock()

    def entry(index):
        if not 0 <= index < len(rows):
            raise HTTPException(404, "Unknown review")
        return rows[index]

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/reviews")
    def reviews():
        return rows

    @app.get("/media/{index}")
    def video(index: int):
        row = media[entry(index)["sample_id"]]
        target = (root / row["video"]).resolve()
        if not target.is_relative_to(root):
            raise HTTPException(400, "Invalid media path")
        return FileResponse(target)

    @app.post("/review/{index}")
    def update(index: int, value: Decision):
        row = entry(index)
        if (
            value.decision not in {"positive", "negative", "uncertain"}
            or not 0 <= value.start < value.end <= media[row["sample_id"]]["duration_s"]
        ):
            raise HTTPException(422, "Invalid decision or interval")
        with lock:
            row.update(decision=value.decision, start=value.start, end=value.end)
            csv_write(path, rows)
        return row

    return app
