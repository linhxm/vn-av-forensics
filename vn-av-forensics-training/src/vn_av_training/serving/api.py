"""Persistent, bounded local demo jobs. A single worker owns each ML pipeline."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from vn_av_training.common.runtime import write_json


def create_app(cfg, analyzer_factory=None):
    root = Path(cfg.get("jobs_dir", "outputs/jobs")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    db = root / "jobs.sqlite3"

    def connection():
        return sqlite3.connect(db, timeout=30)

    with connection() as c:
        c.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        for sid, payload in c.execute("SELECT id,payload FROM jobs").fetchall():
            item = json.loads(payload)
            if item["status"] in ("queued", "processing"):
                item.update(
                    status="interrupted", message="Server restarted; submit the video again"
                )
                c.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(item), sid))

    def put(item):
        with connection() as c:
            c.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?,?)",
                (item["id"], json.dumps(item, ensure_ascii=False)),
            )

    def get(sid):
        if len(sid) != 32 or any(c not in "0123456789abcdef" for c in sid):
            raise HTTPException(404, "Job not found")
        with connection() as c:
            r = c.execute("SELECT payload FROM jobs WHERE id=?", (sid,)).fetchone()
        if not r:
            raise HTTPException(404, "Job not found")
        return json.loads(r[0])

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vn-av")
    slots = threading.BoundedSemaphore(int(cfg.get("max_queue", 3)))
    state = {"analyzer": None}

    @asynccontextmanager
    async def lifespan(app):
        yield
        executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="Vietnamese AV Forensics", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def process(sid):
        item = get(sid)
        try:
            item.update(status="processing", stage="checkpoint", message="Đang nạp pipeline")
            item.setdefault("steps", []).append(
                {
                    "stage": "checkpoint",
                    "message": "Đang nạp checkpoint và backbone",
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            put(item)
            if state["analyzer"] is None:
                if analyzer_factory:
                    state["analyzer"] = analyzer_factory(cfg)
                else:
                    from vn_av_training.serving.relations import RelationAnalyzer as Analyzer

                    state["analyzer"] = Analyzer(cfg)

            def progress(stage, message):
                item.update(stage=stage, message=message)
                item.setdefault("steps", []).append(
                    {
                        "stage": stage,
                        "message": message,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                put(item)

            result = state["analyzer"].analyze(item["input_path"], root / sid, progress)
            item.update(status="complete", stage="complete", message="Hoàn tất", result=result)
        except Exception as exc:  # noqa: BLE001 -- persist failure at the background job boundary
            item.update(status="failed", message="Không thể hoàn tất phân tích", error=str(exc))
        finally:
            put(item)
            slots.release()

    def public(item):
        return {k: v for k, v in item.items() if k != "input_path"}

    @app.get("/api/health")
    def health():
        from vn_av_training.serving.relations import relation_health

        return {**relation_health(cfg), "model_loaded": state["analyzer"] is not None}

    @app.post("/api/jobs", status_code=202)
    async def upload(video: UploadFile = File(...)):  # noqa: B008 -- FastAPI dependency declaration
        if not analyzer_factory and not health()["ready"]:
            raise HTTPException(
                409, "Missing model assets: see README.md and run doctor --stage inference"
            )
        ext = Path(video.filename or "").suffix.lower()
        if ext not in {".mp4", ".webm", ".mov", ".mkv", ".avi", ".mpg"}:
            raise HTTPException(415, "Unsupported video format")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "Analysis queue is full; retry later")
        sid = uuid.uuid4().hex
        folder = root / sid
        folder.mkdir()
        path = folder / ("input" + ext)
        total = 0
        try:
            with path.open("wb") as f:
                while chunk := await video.read(1024 * 1024):
                    total += len(chunk)
                    if total > int(cfg.get("max_upload_mb", 200)) * 1024 * 1024:
                        raise HTTPException(413, "Video exceeds upload limit")
                    f.write(chunk)
            if total == 0:
                raise HTTPException(422, "Empty upload")
            item = {
                "id": sid,
                "status": "queued",
                "stage": "queued",
                "message": "Đang chờ phân tích",
                "filename": video.filename,
                "input_path": str(path),
                "result": None,
                "error": None,
                "steps": [],
            }
            put(item)
            executor.submit(process, sid)
            return public(item)
        except Exception:
            path.unlink(missing_ok=True)
            slots.release()
            raise
        finally:
            await video.close()

    @app.get("/api/jobs/{sid}")
    def status(sid: str):
        return public(get(sid))

    @app.get("/api/jobs/{sid}/media")
    def media(sid: str):
        return FileResponse(get(sid)["input_path"])

    @app.get("/api/jobs/{sid}/report")
    def report(sid: str):
        item = get(sid)
        if item["status"] != "complete":
            raise HTTPException(409, "Result not ready")
        path = root / sid / "result.json"
        if not path.is_file():
            write_json(path, item["result"])
        return FileResponse(path, filename="analysis.json", media_type="application/json")

    @app.get("/api/jobs/{sid}/timeline.csv")
    def timeline(sid: str):
        get(sid)
        path = root / sid / "timeline.csv"
        if not path.is_file():
            raise HTTPException(409, "This model has no temporal output")
        return FileResponse(path, filename="timeline.csv")

    @app.get("/api/jobs/{sid}/evidence/{filename}")
    def evidence(sid: str, filename: str):
        item = get(sid)
        if item["status"] != "complete":
            raise HTTPException(409, "Evidence not ready")
        if (
            not filename.startswith("frame-")
            or not filename.endswith(".jpg")
            or "/" in filename
            or "\\" in filename
        ):
            raise HTTPException(404, "Unknown evidence")
        path = (root / sid / "evidence" / filename).resolve()
        if not path.is_relative_to((root / sid / "evidence").resolve()) or not path.is_file():
            raise HTTPException(404, "Missing evidence")
        return FileResponse(path)

    @app.get("/api/jobs/{sid}/clips/{head}/{index}")
    def clip(sid: str, head: str, index: int, context: float = 0.75, diagnostic: bool = False):
        item = get(sid)
        if item["status"] != "complete":
            raise HTTPException(409, "Result not ready")
        segments = [
            span
            for span in item["result"].get(
                "diagnostic_intervals" if diagnostic else "suspicious_intervals", []
            )
            if span["relation"] == head
        ]
        allowed = item["result"].get("thresholds", {})
        if head not in allowed or not 0 <= index < len(segments):
            raise HTTPException(404, "Segment not found")
        span = segments[index]
        from vn_av_training.serving.evidence import export_clip

        dest = root / sid / (head + ("-diagnostic" if diagnostic else "-assessed"))
        dest.mkdir(exist_ok=True)
        try:
            path = export_clip(
                Path(item["input_path"]),
                dest,
                index,
                [span["start"], span["end"]],
                context,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return FileResponse(path, filename=f"{head}-{index + 1}.mp4", media_type="video/mp4")

    dist = Path(cfg.get("frontend_dist", "demo/dist"))
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        from vn_av_training.serving.demo import PAGE

        @app.get("/", response_class=HTMLResponse)
        def built_in_demo():
            return PAGE

    return app
