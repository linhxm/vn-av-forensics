"""FastAPI service for the TypeScript demo frontend."""

from __future__ import annotations

import copy
import mimetypes
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inference import analyze
from .evidence import CONTEXT_OPTIONS, export_clip

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
CHECKPOINT = ROOT / "checkpoints" / "local.pt"
ASSETS = ROOT / "checkpoints" / "syncnet"
FRONTEND_DIST = ROOT / "frontend" / "dist"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".webm"}

STAGES = [
    ("checkpoint", "Nạp local checkpoint"),
    ("face_tracking", "Face tracking + face crop"),
    ("decode", "Decode frames + PCM audio"),
    ("global_lag", "Global lag normalization"),
    ("feature_extraction", "Frozen SyncNet feature extraction"),
    ("local_scoring", "Local scoring"),
    ("output", "Lưu evidence và kết quả"),
]

app = FastAPI(title="VN-AV Forensics API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _new_stages():
    return [{"id": key, "label": label, "status": "pending", "message": ""} for key, label in STAGES]


def _public(job):
    """Return JSON-safe state; a timeline is intentionally lightweight for browser rendering."""
    payload = {
        "id": job["id"],
        "status": job["status"],
        "message": job["message"],
        "stages": copy.deepcopy(job["stages"]),
        "error": job.get("error"),
    }
    if job.get("result") is not None:
        payload["result"] = job["result"]
        payload["timeline"] = job["timeline"]
        payload["media_url"] = f"/api/jobs/{job['id']}/media/source"
        payload["timeline_csv_url"] = f"/api/jobs/{job['id']}/timeline.csv"
    return payload


def _progress(job_id, stage_id, message):
    with _lock:
        job = _jobs[job_id]
        stage_index = next(i for i, stage in enumerate(job["stages"]) if stage["id"] == stage_id)
        for stage in job["stages"][:stage_index]:
            if stage["status"] == "pending":
                stage["status"] = "complete"
        stage = job["stages"][stage_index]
        stage["status"] = "active"
        stage["message"] = message
        job["message"] = message


def _timeline_records(table):
    records = []
    for row in table.to_dict(orient="records"):
        records.append(
            {
                "time_s": float(row["time_s"]),
                "score": float(row["score"]),
                "valid": bool(row["valid"]),
                "audio_energy": float(row["audio_energy"]),
            }
        )
    return records


def _run_job(job_id):
    try:
        with _lock:
            job = _jobs[job_id]
            job["status"] = "processing"
        result, table = analyze(
            job["input_path"],
            CHECKPOINT,
            job["output_dir"],
            ASSETS,
            device="cuda" if _cuda_available() else "cpu",
            on_progress=lambda stage, message: _progress(job_id, stage, message),
        )
        with _lock:
            job = _jobs[job_id]
            job["status"] = "complete"
            job["message"] = "Hoàn tất phân tích."
            for stage in job["stages"]:
                stage["status"] = "complete"
            job["result"] = result
            job["timeline"] = _timeline_records(table)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        with _lock:
            job = _jobs[job_id]
            job["status"] = "failed"
            for stage in job["stages"]:
                if stage["status"] == "active":
                    stage["status"] = "failed"
            job["error"] = str(exc)
            job["message"] = "Không thể hoàn tất phân tích."


def _cuda_available():
    import torch

    return torch.cuda.is_available()


@app.get("/api/health")
def health():
    return {"checkpoint_ready": CHECKPOINT.is_file()}


@app.post("/api/jobs", status_code=202)
async def create_job(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    if not CHECKPOINT.is_file():
        raise HTTPException(
            status_code=409,
            detail="Thiếu checkpoints/local.pt. Hãy train trên Kaggle và tải checkpoint về trước.",
        )
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Định dạng video không được hỗ trợ.")
    job_id = uuid.uuid4().hex
    output_dir = OUTPUTS / job_id
    output_dir.mkdir(parents=True, exist_ok=False)
    input_path = output_dir / f"input{suffix}"
    with input_path.open("wb") as destination:
        while content := await video.read(1024 * 1024):
            destination.write(content)
    await video.close()
    job = {
        "id": job_id,
        "status": "queued",
        "message": "Video đã nhận, đang chờ phân tích.",
        "stages": _new_stages(),
        "input_path": input_path,
        "output_dir": output_dir,
        "result": None,
        "timeline": None,
        "error": None,
    }
    with _lock:
        _jobs[job_id] = job
    background_tasks.add_task(_run_job, job_id)
    return _public(job)


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy analysis job.")
        return _public(job)


@app.get("/api/jobs/{job_id}/media/source")
def source_media(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy analysis job.")
        path = job["input_path"]
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, media_type=media_type or "application/octet-stream")


@app.get("/api/jobs/{job_id}/clips/{index}")
def download_clip(job_id: str, index: int, context: float = Query(default=0.75)):
    if context not in CONTEXT_OPTIONS:
        raise HTTPException(status_code=422, detail="Context must be 0, 0.75 or 1.5 seconds.")
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        if job["status"] != "complete":
            raise HTTPException(status_code=409, detail="Analysis is not complete.")
        intervals = job["result"]["intervals"]
        if index < 0 or index >= len(intervals):
            raise HTTPException(status_code=404, detail="Evidence interval not found.")
        interval = intervals[index]
        source, output_dir = job["input_path"], job["output_dir"]
    try:
        path = export_clip(source, output_dir, index, interval, context)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Clip export timed out. Please retry.") from exc
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not export this clip. Check the source video and FFmpeg installation.",
        ) from exc
    return FileResponse(path, media_type="video/mp4", filename=f"evidence-{job_id[:8]}-{index + 1}.mp4")


@app.get("/api/jobs/{job_id}/timeline.csv")
def download_timeline(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy analysis job.")
        path = job["output_dir"] / "timeline.csv"
    if not path.is_file():
        raise HTTPException(status_code=409, detail="Timeline chưa sẵn sàng.")
    return FileResponse(path, media_type="text/csv", filename="timeline.csv")


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/")
    def frontend_index():
        return FileResponse(FRONTEND_DIST / "index.html")

else:

    @app.get("/")
    def frontend_not_built():
        return {
            "detail": "Frontend chưa được build. Chạy `npm install && npm run build` trong frontend/."
        }
