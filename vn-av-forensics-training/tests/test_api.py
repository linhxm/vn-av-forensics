import time

from fastapi.testclient import TestClient

from vn_av_training.serving.api import create_app


class TestAnalyzer:
    __test__ = False

    def __init__(self, cfg):
        pass

    def analyze(self, path, output, progress):
        progress("model", "fixture only")
        return {"clip_forgery_score": 0.2, "segments": {}, "timeline": []}


def test_jobs_persist_errors_and_no_phantom_timeline(tmp_path):
    cfg = {"jobs_dir": str(tmp_path / "jobs"), "frontend_dist": str(tmp_path / "missing")}
    app = create_app(cfg, TestAnalyzer)
    with TestClient(app) as client:
        assert not client.get("/api/health").json()["ready"]
        assert client.post("/api/jobs", files={"video": ("bad.txt", b"x")}).status_code == 415
        response = client.post("/api/jobs", files={"video": ("sample.mp4", b"fixture-not-a-video")})
        assert response.status_code == 202
        sid = response.json()["id"]
        for _ in range(100):
            job = client.get(f"/api/jobs/{sid}").json()
            if job["status"] == "complete":
                break
            time.sleep(0.01)
        assert job["status"] == "complete"
        assert "input_path" not in job
        assert client.get(f"/api/jobs/{sid}/timeline.csv").status_code == 409
        assert client.get(f"/api/jobs/{sid}/clips/forgery/0").status_code == 404
        assert client.get(f"/api/jobs/{sid}/report").status_code == 200
    with TestClient(create_app(cfg, TestAnalyzer)) as client:
        assert client.get(f"/api/jobs/{sid}").json()["status"] == "complete"


def test_missing_assets_fail_before_upload(tmp_path):
    with TestClient(create_app({"jobs_dir": str(tmp_path / "jobs")})) as client:
        assert client.post("/api/jobs", files={"video": ("sample.mp4", b"x")}).status_code == 409


def test_evidence_download_uses_selected_assessed_or_diagnostic_interval(tmp_path, monkeypatch):
    from vn_av_training.serving import evidence

    class TwoHeadFixture(TestAnalyzer):
        def analyze(self, path, output, progress):
            return {"schema_version": "av-relations-v2", "thresholds": {"lip_audio_mismatch": .5},
                    "suspicious_intervals": [{"relation": "lip_audio_mismatch", "start": 2, "end": 3}],
                    "diagnostic_intervals": [{"relation": "lip_audio_mismatch", "start": 1, "end": 4}]}

    exported = []
    def export(source, folder, index, interval, context):
        exported.append(interval)
        path = folder / "clip.mp4"
        path.write_bytes(b"fixture")
        return path
    monkeypatch.setattr(evidence, "export_clip", export)
    with TestClient(create_app({"jobs_dir": str(tmp_path / "jobs")}, TwoHeadFixture)) as client:
        sid = client.post("/api/jobs", files={"video": ("test.mp4", b"fixture")}).json()["id"]
        for _ in range(100):
            if client.get(f"/api/jobs/{sid}").json()["status"] == "complete":
                break
            time.sleep(.01)
        url = f"/api/jobs/{sid}/clips/lip_audio_mismatch/0"
        assert client.get(url).status_code == 200
        assert client.get(url + "?diagnostic=true").status_code == 200
    assert exported == [[2, 3], [1, 4]]
