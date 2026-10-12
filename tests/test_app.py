import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402
from bugpredictor import load_promise  # noqa: E402
from bugpredictor.analysis import analyze  # noqa: E402

client = TestClient(app)


def test_analysis_result_schema():
    d = load_promise("log4j-1.1")
    events = []
    r = analyze(d, "lr", generations=2, population=6, seed=1, progress=events.append)
    assert r["split"]["train"] + r["split"]["test"] == d.n
    assert set(r["default"]["metrics"]) >= {"f1", "precision", "recall", "mcc", "auc"}
    assert len(r["ranking"]) == d.n and sorted(x["module"] for x in r["ranking"]) == sorted(d.modules)
    risks = [x["risk"] for x in r["ranking"]]
    assert risks == sorted(risks, reverse=True)
    assert 1 <= len(r["optimized"]["features"]) <= len(d.features)
    assert events[0]["stage"] == "processing" and events[-1]["stage"] == "done"


def test_static_app_and_config():
    assert client.get("/").status_code == 200
    assert "BugPredictor" in client.get("/").text
    assert client.get("/config.js").text.strip() == "window.BP_API = '/api';"
    assert client.get("/data/manifest.json").json()["datasets"][0]["name"] == "ant-1.7"


def test_health_and_datasets():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "svm" in h["models"]
    ds = client.get("/api/datasets").json()["datasets"]
    assert len(ds) == 41 and sum(d["benchmark"] for d in ds) == 11


def test_upload_analyze_and_download():
    csv = "name,wmc,cbo,loc,bug\n" + "\n".join(
        f"C{i},{i % 7},{(3 * i) % 11},{20 + 13 * i},{int(i % 4 == 0)}" for i in range(80))
    up = client.post("/api/upload", files={"file": ("demo.csv", csv, "text/csv")})
    assert up.status_code == 200, up.text
    info = up.json()
    assert info["modules"] == 80 and info["defective"] == 20 and info["features"] == 3
    job = client.post("/api/analyze", json={"dataset": info["id"], "model": "knn", "generations": 2, "population": 6}).json()["job"]
    for _ in range(300):
        st = client.get(f"/api/jobs/{job}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    assert st["result"]["dataset"]["name"] == "demo"
    out = client.get(f"/api/jobs/{job}/predictions.csv")
    assert out.status_code == 200 and out.text.startswith("rank,module,risk,loc,actually_defective") and out.text.count("\n") == 81


def test_bad_requests():
    assert client.post("/api/upload", files={"file": ("x.csv", "a,bug\n1,0\n2,0\n", "text/csv")}).status_code == 400
    assert client.post("/api/analyze", json={"dataset": "nope-1.0"}).status_code == 404
    assert client.post("/api/analyze", json={"dataset": "ant-1.7", "model": "xgb"}).status_code == 400
    assert client.get("/api/jobs/unknown").status_code == 404
    # the old API accepted pickled models (arbitrary code execution); there is no such endpoint any more
    assert client.post("/models/knn", files={"file": ("m.pkl", b"x")}).status_code in (404, 405)
