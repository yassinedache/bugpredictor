"""BugPredictor web server: serves the web app and runs analyses with scikit-learn.

    uvicorn app.server:app --reload          # then open http://127.0.0.1:8000

The same web app also works without this server (static hosting): it then runs a
JavaScript port of the pipeline in the browser. Served from here, it uses this Python
engine (scikit-learn).

Security: uploaded files are parsed as CSV/ARFF text only. Unlike the first version of
the project, the server never unpickles uploaded files (pickle can execute arbitrary code).
"""

from __future__ import annotations

import io
import csv
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bugpredictor import __version__
from bugpredictor.analysis import analyze
from bugpredictor.data import BENCHMARK, list_releases, load, load_promise
from bugpredictor.models import MODELS

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
MAX_UPLOAD = 10 * 1024 * 1024
MAX_ROWS = 20000

app = FastAPI(title="BugPredictor API", version=__version__,
              description="Software defect prediction with a genetic algorithm (leakage-free evaluation).")
uploads: "OrderedDict[str, object]" = OrderedDict()
jobs: "OrderedDict[str, Dict]" = OrderedDict()
lock = threading.Lock()
pool = ThreadPoolExecutor(max_workers=2)


class AnalyzeRequest(BaseModel):
    dataset: str = Field(..., description="a PROMISE release (e.g. 'ant-1.7') or an upload id")
    model: str = Field("knn", description="knn, svm, lr, nb or rf")
    generations: int = Field(15, ge=1, le=60)
    population: int = Field(20, ge=6, le=60)
    seed: int = Field(0, ge=0, le=10**6)


def _remember(store: OrderedDict, key: str, value, limit: int = 50):
    with lock:
        store[key] = value
        while len(store) > limit:
            store.popitem(last=False)


def _dataset(name: str):
    if name in uploads:
        return uploads[name]
    if name in list_releases():
        return load_promise(name)
    raise HTTPException(404, f"unknown dataset '{name}'")


@app.get("/config.js", include_in_schema=False)
def config_js():
    return Response("window.BP_API = '/api';\n", media_type="application/javascript")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "engine": "python",
            "models": {k: v["label"] for k, v in MODELS.items()}}


@app.get("/api/datasets")
def datasets():
    out = []
    for r in list_releases():
        d = load_promise(r)
        out.append({**d.summary(), "benchmark": r in BENCHMARK})
    return {"datasets": out}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (10 MB maximum)")
    name = Path(file.filename or "uploaded").stem[:60] or "uploaded"
    try:
        ds = load(data, name=name)
    except (ValueError, UnicodeError) as e:
        raise HTTPException(400, f"could not read the file: {e}")
    if ds.n > MAX_ROWS:
        raise HTTPException(400, f"too many rows ({ds.n}); the limit is {MAX_ROWS}")
    uid = "upload-" + uuid.uuid4().hex[:12]
    _remember(uploads, uid, ds, limit=20)
    return {"id": uid, **ds.summary()}


def _run(job_id: str, req: AnalyzeRequest, ds):
    job = jobs[job_id]

    def progress(event):
        job["progress"] = event.get("progress", job["progress"])
        job["stage"] = event.get("stage", job["stage"])
        if event.get("stage") == "ga":
            job["history"].append({k: event[k] for k in ("generation", "best", "mean", "features")})

    try:
        job["result"] = analyze(ds, req.model, req.generations, req.population, seed=req.seed, progress=progress)
        job["status"] = "done"
    except Exception as e:  # reported to the client
        job["status"], job["error"] = "error", str(e)
    job["finished"] = time.time()


@app.post("/api/analyze")
def start(req: AnalyzeRequest):
    if req.model not in MODELS:
        raise HTTPException(400, f"unknown model '{req.model}'")
    ds = _dataset(req.dataset)
    job_id = uuid.uuid4().hex[:16]
    _remember(jobs, job_id, {"status": "running", "stage": "processing", "progress": 0.0,
                             "history": [], "started": time.time()})
    pool.submit(_run, job_id, req, ds)
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, ranking: int = 50):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    out = {k: job.get(k) for k in ("status", "stage", "progress", "history", "error")}
    if job.get("result"):
        res = dict(job["result"])
        res["ranking_total"] = len(res["ranking"])
        res["ranking"] = res["ranking"][:max(0, ranking)]
        out["result"] = res
    return out


@app.get("/api/jobs/{job_id}/predictions.csv")
def predictions(job_id: str):
    job = jobs.get(job_id)
    if job is None or not job.get("result"):
        raise HTTPException(404, "no finished analysis with this id")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "module", "risk", "loc", "actually_defective"])
    for i, r in enumerate(job["result"]["ranking"], 1):
        w.writerow([i, r["module"], r["risk"], r["loc"], r["actual"]])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=bugpredictor-risk.csv"})


# the web app itself (index.html, js, css, assets, data) -- mounted last
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
