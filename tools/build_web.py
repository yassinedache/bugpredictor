"""Prepare web/data for the web app: the 11 benchmark releases, a manifest and the research summary.

    python tools/build_web.py

web/ is then a self-contained static site (it can be hosted anywhere, e.g. on Vercel).
research.json is built from results/raw (run experiments/run_all.py first).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bugpredictor.data import BENCHMARK, load_promise, release_path  # noqa: E402

WEB_DATA = ROOT / "web" / "data"
RAW = ROOT / "results" / "raw"


def build_datasets():
    (WEB_DATA / "promise").mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in BENCHMARK:
        shutil.copyfile(release_path(r), WEB_DATA / "promise" / f"{r}.csv")
        manifest.append(load_promise(r).summary())
    (WEB_DATA / "manifest.json").write_text(json.dumps({"datasets": manifest}, indent=1))
    print(f"web/data: {len(manifest)} releases")


def load(name):
    p = RAW / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def build_research():
    base, leak, transfer = load("baseline"), load("leakage"), load("transfer")
    ga = {m: {r: load(f"ga_{m}_{r}") for r in BENCHMARK} for m in ("knn", "svm", "lr")}
    if not (base and leak and transfer) or any(v is None for d in ga.values() for v in d.values()):
        print("research.json not written: some raw results are missing")
        return
    models = ["knn", "lr", "nb", "svm", "rf"]
    mean = lambda xs: float(np.mean(xs))  # noqa: E731
    gmean = lambda m, r, side, k="f1": mean([f[side][k] for f in ga[m][r]["folds"]])  # noqa: E731
    rows = [{"release": r, "leaky_ga": leak[r]["ga_leaky_reported"], "honest_ga": round(gmean("knn", r, "ga"), 4),
             "honest_default": round(gmean("knn", r, "default"), 4), "knn1_leaky": leak[r]["knn1_leaky"]["f1"],
             "knn1_honest": leak[r]["knn1_honest"]["f1"]} for r in BENCHMARK]
    research = {
        "benchmark": BENCHMARK,
        "leakage": {"rows": rows, "mean": {k: round(mean([x[k] for x in rows]), 4) for k in rows[0] if k != "release"}},
        "ga": [{"model": m, "default_f1": round(mean([gmean(m, r, "default") for r in BENCHMARK]), 4),
                "ga_f1": round(mean([gmean(m, r, "ga") for r in BENCHMARK]), 4),
                "wins": sum(gmean(m, r, "ga") > gmean(m, r, "default") for r in BENCHMARK), "n": len(BENCHMARK),
                "features": round(mean([len(f["features"]) for r in BENCHMARK for f in ga[m][r]["folds"]]), 2),
                "total": len(load_promise(BENCHMARK[0]).features),
                "inner_cv": round(mean([f["inner_cv_f1"] for r in BENCHMARK for f in ga[m][r]["folds"]]), 4)}
               for m in ("knn", "svm", "lr")],
        "baseline": {"models": models,
                     "mean": {k: {m: round(mean([base[r][m]["mean"][k] for r in BENCHMARK]), 4) for m in models}
                              for k in ("f1", "mcc", "auc", "recall_20")},
                     "manual_up": round(mean([base[r]["manual_up"] for r in BENCHMARK]), 4)},
        "transfer": [{"model": m, "within": round(mean([base[r][m]["mean"]["f1"] for r in BENCHMARK]), 4),
                      "cross_version": round(mean([transfer[r]["cross_version"][m]["f1"] for r in BENCHMARK]), 4),
                      "cross_project": round(mean([transfer[r]["cross_project"][m]["f1"] for r in BENCHMARK]), 4)}
                     for m in models],
    }
    (WEB_DATA / "research.json").write_text(json.dumps(research, indent=1))
    print("web/data/research.json written")


def build_images():
    src, dst = ROOT / "web" / "assets" / "SDP.png", ROOT / "web" / "assets" / "SDP.webp"
    if src.exists() and (not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime):
        from PIL import Image
        Image.open(src).save(dst, "WEBP", quality=88, method=6)
        print(f"{dst.name}: {dst.stat().st_size // 1024} KB")


if __name__ == "__main__":
    build_datasets()
    build_images()
    build_research()
