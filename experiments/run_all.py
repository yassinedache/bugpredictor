"""Run every experiment of the README and regenerate results/ (tables, figures, results.md).

    python experiments/run_all.py                 # everything (about 1 h 30 on 2 cores)
    python experiments/run_all.py baseline report # only some stages
    python experiments/run_all.py ga --jobs 4     # parallel GA runs

Raw results are cached in results/raw/*.json; delete a file (or pass --force) to recompute it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per process: the GA runs are parallelised instead
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bugpredictor import BENCHMARK, GeneticOptimizer, load_promise  # noqa: E402
from bugpredictor.data import previous_releases  # noqa: E402
from bugpredictor.evaluation import (concat, cross_validate, evaluate_transfer,  # noqa: E402
                                     leaky_cross_validate, stratified_folds, train_predict)
from bugpredictor.metrics import evaluate, manual_up  # noqa: E402
from bugpredictor.preprocessing import Preprocessor  # noqa: E402
from bugpredictor.smote import smote  # noqa: E402

RAW = ROOT / "results" / "raw"
MODELS = ["knn", "lr", "nb", "svm", "rf"]
GA_MODELS = ["knn", "svm", "lr"]
KEYS = ["f1", "mcc", "auc", "recall_20", "precision", "recall", "pf"]


def save(name, obj):
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def cached(name, force):
    p = RAW / f"{name}.json"
    return None if force or not p.exists() else json.loads(p.read_text())


def slim(m):
    return {k: round(float(m[k]), 4) for k in KEYS if k in m}


# --------------------------------------------------------------------------- stages
def stage_baseline(force):
    """Default settings of five classifiers, 5 x 5-fold CV, SMOTE inside the training folds."""
    if cached("baseline", force):
        return
    out = {}
    for r in BENCHMARK:
        d = load_promise(r)
        out[r] = {"summary": d.summary(), "manual_up": manual_up(d.y, d.loc)}
        for m in MODELS:
            res = cross_validate(d, m, k=5, repeats=5, seed=0)
            out[r][m] = {"mean": slim(res["mean"]),
                         "std": {k: round(float(np.nanstd([f[k] for f in res["folds"]])), 4) for k in KEYS}}
        print(f"baseline {r}: " + "  ".join(f"{m} {out[r][m]['mean']['f1']:.3f}" for m in MODELS), flush=True)
    save("baseline", out)


def stage_ablation(force):
    """Effect of the log transform and of SMOTE (both applied inside the folds)."""
    if cached("ablation", force):
        return
    out = {}
    for r in BENCHMARK:
        d = load_promise(r)
        out[r] = {}
        for m in ["knn", "lr", "nb", "svm"]:
            out[r][m] = {name: slim(cross_validate(d, m, k=5, repeats=3, seed=0, log=lg, use_smote=sm)["mean"])
                         for name, lg, sm in [("full", True, True), ("no_log", False, True), ("no_smote", True, False)]}
        print(f"ablation {r} done", flush=True)
    save("ablation", out)


def ga_task(args):
    model, release = args
    name = f"ga_{model}_{release}"
    d = load_promise(release)
    folds = []
    t0 = time.time()
    for f, test in enumerate(stratified_folds(d.y, 5, 0)):
        train = np.setdiff1d(np.arange(d.n), test)
        pred, sc = train_predict(model, {}, d.X[train], d.y[train], d.X[test], None, seed=f)
        default = evaluate(d.y[test], pred, sc, d.loc[test])
        ga = GeneticOptimizer(model, seed=f).run(d.X[train], d.y[train])
        pred, sc = train_predict(model, ga.params, d.X[train], d.y[train], d.X[test], ga.mask, seed=f)
        folds.append({"default": slim(default), "ga": slim(evaluate(d.y[test], pred, sc, d.loc[test])),
                      "inner_cv_f1": round(ga.cv_f1, 4), "params": ga.params,
                      "features": ga.selected(d.features), "history": ga.history, "evaluations": ga.evaluations})
    res = {"model": model, "release": release, "folds": folds, "seconds": round(time.time() - t0, 1)}
    save(name, res)
    mean = lambda key, side: np.mean([f[side][key] for f in folds])  # noqa: E731
    print(f"GA {model:3s} {release:12s} default F1 {mean('f1', 'default'):.3f} -> GA {mean('f1', 'ga'):.3f} "
          f"({np.mean([len(f['features']) for f in folds]):.1f} features, {res['seconds']:.0f} s)", flush=True)
    return name


def stage_ga(force, jobs, models=None):
    """Nested CV: the GA runs on the training part of each outer fold, the outer test fold scores it."""
    tasks = [(m, r) for m in (models or GA_MODELS) for r in BENCHMARK
             if force or not (RAW / f"ga_{m}_{r}.json").exists()]
    # longest tasks first so that the pool finishes together
    tasks.sort(key=lambda t: {"svm": 0, "knn": 1, "lr": 2}[t[0]])
    if not tasks:
        return
    if jobs > 1:
        with Pool(jobs) as pool:
            list(pool.imap_unordered(ga_task, tasks))
    else:
        for t in tasks:
            ga_task(t)


def stage_leakage(force):
    """Reproduce the original protocol (SMOTE before CV) and compare it with the corrected one."""
    if cached("leakage", force):
        return
    out = {}
    for r in BENCHMARK:
        d = load_promise(r)
        row = {}
        for k in (1, 5):
            p = {"n_neighbors": k}
            row[f"knn{k}_leaky"] = slim(leaky_cross_validate(d, "knn", p, k=5, seed=0)["mean"])
            row[f"knn{k}_honest"] = slim(cross_validate(d, "knn", p, k=5, repeats=5, seed=0, log=False)["mean"])
        # the original GA: standardise + SMOTE on the whole dataset, then GA scored by CV on the resampled rows
        Xr, yr = smote(Preprocessor(log=False).fit_transform(d.X), d.y, rng=0)
        ga = GeneticOptimizer("knn", use_smote=False, log=False, seed=0).run(Xr, yr)
        row["ga_leaky_reported"] = round(ga.cv_f1, 4)
        row["ga_leaky_params"] = ga.params
        out[r] = row
        print(f"leakage {r}: k=1 leaky {row['knn1_leaky']['f1']:.3f} vs honest {row['knn1_honest']['f1']:.3f}; "
              f"GA reported {row['ga_leaky_reported']:.3f}", flush=True)
    save("leakage", out)


def stage_transfer(force):
    """Cross-version (train on the previous release) and cross-project (train on the 10 other projects)."""
    if cached("transfer", force):
        return
    data = {r: load_promise(r) for r in BENCHMARK}
    out = {}
    for r in BENCHMARK:
        prev = previous_releases(r)[-1]
        others = concat([data[o] for o in BENCHMARK if o != r], "others")
        old = load_promise(prev)
        out[r] = {"previous": prev, "cross_version": {}, "cross_project": {}}
        for m in MODELS:
            out[r]["cross_version"][m] = slim(evaluate_transfer(old, data[r], m, seed=0))
            out[r]["cross_project"][m] = slim(evaluate_transfer(others, data[r], m, seed=0))
        print(f"transfer {r} (from {prev}) done", flush=True)
    save("transfer", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["baseline", "ablation", "leakage", "transfer", "ga", "report"])
    ap.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", default=",".join(GA_MODELS), help="models optimised by the GA stage")
    a = ap.parse_args()
    for s in a.stages:
        t = time.time()
        if s == "ga":
            stage_ga(a.force, a.jobs, a.models.split(","))
        elif s == "report":
            from report import make_report  # noqa: E402
            make_report()
        else:
            globals()[f"stage_{s}"](a.force)
        print(f"== {s} finished in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
