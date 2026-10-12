"""The analysis run by the web app: default model vs GA-optimised model on held-out modules.

1. split the modules: 70 % for training, 30 % kept aside (stratified);
2. train the default model on the 70 % and score it on the 30 %;
3. run the GA on the 70 % only (inner cross-validation);
4. train the optimised model on the 70 % and score it on the same 30 %;
5. rank every module by risk with out-of-fold predictions (5-fold CV of the optimised model),
   so that no module is scored by a model that saw it during training.

The browser engine (web/js/engine.js) implements the same steps and returns the same JSON.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from .data import Dataset
from .evaluation import stratified_folds, stratified_split, train_predict
from .ga import GeneticOptimizer
from .metrics import evaluate, manual_up, recall_at_effort, scores_to_unit
from .models import MODELS, describe

METRICS = ["f1", "precision", "recall", "mcc", "auc", "pf", "accuracy"]


def _metrics(m: Dict) -> Dict:
    return {k: (None if m.get(k) is None or not np.isfinite(m[k]) else round(float(m[k]), 4)) for k in METRICS}


def _confusion(m: Dict) -> Dict:
    return {k: int(m[k]) for k in ("tp", "fp", "fn", "tn")}


def analyze(ds: Dataset, model: str = "knn", generations: int = 15, population: int = 20,
            test_size: float = 0.3, seed: int = 0,
            progress: Optional[Callable[[Dict], None]] = None) -> Dict:
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}'")
    t0 = time.time()
    say = progress or (lambda event: None)
    say({"stage": "processing", "progress": 0.02, "message": "Splitting the modules (70 % train, 30 % test)"})
    train, test = stratified_split(ds.y, test_size, seed)
    if ds.y[train].sum() < 3 or ds.y[test].sum() < 1:
        raise ValueError("not enough defective modules to train and test (need at least 4)")
    Xtr, ytr, Xte, yte = ds.X[train], ds.y[train], ds.X[test], ds.y[test]
    loc_te = None if ds.loc is None else ds.loc[test]

    say({"stage": "default", "progress": 0.08, "message": "Training the default model"})
    pred, sc = train_predict(model, {}, Xtr, ytr, Xte, None, seed=seed)
    default = evaluate(yte, pred, sc, loc_te)

    def on_generation(h):
        say({"stage": "ga", "progress": 0.1 + 0.8 * h["generation"] / max(1, generations),
             "message": f"Generation {h['generation']}/{generations}", "generation": h["generation"],
             "best": h["best_f1"], "mean": h["mean"], "features": h["features"]})

    ga = GeneticOptimizer(model, pop_size=population, generations=generations, seed=seed).run(
        Xtr, ytr, callback=on_generation)

    say({"stage": "final", "progress": 0.92, "message": "Scoring the optimised model on the held-out modules"})
    pred, sc = train_predict(model, ga.params, Xtr, ytr, Xte, ga.mask, seed=seed)
    optimised = evaluate(yte, pred, sc, loc_te)

    # out-of-fold risk for every module
    risk = np.zeros(ds.n)
    for f, fold in enumerate(stratified_folds(ds.y, 5, seed)):
        tr = np.setdiff1d(np.arange(ds.n), fold)
        _, s = train_predict(model, ga.params, ds.X[tr], ds.y[tr], ds.X[fold], ga.mask, seed=seed + f)
        risk[fold] = s
    risk = scores_to_unit(risk)
    # highest risk first; on equal risk the smaller class first (cheaper to review)
    order = np.lexsort((np.zeros(ds.n) if ds.loc is None else ds.loc, -risk))
    ranking = [{"module": ds.modules[i], "risk": round(float(risk[i]), 4), "actual": int(ds.y[i]),
                "loc": None if ds.loc is None else int(ds.loc[i])} for i in order]

    result = {
        "engine": "python",
        "dataset": ds.summary(),
        "model": model, "model_label": MODELS[model]["label"],
        "split": {"train": int(len(train)), "test": int(len(test)), "test_defective": int(yte.sum()),
                  "seed": seed},
        "default": {"metrics": _metrics(default), "confusion": _confusion(default),
                    "params": describe(model, {}), "features": ds.features},
        "optimized": {"metrics": _metrics(optimised), "confusion": _confusion(optimised),
                      "params": describe(model, ga.params), "features": ga.selected(ds.features)},
        "ga": {"history": ga.history, "evaluations": ga.evaluations, "inner_cv_f1": round(ga.cv_f1, 4),
               "population": population, "generations": generations},
        "ranking": ranking,
        "effort": None if ds.loc is None else {
            "model": round(recall_at_effort(ds.y, risk, ds.loc), 4),
            "smallest_first": round(manual_up(ds.y, ds.loc), 4)},
        "seconds": round(time.time() - t0, 2),
    }
    say({"stage": "done", "progress": 1.0, "message": "Analysis complete"})
    return result
