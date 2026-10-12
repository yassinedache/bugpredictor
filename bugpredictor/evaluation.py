"""Leakage-free evaluation: every fitted step (preprocessing, SMOTE, model) sees training rows only."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import Dataset
from .metrics import evaluate, mean_metrics
from .models import fit_model, risk_scores
from .preprocessing import Preprocessor
from .smote import smote as smote_fn


def stratified_folds(y: np.ndarray, k: int, rng) -> List[np.ndarray]:
    """Split row indices into k folds that keep the share of defective modules."""
    rng = np.random.default_rng(rng)
    folds = [[] for _ in range(k)]
    offset = 0
    for c in (0, 1):
        idx = np.flatnonzero(np.asarray(y) == c)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[(i + offset) % k].append(j)
        offset += len(idx)
    return [np.sort(np.array(f, dtype=int)) for f in folds]


def stratified_split(y: np.ndarray, test_size: float, rng):
    """One stratified train/test split; returns (train_idx, test_idx)."""
    rng = np.random.default_rng(rng)
    test = []
    for c in (0, 1):
        idx = np.flatnonzero(np.asarray(y) == c)
        rng.shuffle(idx)
        test.extend(idx[:max(1, int(round(test_size * len(idx))))])
    test = np.sort(np.array(test, dtype=int))
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, test


def train_predict(model: str, params: Dict, X_train, y_train, X_test, mask=None,
                  use_smote: bool = True, seed: int = 0, log: bool = True):
    """Fit preprocessing, SMOTE and the model on the training rows; return (labels, scores) on the test rows."""
    if mask is not None:
        X_train, X_test = X_train[:, mask], X_test[:, mask]
    prep = Preprocessor(log=log).fit(X_train)
    Xtr, Xte = prep.transform(X_train), prep.transform(X_test)
    ytr = y_train
    if use_smote:
        Xtr, ytr = smote_fn(Xtr, ytr, rng=seed)
    fitted = fit_model(model, params, Xtr, ytr, seed)
    return fitted.predict(Xte).astype(int), risk_scores(fitted, Xte)


def cross_validate(ds: Dataset, model: str, params: Optional[Dict] = None, mask=None, k: int = 5,
                   repeats: int = 1, seed: int = 0, use_smote: bool = True, log: bool = True) -> Dict:
    """Repeated stratified k-fold CV. Returns mean metrics, per-fold rows and out-of-fold scores (last repeat)."""
    params = params or {}
    rows, oof = [], np.zeros(ds.n)
    for r in range(repeats):
        for f, test in enumerate(stratified_folds(ds.y, k, seed + 1000 * r)):
            train = np.setdiff1d(np.arange(ds.n), test)
            pred, scores = train_predict(model, params, ds.X[train], ds.y[train], ds.X[test], mask,
                                         use_smote, seed + 1000 * r + f, log)
            rows.append(evaluate(ds.y[test], pred, scores, None if ds.loc is None else ds.loc[test]))
            if r == repeats - 1:
                oof[test] = scores
    return {"mean": mean_metrics(rows), "folds": rows, "oof_scores": oof}


def leaky_cross_validate(ds: Dataset, model: str, params: Optional[Dict] = None, mask=None, k: int = 5,
                         seed: int = 0, log: bool = False) -> Dict:
    """The protocol of the original notebooks, kept to measure its bias. DO NOT use for real estimates.

    Scaling and SMOTE are applied to the whole dataset first; cross-validation then runs on
    the resampled rows, so synthetic neighbours of each test module sit in the training folds
    (and the synthetic rows themselves are tested too).
    """
    params = params or {}
    X = ds.X if mask is None else ds.X[:, mask]
    Xs = Preprocessor(log=log).fit_transform(X)
    Xr, yr = smote_fn(Xs, ds.y, rng=seed)
    rows = []
    for f, test in enumerate(stratified_folds(yr, k, seed)):
        train = np.setdiff1d(np.arange(len(yr)), test)
        fitted = fit_model(model, params, Xr[train], yr[train], seed)
        rows.append(evaluate(yr[test], fitted.predict(Xr[test]), risk_scores(fitted, Xr[test])))
    return {"mean": mean_metrics(rows), "folds": rows}


def evaluate_transfer(train: Dataset, test: Dataset, model: str, params: Optional[Dict] = None,
                      seed: int = 0, use_smote: bool = True) -> Dict[str, float]:
    """Train on one dataset (e.g. an older release or other projects), test on another."""
    common = [f for f in test.features if f in train.features]
    Xtr = train.X[:, [train.features.index(f) for f in common]]
    Xte = test.X[:, [test.features.index(f) for f in common]]
    pred, scores = train_predict(model, params or {}, Xtr, train.y, Xte, None, use_smote, seed)
    return evaluate(test.y, pred, scores, test.loc)


def concat(datasets: Sequence[Dataset], name: str) -> Dataset:
    features = [f for f in datasets[0].features if all(f in d.features for d in datasets)]
    X = np.vstack([d.X[:, [d.features.index(f) for f in features]] for d in datasets])
    y = np.concatenate([d.y for d in datasets])
    loc = np.concatenate([d.loc for d in datasets]) if all(d.loc is not None for d in datasets) else None
    modules = [m for d in datasets for m in d.modules]
    return Dataset(name=name, X=X, y=y, features=features, modules=modules, loc=loc)
