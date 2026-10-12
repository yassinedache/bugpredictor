"""Classification and effort-aware metrics for defect prediction (class 1 = defective)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def confusion(y_true, y_pred) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {"tp": int(((y_true == 1) & (y_pred == 1)).sum()), "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()), "tn": int(((y_true == 0) & (y_pred == 0)).sum())}


def auc(y_true, scores) -> float:
    """Area under the ROC curve = P(score of a defective module > score of a clean one), ties count 1/2."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n1, n0 = int(y_true.sum()), int(len(y_true) - y_true.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):                      # average ranks for tied scores
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[y_true == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def recall_at_effort(y_true, scores, loc, effort: float = 0.2) -> float:
    """Share of defective modules found when inspecting modules by decreasing risk density.

    Modules are sorted by score / LOC (risk per line of code, so that small risky classes
    come first) and inspected until ``effort`` (20 % by default) of all lines are read.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0 or loc is None:
        return float("nan")
    loc = np.maximum(np.asarray(loc, dtype=float), 1.0)
    density = np.asarray(scores, dtype=float) / loc
    order = np.lexsort((loc, -density))     # highest density first, smaller module on ties
    inspected = np.cumsum(loc[order]) <= effort * loc.sum()
    return float(y_true[order][inspected].sum() / y_true.sum())


def scores_to_unit(scores) -> np.ndarray:
    """Map arbitrary decision scores to (0, 1) monotonically (for risk densities)."""
    s = np.asarray(scores, dtype=float)
    if s.min() >= 0 and s.max() <= 1:
        return s
    return 1.0 / (1.0 + np.exp(-s))


def evaluate(y_true, y_pred, scores=None, loc=None) -> Dict[str, float]:
    c = confusion(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    out = {"precision": precision, "recall": recall, "f1": f1, "mcc": float(mcc),
           "accuracy": (tp + tn) / max(1, tp + tn + fp + fn), "balanced_accuracy": (recall + spec) / 2,
           "pf": fp / (fp + tn) if fp + tn else 0.0}
    if scores is not None:
        out["auc"] = auc(y_true, scores)
        out["recall_20"] = recall_at_effort(y_true, scores_to_unit(scores), loc) if loc is not None else float("nan")
    out.update(c)
    return out


METRIC_NAMES = ["f1", "mcc", "auc", "recall_20", "precision", "recall", "pf", "balanced_accuracy", "accuracy"]


def mean_metrics(rows) -> Dict[str, float]:
    keys = [k for k in rows[0] if k not in ("tp", "fp", "fn", "tn")]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


def manual_up(y_true, loc) -> Optional[float]:
    """Effort-aware baseline from the literature: inspect the smallest classes first."""
    if loc is None:
        return None
    # with equal scores the risk density is 1 / LOC, so the smallest classes come first
    return recall_at_effort(y_true, np.ones(len(loc)), loc)
