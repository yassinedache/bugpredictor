"""SMOTE (Chawla et al., 2002): synthetic minority over-sampling, written with NumPy.

Each synthetic module is placed at a random point on the segment between a minority
module and one of its k nearest minority neighbours. SMOTE must only ever see the
training rows: applied before the train/test split, synthetic copies of test modules
end up in the training set and the scores become meaningless (see the README).
"""

from __future__ import annotations

import numpy as np


def smote(X: np.ndarray, y: np.ndarray, k: int = 5, rng=None, ratio: float = 1.0):
    """Return (X_resampled, y_resampled) with the minority class grown to ``ratio`` x majority.

    The original rows come first, unchanged; the synthetic rows are appended.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    minority = classes[np.argmin(counts)]
    idx = np.flatnonzero(y == minority)
    n_min, n_maj = counts.min(), counts.max()
    n_new = int(round(ratio * n_maj)) - n_min
    if n_new <= 0 or n_min < 2:
        return X, y
    P = X[idx]
    k = min(k, n_min - 1)
    # pairwise squared distances inside the minority class
    sq = (P * P).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * P @ P.T
    np.fill_diagonal(D, np.inf)
    neighbours = np.argsort(D, axis=1, kind="stable")[:, :k]
    base = rng.integers(0, n_min, size=n_new)
    pick = neighbours[base, rng.integers(0, k, size=n_new)]
    gap = rng.random(n_new)[:, None]
    synthetic = P[base] + gap * (P[pick] - P[base])
    return np.vstack([X, synthetic]), np.concatenate([y, np.full(n_new, minority, dtype=y.dtype)])
