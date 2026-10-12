"""Preprocessing fitted on training data only: median imputation, log transform, z-score."""

from __future__ import annotations

import warnings

import numpy as np


class Preprocessor:
    """Fit on the training rows, then apply the same transformation to any rows.

    Code metrics are heavily right-skewed (a few classes have thousands of lines), so
    non-negative columns go through log(1 + x) before standardisation. Every statistic
    (medians, means, standard deviations) comes from the training rows: computing them
    on the whole dataset would leak information from the test rows.
    """

    def __init__(self, log: bool = True):
        self.log = log

    def fit(self, X: np.ndarray) -> "Preprocessor":
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():          # a column that is entirely missing gives NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median_ = np.nan_to_num(np.nanmedian(X, axis=0))
        Z = np.where(np.isnan(X), self.median_, X)
        self.log_cols_ = (Z.min(axis=0) >= 0) if self.log else np.zeros(X.shape[1], bool)
        Z = self._log(Z)
        self.mean_ = Z.mean(axis=0)
        std = Z.std(axis=0)
        self.std_ = np.where(std > 1e-12, std, 1.0)
        return self

    def _log(self, Z: np.ndarray) -> np.ndarray:
        Z = Z.copy()
        c = self.log_cols_
        Z[:, c] = np.log1p(np.maximum(Z[:, c], 0.0))
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Z = np.where(np.isnan(X), self.median_, X)
        return (self._log(Z) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
