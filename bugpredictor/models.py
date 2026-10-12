"""Classifiers, their default settings and the hyperparameter ranges searched by the GA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class Param:
    """One hyperparameter gene. The GA stores a number u in [0, 1] and decodes it here."""
    name: str
    kind: str                       # "int", "log" (log-uniform float) or "choice"
    low: float = 0.0
    high: float = 1.0
    choices: Sequence[Any] = ()

    def decode(self, u: float):
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "int":
            return int(round(self.low + u * (self.high - self.low)))
        if self.kind == "log":
            return float(10 ** (math.log10(self.low) + u * (math.log10(self.high) - math.log10(self.low))))
        return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]

    def encode(self, value) -> float:
        if self.kind == "int":
            return (value - self.low) / (self.high - self.low)
        if self.kind == "log":
            return (math.log10(value) - math.log10(self.low)) / (math.log10(self.high) - math.log10(self.low))
        return (list(self.choices).index(value) + 0.5) / len(self.choices)


MODELS: Dict[str, Dict[str, Any]] = {
    "knn": {"label": "K-nearest neighbours",
            "default": {"n_neighbors": 5, "weights": "uniform", "p": 2},
            "space": [Param("n_neighbors", "int", 1, 30), Param("weights", "choice", choices=("uniform", "distance")),
                      Param("p", "choice", choices=(1, 2))]},
    "svm": {"label": "Support vector machine (RBF)",
            "default": {"C": 1.0, "gamma": "scale"},
            "space": [Param("C", "log", 1e-2, 1e3), Param("gamma", "log", 1e-4, 10.0)]},
    "lr": {"label": "Logistic regression",
           "default": {"C": 1.0},
           "space": [Param("C", "log", 1e-3, 1e2)]},
    "nb": {"label": "Gaussian naive Bayes",
           "default": {"var_smoothing": 1e-9},
           "space": [Param("var_smoothing", "log", 1e-12, 1.0)]},
    "rf": {"label": "Random forest",
           "default": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
           "space": [Param("n_estimators", "int", 20, 200), Param("max_depth", "int", 2, 20),
                     Param("min_samples_leaf", "int", 1, 10)]},
}


def make_model(name: str, params: Dict[str, Any] | None = None, seed: int = 0):
    p = dict(MODELS[name]["default"])
    p.update(params or {})
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=int(p["n_neighbors"]), weights=p["weights"], p=int(p["p"]))
    if name == "svm":
        return SVC(C=p["C"], gamma=p["gamma"], kernel="rbf")
    if name == "lr":
        return LogisticRegression(C=p["C"], max_iter=2000)
    if name == "nb":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if name == "rf":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_depth=p["max_depth"],
                                      min_samples_leaf=int(p["min_samples_leaf"]), random_state=seed, n_jobs=1)
    raise ValueError(f"unknown model '{name}' (choose from {', '.join(MODELS)})")


def fit_model(name: str, params, X, y, seed: int = 0):
    model = make_model(name, params, seed)
    if name == "knn":   # k cannot exceed the number of training rows
        model.set_params(n_neighbors=min(model.n_neighbors, len(y)))
    return model.fit(X, y)


def risk_scores(model, X) -> np.ndarray:
    """Higher = more likely defective. Probabilities when available, else decision values."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]
    return model.decision_function(X)


def describe(name: str, params: Dict[str, Any]) -> List[Dict[str, str]]:
    """Human-readable parameter list for reports and the web app."""
    out = []
    for key, value in {**MODELS[name]["default"], **params}.items():
        if isinstance(value, float):
            value = f"{value:.3g}"
        out.append({"name": key, "value": "None" if value is None else str(value)})
    return out
