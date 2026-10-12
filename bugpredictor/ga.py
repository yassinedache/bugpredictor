"""Genetic algorithm that selects code metrics and tunes the classifier at the same time.

An individual = a bit mask over the features + one number in [0, 1] per hyperparameter
(decoded by ``models.Param``). Its fitness is the mean F1 of an inner cross-validation on
the training rows only, minus a small penalty per selected feature. The test rows are
never seen during the search: they are kept for the final, honest evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .metrics import evaluate
from .models import MODELS
from .evaluation import stratified_folds, train_predict


@dataclass
class GAResult:
    mask: np.ndarray
    params: Dict
    fitness: float
    cv_f1: float
    history: List[Dict] = field(default_factory=list)
    evaluations: int = 0

    def selected(self, features: List[str]) -> List[str]:
        return [f for f, m in zip(features, self.mask) if m]


class GeneticOptimizer:
    def __init__(self, model: str, pop_size: int = 20, generations: int = 15, crossover_rate: float = 0.8,
                 bit_mutation: Optional[float] = None, param_mutation: float = 0.25, tournament: int = 3,
                 elite: int = 2, patience: int = 6, feature_penalty: float = 0.005, inner_folds: int = 5,
                 select_features: bool = True, tune_params: bool = True, use_smote: bool = True,
                 log: bool = True, seed: int = 0):
        self.model = model
        self.space = MODELS[model]["space"]
        self.pop_size, self.generations = pop_size, generations
        self.crossover_rate, self.bit_mutation, self.param_mutation = crossover_rate, bit_mutation, param_mutation
        self.tournament, self.elite, self.patience = tournament, elite, patience
        self.feature_penalty, self.inner_folds = feature_penalty, inner_folds
        self.select_features, self.tune_params = select_features, tune_params
        self.use_smote, self.log = use_smote, log
        self.seed = seed

    # ---------------------------------------------------------------- genome helpers
    def decode(self, genes: Optional[np.ndarray]) -> Dict:
        if genes is None or not self.tune_params:     # None = the exact default settings
            return dict(MODELS[self.model]["default"])
        params = {p.name: p.decode(u) for p, u in zip(self.space, genes)}
        if self.model == "svm" and "gamma" not in params:
            params["gamma"] = "scale"
        return params

    def _default_genes(self) -> np.ndarray:
        """Closest genes to the defaults (gamma='scale' or max_depth=None have no exact gene)."""
        d = MODELS[self.model]["default"]
        out = []
        for p in self.space:
            v = d.get(p.name)
            known = v is not None and (p.kind == "choice" or not isinstance(v, str))
            out.append(p.encode(v) if known else 0.5)
        return np.clip(np.array(out, float), 0, 1)

    def _repair(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            mask[self.rng.integers(len(mask))] = True
        return mask

    def _key(self, mask, genes):
        return (mask.tobytes(), tuple(repr(v) for v in self.decode(genes).values()))

    # ---------------------------------------------------------------- fitness
    def _fitness(self, mask, genes, X, y, loc):
        key = self._key(mask, genes)
        if key in self.cache:
            return self.cache[key]
        params = self.decode(genes)
        f1s = []
        for i, test in enumerate(self.folds):
            train = np.setdiff1d(np.arange(len(y)), test)
            pred, _ = train_predict(self.model, params, X[train], y[train], X[test], mask,
                                    self.use_smote, self.seed + i, self.log)
            f1s.append(evaluate(y[test], pred)["f1"])
        cv = float(np.mean(f1s))
        fit = cv - self.feature_penalty * mask.sum()
        self.cache[key] = (fit, cv)
        self.evaluations += 1
        return fit, cv

    # ---------------------------------------------------------------- main loop
    def run(self, X: np.ndarray, y: np.ndarray, loc=None,
            callback: Optional[Callable[[Dict], None]] = None) -> GAResult:
        self.rng = np.random.default_rng(self.seed)
        self.cache, self.evaluations = {}, 0
        self.folds = stratified_folds(y, self.inner_folds, self.seed)
        d, g = X.shape[1], len(self.space)
        pbit = self.bit_mutation if self.bit_mutation is not None else 1.0 / d

        def random_individual():
            mask = self.rng.random(d) < 0.5 if self.select_features else np.ones(d, bool)
            genes = self.rng.random(g) if self.tune_params else None
            return self._repair(mask), genes

        # the default model on all features is part of the first population, so the search
        # starts from the baseline and can only keep it or find something better on the inner CV
        pop = [(np.ones(d, bool), None)] + [random_individual() for _ in range(self.pop_size - 1)]
        scores = [self._fitness(m, gn, X, y, loc) for m, gn in pop]
        history, best_seen, stall = [], -np.inf, 0
        for gen in range(self.generations + 1):
            fits = np.array([s[0] for s in scores])
            order = np.argsort(-fits, kind="stable")
            b = order[0]
            history.append({"generation": gen, "best": float(fits[b]), "mean": float(fits.mean()),
                            "best_f1": float(scores[b][1]), "features": int(pop[b][0].sum()),
                            "evaluations": self.evaluations})
            if callback:
                callback(history[-1])
            if fits[b] > best_seen + 1e-12:
                best_seen, stall = fits[b], 0
            else:
                stall += 1
            if gen == self.generations or stall >= self.patience:
                break
            children = [(pop[i][0].copy(), None if pop[i][1] is None else pop[i][1].copy())
                        for i in order[:self.elite]]
            while len(children) < self.pop_size:
                (m1, g1), (m2, g2) = self._tournament(pop, fits), self._tournament(pop, fits)
                m1, m2 = m1.copy(), m2.copy()
                g1 = self._default_genes() if g1 is None else g1.copy()
                g2 = self._default_genes() if g2 is None else g2.copy()
                if self.rng.random() < self.crossover_rate:
                    if self.select_features:
                        swap = self.rng.random(d) < 0.5          # uniform crossover on the bits
                        m1[swap], m2[swap] = m2[swap], m1[swap]
                    if self.tune_params:
                        swap = self.rng.random(g) < 0.5
                        g1[swap], g2[swap] = g2[swap], g1[swap]
                for m, gn in ((m1, g1), (m2, g2)):
                    if self.select_features:
                        flip = self.rng.random(d) < pbit
                        m[flip] = ~m[flip]
                    if self.tune_params:
                        mut = self.rng.random(g) < self.param_mutation
                        gn[mut] = np.clip(gn[mut] + self.rng.normal(0, 0.15, mut.sum()), 0, 1)
                    children.append((self._repair(m), gn))
            pop = children[:self.pop_size]
            scores = [self._fitness(m, gn, X, y, loc) for m, gn in pop]
        fits = np.array([s[0] for s in scores])
        b = int(np.argmax(fits))
        return GAResult(mask=pop[b][0].copy(), params=self.decode(pop[b][1]), fitness=float(fits[b]),
                        cv_f1=float(scores[b][1]), history=history, evaluations=self.evaluations)

    def _tournament(self, pop, fits):
        contenders = self.rng.integers(0, len(pop), self.tournament)
        w = contenders[np.argmax(fits[contenders])]
        return pop[w]
