import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

from bugpredictor import BENCHMARK, GeneticOptimizer, cross_validate, leaky_cross_validate, list_releases, load, load_promise
from bugpredictor.data import previous_releases
from bugpredictor.evaluation import stratified_folds, stratified_split, train_predict
from bugpredictor.metrics import auc, evaluate, manual_up, recall_at_effort
from bugpredictor.models import MODELS, Param
from bugpredictor.preprocessing import Preprocessor
from bugpredictor.smote import smote


# ----------------------------------------------------------------------------- data
def test_promise_releases():
    assert len(list_releases()) == 41
    assert all(r in list_releases() for r in BENCHMARK)
    d = load_promise("ant-1.7")
    assert d.X.shape == (745, 20) and d.y.sum() == 166
    assert d.features[0] == "wmc" and "loc" in d.features and "name" not in d.features
    assert d.modules[0].startswith("org.apache.tools.ant")
    assert np.array_equal(d.loc, d.X[:, d.features.index("loc")])
    assert previous_releases("ant-1.7") == ["ant-1.3", "ant-1.4", "ant-1.5", "ant-1.6"]


def test_csv_semicolon_and_text_target():
    d = load(b"file;loc;cbo;label\nA.java;10;1;clean\nB.java;200;7;buggy\nC.java;30;2;clean\n")
    assert d.features == ["loc", "cbo"] and d.y.tolist() == [0, 1, 0] and d.modules == ["A.java", "B.java", "C.java"]
    assert d.target == "label" and d.loc.tolist() == [10, 200, 30]


def test_arff_with_missing_values_and_nominal_target():
    text = b"""% comment
@RELATION r
@ATTRIBUTE wmc NUMERIC
@ATTRIBUTE rfc REAL
@ATTRIBUTE defects {false,true}
@DATA
1,2,false
?,5,true
3,'7',false
"""
    d = load(text)
    assert d.features == ["wmc", "rfc"] and d.y.tolist() == [0, 1, 0]
    assert np.isnan(d.X[1, 0]) and any("missing" in n for n in d.notes)


def test_bad_files_are_rejected():
    with pytest.raises(ValueError, match="single class"):
        load(b"a,bug\n1,0\n2,0\n")
    with pytest.raises(ValueError, match="numeric feature"):
        load(b"name,bug\nx,1\ny,0\n")


# ----------------------------------------------------------------------------- preprocessing and SMOTE
def test_preprocessor_uses_training_statistics_only():
    Xtr = np.array([[1.0, -1.0, 5.0], [3.0, 1.0, 5.0], [np.nan, 0.0, 5.0]])
    p = Preprocessor().fit(Xtr)
    assert p.median_[0] == 2.0
    assert p.log_cols_.tolist() == [True, False, True]       # column 2 has negative values
    Z = p.transform(Xtr)
    assert np.allclose(Z.mean(axis=0)[:2], 0) and np.allclose(Z[:, 2], 0)   # constant column: std set to 1
    Xte = np.array([[100.0, 10.0, 5.0]])
    assert np.allclose(p.transform(Xte), (p._log(Xte) - p.mean_) / p.std_)


def test_smote_balances_and_interpolates():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(40, 3)), [[0, 0, 0], [1, 2, 3]]])
    y = np.array([0] * 40 + [1, 1])
    Xr, yr = smote(X, y, rng=1)
    assert (yr == 1).sum() == (yr == 0).sum() == 40
    assert np.array_equal(Xr[:42], X) and np.array_equal(yr[:42], y)        # originals first, unchanged
    g = Xr[42:] / np.array([1, 2, 3])                                     # all on the segment [(0,0,0), (1,2,3)]
    assert np.allclose(g, g[:, :1]) and (g >= 0).all() and (g <= 1).all()
    Xb, yb = smote(X[:2], np.array([0, 1]))                               # cannot oversample one point
    assert len(yb) == 2
    assert np.array_equal(smote(X, y, rng=5)[0], smote(X, y, rng=5)[0])


# ----------------------------------------------------------------------------- metrics
def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 2)          # with ties
    p = (s > 0.4).astype(int)
    m = evaluate(y, p, s)
    assert m["precision"] == pytest.approx(precision_score(y, p))
    assert m["recall"] == pytest.approx(recall_score(y, p))
    assert m["f1"] == pytest.approx(f1_score(y, p))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, p))
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_recall_at_effort():
    y = np.array([1, 0, 1, 0])
    loc = np.array([10, 10, 60, 20])
    # density order: module 0 (0.9/10), 1 (0.5/10), 3, 2; 20 % of 100 lines = modules 0 and 1
    assert recall_at_effort(y, np.array([0.9, 0.5, 0.8, 0.1]), loc) == 0.5
    assert manual_up(y, loc) == 0.5            # smallest first: modules 0 and 1 (10 + 10 lines)


# ----------------------------------------------------------------------------- evaluation
def test_stratified_folds_partition_and_keep_the_class_ratio():
    y = np.array([1] * 30 + [0] * 70)
    folds = stratified_folds(y, 5, 0)
    assert sorted(np.concatenate(folds).tolist()) == list(range(100))
    assert all(y[f].sum() == 6 for f in folds)
    tr, te = stratified_split(y, 0.3, 0)
    assert len(np.intersect1d(tr, te)) == 0 and y[te].sum() == 9 and len(te) == 30


def test_smote_before_cv_inflates_the_scores():
    """The bug of the original notebooks: SMOTE on the whole dataset leaks test information."""
    d = load_promise("synapse-1.2")
    leaky = leaky_cross_validate(d, "knn", {"n_neighbors": 1}, seed=0)["mean"]["f1"]
    honest = cross_validate(d, "knn", {"n_neighbors": 1}, repeats=3, seed=0, log=False)["mean"]["f1"]
    assert leaky > honest + 0.15


def test_train_predict_never_sees_test_rows():
    d = load_promise("log4j-1.1")
    tr, te = stratified_split(d.y, 0.3, 0)
    X = d.X.copy()
    X[te] = 1e9                                # absurd test rows would distort the scaler if it saw them
    pred1, s1 = train_predict("lr", {}, X[tr], d.y[tr], d.X[te], seed=0)
    pred2, s2 = train_predict("lr", {}, d.X[tr], d.y[tr], d.X[te], seed=0)
    assert np.allclose(s1, s2)


# ----------------------------------------------------------------------------- GA
def test_param_encoding_roundtrip():
    for p, v in [(Param("k", "int", 1, 30), 17), (Param("C", "log", 1e-3, 1e2), 0.5), (Param("w", "choice", choices=("a", "b")), "b")]:
        assert p.decode(p.encode(v)) == pytest.approx(v)


def test_ga_is_reproducible_and_elitist():
    d = load_promise("log4j-1.1")
    a = GeneticOptimizer("knn", pop_size=8, generations=4, seed=3).run(d.X, d.y)
    b = GeneticOptimizer("knn", pop_size=8, generations=4, seed=3).run(d.X, d.y)
    assert np.array_equal(a.mask, b.mask) and a.params == b.params and a.fitness == b.fitness
    best = [h["best"] for h in a.history]
    assert all(x <= y + 1e-12 for x, y in zip(best, best[1:]))          # elitism: never worse
    assert a.history[0]["features"] <= d.X.shape[1]
    default = GeneticOptimizer("knn", generations=0, pop_size=1, seed=3).run(d.X, d.y)
    assert default.params == MODELS["knn"]["default"] and default.mask.all()
    assert a.fitness >= default.fitness - 1e-12                          # the default is in the first population


def test_ga_options():
    d = load_promise("log4j-1.1")
    r = GeneticOptimizer("lr", pop_size=6, generations=2, select_features=False, seed=0).run(d.X, d.y)
    assert r.mask.all()
    r = GeneticOptimizer("nb", pop_size=6, generations=2, tune_params=False, seed=0).run(d.X, d.y)
    assert r.params == MODELS["nb"]["default"]
