"""The browser engine (web/js/ml.js) must give the same numbers as the Python package.

Runs the JavaScript with Node.js on inputs prepared here; skipped when Node is not installed.
Random steps (SMOTE, the GA) use different generators in the two languages, so they are
checked through their deterministic parts: preprocessing, models, metrics, parsing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bugpredictor.data import load, load_promise
from bugpredictor.metrics import auc, evaluate, recall_at_effort
from bugpredictor.models import fit_model, risk_scores
from bugpredictor.preprocessing import Preprocessor

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

JS = r"""
const ML = require(process.argv[1]);
const inp = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = {};
const prep = new ML.Preprocessor(true).fit(inp.Xraw);
out.prep = prep.transform(inp.Xraw_test);
out.models = inp.models.map(m => {
  const model = ML.MODELS[m.name].make(Object.assign({}, ML.MODELS[m.name].default, m.params)).fit(inp.Xtr, inp.ytr);
  return model.proba(inp.Xte);
});
out.auc = ML.auc(inp.y_auc, inp.s_auc);
out.effort = ML.recallAtEffort(inp.y_auc, inp.s_auc, inp.loc);
out.eval = ML.evaluate(inp.y_auc, inp.pred, inp.s_auc);
const ds = ML.parse(inp.csv, 'x');
out.parsed = {features: ds.features, y: ds.y, modules: ds.modules.slice(0, 3), X0: ds.X[0]};
const ds2 = ML.parse(inp.arff, 'x');
out.arff = {features: ds2.features, y: ds2.y, X: ds2.X};
process.stdout.write(JSON.stringify(out));
"""

MODELS = [("knn", {"n_neighbors": 1}), ("knn", {"n_neighbors": 5}), ("knn", {"n_neighbors": 14, "weights": "distance"}),
          ("knn", {"n_neighbors": 7, "p": 1}), ("knn", {"n_neighbors": 9, "p": 1, "weights": "distance"}),
          ("lr", {"C": 0.01}), ("lr", {"C": 1.0}), ("lr", {"C": 50.0}), ("nb", {}), ("nb", {"var_smoothing": 0.05})]

ARFF = """% test
@RELATION t
@ATTRIBUTE wmc NUMERIC
@ATTRIBUTE 'loc' REAL
@ATTRIBUTE bug {false,true}
@DATA
1,10,false
2,?,true
3,30,'true'
"""


def run_js(payload):
    proc = subprocess.run([NODE, "-e", JS, str(ROOT / "web" / "js" / "ml.js")], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_js_matches_python(tmp_path):
    rng = np.random.default_rng(3)
    d = load_promise("ant-1.7")
    Xraw = d.X[:300].copy()
    Xraw[rng.random(Xraw.shape) < 0.02] = np.nan                # some missing values
    Xraw_test = d.X[300:340]
    prep = Preprocessor().fit(Xraw)
    Z = prep.transform(d.X[:400])
    Xtr, ytr, Xte = Z[:300] + 1e-9 * rng.random((300, Z.shape[1])), d.y[:300], Z[300:400]   # break exact ties
    y_auc = d.y[:200]
    s_auc = np.round(rng.random(200), 1)                        # many ties
    pred = (s_auc > 0.5).astype(int)
    csv_text = (ROOT / "data" / "promise" / "ivy" / "ivy-2.0.csv").read_text()
    payload = {"Xraw": np.where(np.isnan(Xraw), None, Xraw).tolist(), "Xraw_test": Xraw_test.tolist(),
               "Xtr": Xtr.tolist(), "ytr": ytr.tolist(), "Xte": Xte.tolist(),
               "models": [{"name": n, "params": p} for n, p in MODELS],
               "y_auc": y_auc.tolist(), "s_auc": s_auc.tolist(), "loc": d.loc[:200].tolist(), "pred": pred.tolist(),
               "csv": csv_text, "arff": ARFF}
    out = run_js(payload)
    # missing values travel as JSON null; the JS preprocessor treats null like NaN
    np.testing.assert_allclose(out["prep"], prep.transform(Xraw_test), atol=1e-9)
    for (name, params), js in zip(MODELS, out["models"]):
        m = fit_model(name, params, Xtr, ytr)
        if name == "lr":   # scikit-learn stops at tol=1e-4 by default; compare with the exact optimum
            m.set_params(tol=1e-12, max_iter=100000).fit(Xtr, ytr)
        py = risk_scores(m, Xte)
        np.testing.assert_allclose(js, py, atol=1e-6 if name == "lr" else 1e-9, err_msg=f"{name} {params}")
    assert out["auc"] == pytest.approx(auc(y_auc, s_auc), abs=1e-12)
    assert out["effort"] == pytest.approx(recall_at_effort(y_auc, s_auc, d.loc[:200]), abs=1e-12)
    py = evaluate(y_auc, pred, s_auc)
    for k in ("f1", "mcc", "precision", "recall", "pf", "accuracy", "auc"):
        assert out["eval"][k] == pytest.approx(py[k], abs=1e-12), k
    ivy = load_promise("ivy-2.0")
    assert out["parsed"]["features"] == ivy.features
    assert out["parsed"]["y"] == ivy.y.tolist()
    assert out["parsed"]["modules"] == ivy.modules[:3]
    np.testing.assert_allclose(out["parsed"]["X0"], ivy.X[0])
    a = load(ARFF.encode())
    assert out["arff"]["features"] == a.features and out["arff"]["y"] == a.y.tolist()
