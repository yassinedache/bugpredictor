/* BugPredictor browser engine: a JavaScript port of the Python package (bugpredictor/).
   Same steps, same JSON result; used when the app is served as static files (no Python server).
   Models available here: k-nearest neighbours, logistic regression, Gaussian naive Bayes. */
(function (root) {
  "use strict";

  /* ------------------------------------------------------------------ random numbers */
  function rng(seed) {                       // mulberry32: small, fast, reproducible
    var a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randint(r, n) { return Math.floor(r() * n); }
  function shuffle(a, r) { for (var i = a.length - 1; i > 0; i--) { var j = randint(r, i + 1), t = a[i]; a[i] = a[j]; a[j] = t; } return a; }
  function gauss(r) { var u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  /* ------------------------------------------------------------------ parsing */
  var TARGETS = ["bug", "bugs", "defects", "defect", "defective", "buggy", "faulty", "fault", "faults", "class", "label", "target", "problems", "error", "errors"];
  var IDS = ["name", "module", "file", "filename", "class_name", "classname", "path", "id"];
  var TRUE_W = ["true", "yes", "y", "t", "1", "buggy", "defective", "bug", "faulty", "fault"];
  var FALSE_W = ["false", "no", "n", "f", "0", "clean", "non-defective", "nondefective", "ok"];

  function splitLine(line, delim, quote) {
    var out = [], cur = "", q = false;
    for (var i = 0; i < line.length; i++) {
      var c = line[i];
      if (q) {
        if (c === quote) { if (line[i + 1] === quote) { cur += c; i++; } else q = false; }
        else cur += c;
      } else if (c === quote) q = true;
      else if (c === delim) { out.push(cur); cur = ""; }
      else cur += c;
    }
    out.push(cur);
    return out;
  }
  function parseCSV(text) {
    var lines = text.split(/\r?\n/).filter(function (l) { return l.trim() !== ""; });
    if (lines.length < 2) throw new Error("the file needs a header line and at least one data line");
    var best = ",", bestN = 0;
    [",", ";", "\t", "|"].forEach(function (d) { var n = lines[0].split(d).length; if (n > bestN) { bestN = n; best = d; } });
    return { columns: splitLine(lines[0], best, '"').map(function (s) { return s.trim(); }),
             rows: lines.slice(1).map(function (l) { return splitLine(l, best, '"'); }) };
  }
  function parseARFF(text) {
    var names = [], rows = [], data = false;
    text.split(/\r?\n/).forEach(function (raw) {
      var line = raw.trim();
      if (!line || line[0] === "%") return;
      var low = line.toLowerCase();
      if (!data) {
        if (low.indexOf("@attribute") === 0) {
          var m = /^@attribute\s+('([^']*)'|"([^"]*)"|(\S+))\s+/i.exec(line);
          if (!m) throw new Error("cannot read attribute line: " + line);
          names.push(m[2] || m[3] || m[4]);
        } else if (low.indexOf("@data") === 0) data = true;
        return;
      }
      if (line[0] === "{") throw new Error("sparse ARFF data is not supported");
      rows.push(splitLine(line, ",", "'").map(function (s) { return s.trim(); }));
    });
    if (!names.length) throw new Error("no @attribute found: is this really an ARFF file?");
    return { columns: names, rows: rows };
  }
  function num(v) {
    v = String(v).trim().replace(/^['"]|['"]$/g, "");
    if (v === "" || v === "?" || /^(na|nan|null|none)$/i.test(v)) return NaN;
    var x = Number(v);
    if (isNaN(x)) throw new Error("nan");
    return x;
  }
  function detectTarget(cols) {
    var low = cols.map(function (c) { return c.toLowerCase(); });
    for (var i = 0; i < TARGETS.length; i++) { var j = low.indexOf(TARGETS[i]); if (j >= 0) return cols[j]; }
    return cols[cols.length - 1];
  }
  function parse(text, name, target) {
    var head = text.replace(/^\s+/, "").split(/\r?\n/).slice(0, 50).filter(function (l) { return l[0] !== "%"; }).join("\n").toLowerCase();
    var tab = (head.indexOf("@relation") >= 0 || head.indexOf("@attribute") >= 0) ? parseARFF(text) : parseCSV(text);
    var cols = tab.columns, width = cols.length;
    var rows = tab.rows.filter(function (r) { return r.length === width; });
    if (!rows.length) throw new Error("no complete data row (check the separator)");
    target = target || detectTarget(cols);
    var t = cols.indexOf(target);
    if (t < 0) throw new Error("target column '" + target + "' not found");
    var tv = rows.map(function (r) { return String(r[t]).trim().replace(/^['"]|['"]$/g, "").toLowerCase(); });
    var y;
    if (tv.every(function (v) { return v !== "" && !isNaN(Number(v)); })) y = tv.map(function (v) { return Number(v) > 0 ? 1 : 0; });
    else y = tv.map(function (v) {
      if (TRUE_W.indexOf(v) >= 0) return 1;
      if (FALSE_W.indexOf(v) >= 0) return 0;
      throw new Error("cannot read target value '" + v + "' in column '" + target + "'");
    });
    var pos = y.reduce(function (a, b) { return a + b; }, 0);
    if (pos === 0 || pos === y.length) throw new Error("the target has a single class: there is nothing to predict");
    var features = [], colsIdx = [], notes = [], moduleCol = -1;
    for (var j = 0; j < width; j++) {
      if (j === t) continue;
      var vals, ok = true;
      try { vals = rows.map(function (r) { return num(r[j]); }); } catch (e) { ok = false; }
      var lc = cols[j].toLowerCase();
      if (!ok) {
        if (moduleCol < 0 && (IDS.indexOf(lc) >= 0 || j === 0)) moduleCol = j;
        else notes.push("non-numeric column '" + cols[j] + "' ignored");
        continue;
      }
      if (IDS.indexOf(lc) >= 0 && moduleCol < 0) { moduleCol = j; continue; }
      if (vals.every(function (v) { return isNaN(v); })) { notes.push("empty column '" + cols[j] + "' ignored"); continue; }
      features.push(cols[j]); colsIdx.push(vals);
    }
    if (!features.length) throw new Error("no numeric feature column found");
    var X = rows.map(function (_, i) { return colsIdx.map(function (c) { return c[i]; }); });
    var li = features.map(function (f) { return f.toLowerCase(); }).indexOf("loc");
    var missing = 0;
    X.forEach(function (r) { r.forEach(function (v) { if (isNaN(v)) missing++; }); });
    if (missing) notes.push(missing + " missing values (filled with the training median)");
    return {
      name: name || "uploaded", X: X, y: y, features: features, target: target, notes: notes,
      modules: rows.map(function (r, i) { return moduleCol >= 0 ? String(r[moduleCol]).trim() : "row " + (i + 1); }),
      loc: li >= 0 ? X.map(function (r) { return r[li]; }) : null
    };
  }
  function summary(ds) {
    var d = ds.y.reduce(function (a, b) { return a + b; }, 0);
    return { name: ds.name, modules: ds.y.length, features: ds.features.length, defective: d,
             defect_rate: Math.round(d / ds.y.length * 1e4) / 1e4, target: ds.target, has_loc: !!ds.loc, notes: ds.notes };
  }

  /* ------------------------------------------------------------------ preprocessing */
  function missing(v) { return v == null || isNaN(v); }   // NaN, or null coming from JSON
  function median(a) {
    var b = a.filter(function (v) { return !missing(v); }).sort(function (p, q) { return p - q; });
    if (!b.length) return 0;
    var m = b.length >> 1;
    return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
  }
  function Preprocessor(log) { this.log = log !== false; }
  Preprocessor.prototype.fit = function (X) {
    var d = X[0].length, n = X.length, self = this;
    this.median = []; this.logCol = []; this.mean = []; this.std = [];
    for (var j = 0; j < d; j++) {
      var col = X.map(function (r) { return r[j]; });
      var med = median(col);
      var filled = col.map(function (v) { return missing(v) ? med : v; });
      var mn = Math.min.apply(null, filled);
      var lg = this.log && mn >= 0;
      var z = filled.map(function (v) { return lg ? Math.log1p(Math.max(v, 0)) : v; });
      var mu = 0; z.forEach(function (v) { mu += v; }); mu /= n;
      var s2 = 0; z.forEach(function (v) { s2 += (v - mu) * (v - mu); });
      var sd = Math.sqrt(s2 / n);
      self.median.push(med); self.logCol.push(lg); self.mean.push(mu); self.std.push(sd > 1e-12 ? sd : 1);
    }
    return this;
  };
  Preprocessor.prototype.transform = function (X) {
    var self = this;
    return X.map(function (r) {
      return r.map(function (v, j) {
        if (missing(v)) v = self.median[j];
        if (self.logCol[j]) v = Math.log1p(Math.max(v, 0));
        return (v - self.mean[j]) / self.std[j];
      });
    });
  };

  /* ------------------------------------------------------------------ SMOTE */
  function smote(X, y, r, k) {
    k = k || 5;
    var n1 = 0; y.forEach(function (v) { n1 += v; });
    var n0 = y.length - n1, minority = n1 <= n0 ? 1 : 0;
    var idx = []; y.forEach(function (v, i) { if (v === minority) idx.push(i); });
    var nMin = idx.length, nNew = Math.max(n0, n1) - nMin;
    if (nNew <= 0 || nMin < 2) return { X: X, y: y };
    k = Math.min(k, nMin - 1);
    var P = idx.map(function (i) { return X[i]; }), d = P[0].length;
    var nb = P.map(function (p, a) {
      var ds = [];
      for (var b = 0; b < nMin; b++) {
        if (b === a) continue;
        var s = 0;
        for (var j = 0; j < d; j++) { var t = p[j] - P[b][j]; s += t * t; }
        ds.push([s, b]);
      }
      ds.sort(function (u, v) { return u[0] - v[0] || u[1] - v[1]; });
      return ds.slice(0, k).map(function (e) { return e[1]; });
    });
    var Xo = X.slice(), yo = y.slice();
    for (var s = 0; s < nNew; s++) {
      var a = randint(r, nMin), b = nb[a][randint(r, k)], g = r(), row = new Array(d);
      for (var j = 0; j < d; j++) row[j] = P[a][j] + g * (P[b][j] - P[a][j]);
      Xo.push(row); yo.push(minority);
    }
    return { X: Xo, y: yo };
  }

  /* ------------------------------------------------------------------ models */
  function KNN(p) { this.k = p.n_neighbors; this.weights = p.weights; this.p = p.p; }
  KNN.prototype.fit = function (X, y) { this.X = X; this.y = y; this.k = Math.min(this.k, X.length); return this; };
  KNN.prototype.proba = function (Q) {
    var X = this.X, y = this.y, k = this.k, man = this.p === 1, n = X.length, d = X[0].length, dist = this.weights === "distance";
    return Q.map(function (q) {
      var bd = [], bi = [];                       // the k best, kept sorted
      for (var i = 0; i < n; i++) {
        var s = 0, xi = X[i];
        if (man) for (var j = 0; j < d; j++) s += Math.abs(q[j] - xi[j]);
        else for (j = 0; j < d; j++) { var t = q[j] - xi[j]; s += t * t; }
        if (bd.length < k || s < bd[bd.length - 1]) {
          var pos = bd.length;
          while (pos > 0 && bd[pos - 1] > s) pos--;
          bd.splice(pos, 0, s); bi.splice(pos, 0, i);
          if (bd.length > k) { bd.pop(); bi.pop(); }
        }
      }
      var w1 = 0, w = 0;
      if (dist) {
        var zero = bd.filter(function (v) { return v === 0; }).length;
        for (var m = 0; m < bd.length; m++) {
          var dm = man ? bd[m] : Math.sqrt(bd[m]);
          var wt = zero ? (bd[m] === 0 ? 1 : 0) : 1 / dm;
          w += wt; w1 += wt * y[bi[m]];
        }
      } else for (m = 0; m < bd.length; m++) { w += 1; w1 += y[bi[m]]; }
      return w1 / w;
    });
  };
  KNN.prototype.predict = function (Q) { return this.proba(Q).map(function (p) { return p > 0.5 ? 1 : 0; }); };

  function solve(A, b) {                        // Gaussian elimination with partial pivoting
    var n = b.length, M = A.map(function (r, i) { return r.concat([b[i]]); });
    for (var c = 0; c < n; c++) {
      var p = c;
      for (var r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      var t = M[c]; M[c] = M[p]; M[p] = t;
      for (r = c + 1; r < n; r++) {
        var f = M[r][c] / M[c][c];
        for (var j = c; j <= n; j++) M[r][j] -= f * M[c][j];
      }
    }
    var x = new Array(n);
    for (r = n - 1; r >= 0; r--) { var s = M[r][n]; for (j = r + 1; j < n; j++) s -= M[r][j] * x[j]; x[r] = s / M[r][r]; }
    return x;
  }
  function sigmoid(z) { return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)); }
  /* L2-regularised logistic regression, same objective as scikit-learn:
     C * sum(log-loss) + ||w||^2 / 2 (intercept not penalised), minimised by Newton's method. */
  function LogReg(p) { this.C = p.C; }
  LogReg.prototype.objective = function (X, y, w) {
    var C = this.C, d = X[0].length, s = 0;
    for (var i = 0; i < X.length; i++) {
      var z = w[d]; for (var j = 0; j < d; j++) z += w[j] * X[i][j];
      s += (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))) - y[i] * z;
    }
    var r = 0; for (j = 0; j < d; j++) r += w[j] * w[j];
    return C * s + r / 2;
  };
  LogReg.prototype.fit = function (X, y) {
    var d = X[0].length, n = X.length, C = this.C, w = new Array(d + 1).fill(0);
    var f = this.objective(X, y, w);
    for (var it = 0; it < 100; it++) {
      var g = new Array(d + 1).fill(0), H = [];
      for (var a = 0; a <= d; a++) H.push(new Array(d + 1).fill(0));
      for (var i = 0; i < n; i++) {
        var xi = X[i], z = w[d];
        for (var j = 0; j < d; j++) z += w[j] * xi[j];
        var p = sigmoid(z), e = C * (p - y[i]), v = C * p * (1 - p);
        for (j = 0; j < d; j++) { g[j] += e * xi[j]; for (var k = 0; k <= j; k++) H[j][k] += v * xi[j] * xi[k]; H[d][j] += v * xi[j]; }
        g[d] += e; H[d][d] += v;
      }
      for (j = 0; j < d; j++) { g[j] += w[j]; H[j][j] += 1; }
      for (j = 0; j <= d; j++) for (k = j + 1; k <= d; k++) H[j][k] = H[k][j];
      H[d][d] += 1e-10;
      var step = solve(H, g), t = 1, wn, fn;
      for (var ls = 0; ls < 30; ls++) {             // backtracking line search
        wn = w.map(function (v, q) { return v - t * step[q]; });
        fn = this.objective(X, y, wn);
        if (fn <= f) break;
        t /= 2;
      }
      var moved = 0; for (j = 0; j <= d; j++) moved = Math.max(moved, Math.abs(wn[j] - w[j]));
      w = wn; f = fn;
      if (moved < 1e-9) break;
    }
    this.w = w;
    return this;
  };
  LogReg.prototype.proba = function (Q) {
    var w = this.w, d = w.length - 1;
    return Q.map(function (q) { var z = w[d]; for (var j = 0; j < d; j++) z += w[j] * q[j]; return sigmoid(z); });
  };
  LogReg.prototype.predict = function (Q) { return this.proba(Q).map(function (p) { return p > 0.5 ? 1 : 0; }); };

  /* Gaussian naive Bayes, as scikit-learn: variances + var_smoothing * (largest feature variance). */
  function GNB(p) { this.vs = p.var_smoothing; }
  GNB.prototype.fit = function (X, y) {
    var d = X[0].length, n = X.length, all = [];
    for (var j = 0; j < d; j++) {
      var m = 0; for (var i = 0; i < n; i++) m += X[i][j]; m /= n;
      var v = 0; for (i = 0; i < n; i++) v += (X[i][j] - m) * (X[i][j] - m); all.push(v / n);
    }
    var eps = this.vs * Math.max.apply(null, all);
    this.stats = [0, 1].map(function (c) {
      var rows = X.filter(function (_, i) { return y[i] === c; }), nc = rows.length, mu = [], va = [];
      for (var j = 0; j < d; j++) {
        var m = 0; rows.forEach(function (r) { m += r[j]; }); m /= nc;
        var v = 0; rows.forEach(function (r) { v += (r[j] - m) * (r[j] - m); });
        mu.push(m); va.push(v / nc + eps);
      }
      return { prior: nc / n, mu: mu, va: va };
    });
    return this;
  };
  GNB.prototype.proba = function (Q) {
    var S = this.stats;
    return Q.map(function (q) {
      var ll = S.map(function (s) {
        var l = Math.log(s.prior);
        for (var j = 0; j < q.length; j++) l -= 0.5 * Math.log(2 * Math.PI * s.va[j]) + 0.5 * (q[j] - s.mu[j]) * (q[j] - s.mu[j]) / s.va[j];
        return l;
      });
      var mx = Math.max(ll[0], ll[1]);
      var e0 = Math.exp(ll[0] - mx), e1 = Math.exp(ll[1] - mx);
      return e1 / (e0 + e1);
    });
  };
  GNB.prototype.predict = function (Q) { return this.proba(Q).map(function (p) { return p > 0.5 ? 1 : 0; }); };

  function P(name, kind, low, high, choices) { return { name: name, kind: kind, low: low, high: high, choices: choices }; }
  var MODELS = {
    knn: { label: "K-nearest neighbours", make: function (p) { return new KNN(p); },
           default: { n_neighbors: 5, weights: "uniform", p: 2 },
           space: [P("n_neighbors", "int", 1, 30), P("weights", "choice", 0, 1, ["uniform", "distance"]), P("p", "choice", 0, 1, [1, 2])] },
    lr: { label: "Logistic regression", make: function (p) { return new LogReg(p); },
          default: { C: 1.0 }, space: [P("C", "log", 1e-3, 1e2)] },
    nb: { label: "Gaussian naive Bayes", make: function (p) { return new GNB(p); },
          default: { var_smoothing: 1e-9 }, space: [P("var_smoothing", "log", 1e-12, 1.0)] }
  };
  function decodeParam(p, u) {
    u = Math.min(Math.max(u, 0), 1);
    if (p.kind === "int") return Math.round(p.low + u * (p.high - p.low));
    if (p.kind === "log") return Math.pow(10, Math.log10(p.low) + u * (Math.log10(p.high) - Math.log10(p.low)));
    return p.choices[Math.min(Math.floor(u * p.choices.length), p.choices.length - 1)];
  }
  function encodeParam(p, v) {
    if (p.kind === "int") return (v - p.low) / (p.high - p.low);
    if (p.kind === "log") return (Math.log10(v) - Math.log10(p.low)) / (Math.log10(p.high) - Math.log10(p.low));
    return (p.choices.indexOf(v) + 0.5) / p.choices.length;
  }
  function withDefaults(model, params) { var o = {}, d = MODELS[model].default, k; for (k in d) o[k] = d[k]; for (k in params || {}) o[k] = params[k]; return o; }
  function describe(model, params) {
    var o = withDefaults(model, params);
    return Object.keys(o).map(function (k) {
      var v = o[k];
      return { name: k, value: typeof v === "number" && !Number.isInteger(v) ? fmt3(v) : String(v) };
    });
  }
  function fmt3(v) { var s = Number(v).toPrecision(3); return String(Number(s)).length < s.length ? String(Number(s)) : s; }

  /* ------------------------------------------------------------------ metrics */
  function confusion(y, p) {
    var c = { tp: 0, fp: 0, fn: 0, tn: 0 };
    for (var i = 0; i < y.length; i++) { if (y[i]) { if (p[i]) c.tp++; else c.fn++; } else { if (p[i]) c.fp++; else c.tn++; } }
    return c;
  }
  function auc(y, s) {
    var n1 = 0; y.forEach(function (v) { n1 += v; });
    var n0 = y.length - n1;
    if (!n1 || !n0) return NaN;
    var idx = s.map(function (_, i) { return i; }).sort(function (a, b) { return s[a] - s[b] || a - b; });
    var ranks = new Array(s.length), i = 0;
    while (i < idx.length) {
      var j = i;
      while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++;
      for (var q = i; q <= j; q++) ranks[idx[q]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    var r1 = 0; y.forEach(function (v, k) { if (v) r1 += ranks[k]; });
    return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0);
  }
  function recallAtEffort(y, s, loc, effort) {
    effort = effort || 0.2;
    var pos = 0; y.forEach(function (v) { pos += v; });
    if (!pos || !loc) return NaN;
    var L = loc.map(function (v) { return Math.max(v, 1); }), total = 0;
    L.forEach(function (v) { total += v; });
    var idx = y.map(function (_, i) { return i; }).sort(function (a, b) {
      var da = s[a] / L[a], db = s[b] / L[b];
      return db - da || L[a] - L[b] || a - b;
    });
    var cum = 0, found = 0;
    for (var k = 0; k < idx.length; k++) { cum += L[idx[k]]; if (cum > effort * total) break; found += y[idx[k]]; }
    return found / pos;
  }
  function evaluate(y, pred, scores) {
    var c = confusion(y, pred), tp = c.tp, fp = c.fp, fn = c.fn, tn = c.tn;
    var precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
    var f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
    var den = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    var out = { precision: precision, recall: recall, f1: f1, mcc: den ? (tp * tn - fp * fn) / den : 0,
                accuracy: (tp + tn) / Math.max(1, tp + tn + fp + fn), pf: fp + tn ? fp / (fp + tn) : 0,
                tp: tp, fp: fp, fn: fn, tn: tn };
    if (scores) out.auc = auc(y, scores);
    return out;
  }

  /* ------------------------------------------------------------------ splits and training */
  function stratifiedFolds(y, k, r) {
    var folds = []; for (var f = 0; f < k; f++) folds.push([]);
    var offset = 0;
    [0, 1].forEach(function (c) {
      var idx = []; y.forEach(function (v, i) { if (v === c) idx.push(i); });
      shuffle(idx, r);
      idx.forEach(function (j, i) { folds[(i + offset) % k].push(j); });
      offset += idx.length;
    });
    return folds.map(function (f) { return f.sort(function (a, b) { return a - b; }); });
  }
  function stratifiedSplit(y, testSize, r) {
    var test = [];
    [0, 1].forEach(function (c) {
      var idx = []; y.forEach(function (v, i) { if (v === c) idx.push(i); });
      shuffle(idx, r);
      test = test.concat(idx.slice(0, Math.max(1, Math.round(testSize * idx.length))));
    });
    var inTest = {}; test.forEach(function (i) { inTest[i] = true; });
    var train = []; y.forEach(function (_, i) { if (!inTest[i]) train.push(i); });
    return { train: train, test: test.sort(function (a, b) { return a - b; }) };
  }
  function take(A, idx) { return idx.map(function (i) { return A[i]; }); }
  function cols(X, mask) {
    if (!mask) return X;
    var keep = []; mask.forEach(function (m, j) { if (m) keep.push(j); });
    return X.map(function (r) { return keep.map(function (j) { return r[j]; }); });
  }
  function trainPredict(model, params, Xtr, ytr, Xte, mask, seed) {
    Xtr = cols(Xtr, mask); Xte = cols(Xte, mask);
    var prep = new Preprocessor(true).fit(Xtr);
    var A = prep.transform(Xtr), B = prep.transform(Xte);
    var res = smote(A, ytr, rng(seed + 7919));
    var m = MODELS[model].make(withDefaults(model, params)).fit(res.X, res.y);
    var s = m.proba(B);
    return { pred: s.map(function (p) { return p > 0.5 ? 1 : 0; }), scores: s };
  }

  /* ------------------------------------------------------------------ genetic algorithm */
  function GA(model, o) {
    o = o || {};
    this.model = model; this.space = MODELS[model].space;
    this.pop = o.population || 20; this.gens = o.generations == null ? 15 : o.generations;
    this.cx = 0.8; this.pm = 0.25; this.tour = 3; this.elite = 2; this.patience = 6;
    this.penalty = 0.005; this.inner = 5; this.seed = o.seed || 0;
  }
  GA.prototype.decode = function (genes) {
    var self = this, out = {};
    if (!genes) return withDefaults(this.model, {});
    this.space.forEach(function (p, i) { out[p.name] = decodeParam(p, genes[i]); });
    return out;
  };
  GA.prototype.defaultGenes = function () {
    var d = MODELS[this.model].default;
    return this.space.map(function (p) { return Math.min(1, Math.max(0, encodeParam(p, d[p.name]))); });
  };
  GA.prototype.run = function (X, y, onGen) {
    var self = this, r = rng(this.seed * 2654435761 + 1), d = X[0].length, g = this.space.length, pbit = 1 / d;
    var folds = stratifiedFolds(y, this.inner, rng(this.seed + 17)), cache = {}, evals = 0;
    function repair(m) { if (!m.some(Boolean)) m[randint(r, d)] = true; return m; }
    function fitness(m, genes) {
      var params = self.decode(genes), key = m.map(Number).join("") + JSON.stringify(params);
      if (cache[key]) return cache[key];
      var f1 = 0;
      folds.forEach(function (test, i) {
        var inTest = {}; test.forEach(function (q) { inTest[q] = true; });
        var tr = []; y.forEach(function (_, q) { if (!inTest[q]) tr.push(q); });
        var o = trainPredict(self.model, params, take(X, tr), take(y, tr), take(X, test), m, self.seed + i);
        f1 += evaluate(take(y, test), o.pred).f1;
      });
      f1 /= folds.length;
      var nsel = m.filter(Boolean).length;
      evals++;
      return (cache[key] = { fit: f1 - self.penalty * nsel, f1: f1 });
    }
    var pop = [{ m: new Array(d).fill(true), g: null }];
    while (pop.length < this.pop) {
      var m = []; for (var j = 0; j < d; j++) m.push(r() < 0.5);
      var gn = []; for (j = 0; j < g; j++) gn.push(r());
      pop.push({ m: repair(m), g: gn });
    }
    var sc = pop.map(function (ind) { return fitness(ind.m, ind.g); });
    var history = [], bestSeen = -Infinity, stall = 0;
    for (var gen = 0; gen <= this.gens; gen++) {
      var order = sc.map(function (_, i) { return i; }).sort(function (a, b) { return sc[b].fit - sc[a].fit || a - b; });
      var b = order[0], mean = 0;
      sc.forEach(function (s) { mean += s.fit; });
      var h = { generation: gen, best: sc[b].fit, mean: mean / sc.length, best_f1: sc[b].f1, features: pop[b].m.filter(Boolean).length, evaluations: evals };
      history.push(h);
      if (onGen) onGen(h);
      if (sc[b].fit > bestSeen + 1e-12) { bestSeen = sc[b].fit; stall = 0; } else stall++;
      if (gen === this.gens || stall >= this.patience) break;
      var tour = function () {
        var w = -1;
        for (var t = 0; t < self.tour; t++) { var c = randint(r, pop.length); if (w < 0 || sc[c].fit > sc[w].fit) w = c; }
        return pop[w];
      };
      var kids = order.slice(0, this.elite).map(function (i) { return { m: pop[i].m.slice(), g: pop[i].g ? pop[i].g.slice() : null }; });
      while (kids.length < this.pop) {
        var p1 = tour(), p2 = tour();
        var a = { m: p1.m.slice(), g: (p1.g || self.defaultGenes()).slice() }, c2 = { m: p2.m.slice(), g: (p2.g || self.defaultGenes()).slice() };
        if (r() < this.cx) {
          for (j = 0; j < d; j++) if (r() < 0.5) { var t1 = a.m[j]; a.m[j] = c2.m[j]; c2.m[j] = t1; }
          for (j = 0; j < g; j++) if (r() < 0.5) { var t2 = a.g[j]; a.g[j] = c2.g[j]; c2.g[j] = t2; }
        }
        [a, c2].forEach(function (k) {
          for (var q = 0; q < d; q++) if (r() < pbit) k.m[q] = !k.m[q];
          for (q = 0; q < g; q++) if (r() < self.pm) k.g[q] = Math.min(1, Math.max(0, k.g[q] + 0.15 * gauss(r)));
          kids.push({ m: repair(k.m), g: k.g });
        });
      }
      pop = kids.slice(0, this.pop);
      sc = pop.map(function (ind) { return fitness(ind.m, ind.g); });
    }
    var bi = 0; sc.forEach(function (s, i) { if (s.fit > sc[bi].fit) bi = i; });
    return { mask: pop[bi].m, params: this.decode(pop[bi].g), fitness: sc[bi].fit, cv_f1: sc[bi].f1, history: history, evaluations: evals };
  };

  /* ------------------------------------------------------------------ the analysis of the web app */
  var KEYS = ["f1", "precision", "recall", "mcc", "auc", "pf", "accuracy"];
  function round4(v) { return v == null || !isFinite(v) ? null : Math.round(v * 1e4) / 1e4; }
  function pick(m) { var o = {}; KEYS.forEach(function (k) { o[k] = round4(m[k]); }); return o; }
  function conf(m) { return { tp: m.tp, fp: m.fp, fn: m.fn, tn: m.tn }; }

  function analyze(ds, opts, progress) {
    opts = opts || {};
    var model = opts.model || "knn", gens = opts.generations == null ? 15 : opts.generations, popN = opts.population || 20, seed = opts.seed || 0;
    var say = progress || function () {}, t0 = Date.now();
    if (!MODELS[model]) throw new Error("model '" + model + "' is only available with the Python engine");
    say({ stage: "processing", progress: 0.02 });
    var sp = stratifiedSplit(ds.y, 0.3, rng(seed + 101)), tr = sp.train, te = sp.test;
    var ytr = take(ds.y, tr), yte = take(ds.y, te);
    if (ytr.reduce(function (a, b) { return a + b; }, 0) < 3 || yte.reduce(function (a, b) { return a + b; }, 0) < 1)
      throw new Error("not enough defective modules to train and test (need at least 4)");
    var Xtr = take(ds.X, tr), Xte = take(ds.X, te);
    say({ stage: "default", progress: 0.08 });
    var o = trainPredict(model, {}, Xtr, ytr, Xte, null, seed);
    var def = evaluate(yte, o.pred, o.scores);
    var ga = new GA(model, { population: popN, generations: gens, seed: seed }).run(Xtr, ytr, function (h) {
      say({ stage: "ga", progress: 0.1 + 0.8 * h.generation / Math.max(1, gens), generation: h.generation, best: h.best_f1, mean: h.mean, features: h.features });
    });
    say({ stage: "final", progress: 0.92 });
    o = trainPredict(model, ga.params, Xtr, ytr, Xte, ga.mask, seed);
    var opt = evaluate(yte, o.pred, o.scores);
    var risk = new Array(ds.y.length);
    stratifiedFolds(ds.y, 5, rng(seed + 303)).forEach(function (fold, f) {
      var inF = {}; fold.forEach(function (i) { inF[i] = true; });
      var trn = []; ds.y.forEach(function (_, i) { if (!inF[i]) trn.push(i); });
      var oo = trainPredict(model, ga.params, take(ds.X, trn), take(ds.y, trn), take(ds.X, fold), ga.mask, seed + f);
      fold.forEach(function (i, q) { risk[i] = oo.scores[q]; });
    });
    var locs = ds.loc || ds.y.map(function () { return 0; });   // equal risk: the smaller class first (cheaper to review)
    var order = risk.map(function (_, i) { return i; }).sort(function (a, b) { return risk[b] - risk[a] || locs[a] - locs[b] || a - b; });
    var selected = ds.features.filter(function (_, j) { return ga.mask[j]; });
    say({ stage: "done", progress: 1 });
    return {
      engine: "browser", dataset: summary(ds), model: model, model_label: MODELS[model].label,
      split: { train: tr.length, test: te.length, test_defective: yte.reduce(function (a, b) { return a + b; }, 0), seed: seed },
      default: { metrics: pick(def), confusion: conf(def), params: describe(model, {}), features: ds.features },
      optimized: { metrics: pick(opt), confusion: conf(opt), params: describe(model, ga.params), features: selected },
      ga: { history: ga.history, evaluations: ga.evaluations, inner_cv_f1: round4(ga.cv_f1), population: popN, generations: gens },
      ranking: order.map(function (i) { return { module: ds.modules[i], risk: round4(risk[i]), actual: ds.y[i], loc: ds.loc ? ds.loc[i] : null }; }),
      effort: ds.loc ? { model: round4(recallAtEffort(ds.y, risk, ds.loc)), smallest_first: round4(recallAtEffort(ds.y, ds.y.map(function () { return 1; }), ds.loc)) } : null,
      seconds: (Date.now() - t0) / 1000
    };
  }

  var API = { rng: rng, parse: parse, summary: summary, Preprocessor: Preprocessor, smote: smote, KNN: KNN, LogReg: LogReg, GNB: GNB,
              MODELS: MODELS, evaluate: evaluate, auc: auc, recallAtEffort: recallAtEffort, stratifiedFolds: stratifiedFolds,
              stratifiedSplit: stratifiedSplit, trainPredict: trainPredict, GA: GA, analyze: analyze, describe: describe };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.BPML = API;
})(typeof self !== "undefined" ? self : this);
