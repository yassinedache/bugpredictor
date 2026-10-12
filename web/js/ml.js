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

  /* Support vector machine with an RBF kernel, trained like libsvm (the solver inside scikit-learn's SVC):
     SMO on the dual problem, second-order working-set selection (Fan, Chen & Lin, 2005), stopping tolerance 1e-3. */
  function SVC(p) { this.C = p.C; this.gamma = p.gamma; }
  SVC.prototype.fit = function (X, y01) {
    var n = X.length, d = X[0].length, C = this.C, TAU = 1e-12, EPS = 1e-3, i, j, k;
    var gamma = this.gamma;
    if (gamma === "scale" || gamma == null) {               // scikit-learn: 1 / (n_features * X.var())
      var m = 0, s2 = 0;
      for (i = 0; i < n; i++) for (j = 0; j < d; j++) m += X[i][j];
      m /= n * d;
      for (i = 0; i < n; i++) for (j = 0; j < d; j++) s2 += (X[i][j] - m) * (X[i][j] - m);
      var v = s2 / (n * d);
      gamma = v > 0 ? 1 / (d * v) : 1;
    }
    var flat = new Float64Array(n * d), sq = new Float64Array(n), y = new Int8Array(n);
    for (i = 0; i < n; i++) {
      var s = 0;
      for (j = 0; j < d; j++) { var x = X[i][j]; flat[i * d + j] = x; s += x * x; }
      sq[i] = s; y[i] = y01[i] ? 1 : -1;
    }
    var cache = new Map(), maxRows = Math.max(2, Math.floor(1.5e7 / n));   // about 120 MB of kernel rows at most
    function row(a) {                                       // Q[a][t] = y_a y_t K(x_a, x_t)
      var r = cache.get(a);
      if (r) return r;
      r = new Float64Array(n);
      var oa = a * d;
      for (var t = 0; t < n; t++) {
        var dot = 0, ot = t * d;
        for (var q = 0; q < d; q++) dot += flat[oa + q] * flat[ot + q];
        r[t] = y[a] * y[t] * Math.exp(-gamma * (sq[a] + sq[t] - 2 * dot));
      }
      if (cache.size >= maxRows) cache.delete(cache.keys().next().value);
      cache.set(a, r);
      return r;
    }
    var alpha = new Float64Array(n), G = new Float64Array(n).fill(-1), maxIter = Math.max(1e6, 100 * n);
    for (var it = 0; it < maxIter; it++) {
      var Gmax = -Infinity, ii = -1;
      for (k = 0; k < n; k++) {
        if (y[k] === 1) { if (alpha[k] < C && -G[k] >= Gmax) { Gmax = -G[k]; ii = k; } }
        else if (alpha[k] > 0 && G[k] >= Gmax) { Gmax = G[k]; ii = k; }
      }
      var Qi = ii >= 0 ? row(ii) : null, Gmax2 = -Infinity, jj = -1, best = Infinity;
      for (k = 0; k < n; k++) {
        var gd, quad;
        if (y[k] === 1) {
          if (alpha[k] > 0) {
            gd = Gmax + G[k];
            if (G[k] >= Gmax2) Gmax2 = G[k];
            if (gd > 0) { quad = 2 - 2 * y[ii] * Qi[k]; if (quad <= 0) quad = TAU; if (-(gd * gd) / quad <= best) { best = -(gd * gd) / quad; jj = k; } }
          }
        } else if (alpha[k] < C) {
          gd = Gmax - G[k];
          if (-G[k] >= Gmax2) Gmax2 = -G[k];
          if (gd > 0) { quad = 2 + 2 * y[ii] * Qi[k]; if (quad <= 0) quad = TAU; if (-(gd * gd) / quad <= best) { best = -(gd * gd) / quad; jj = k; } }
        }
      }
      if (Gmax + Gmax2 < EPS || jj < 0) break;
      i = ii; j = jj;
      var Qj = row(j), ai = alpha[i], aj = alpha[j], oi = ai, oj = aj, qd, delta;
      if (y[i] !== y[j]) {
        qd = 2 + 2 * Qi[j]; if (qd <= 0) qd = TAU;
        delta = (-G[i] - G[j]) / qd;
        var diff = ai - aj;
        ai += delta; aj += delta;
        if (diff > 0) { if (aj < 0) { aj = 0; ai = diff; } } else if (ai < 0) { ai = 0; aj = -diff; }
        if (diff > 0) { if (ai > C) { ai = C; aj = C - diff; } } else if (aj > C) { aj = C; ai = C + diff; }
      } else {
        qd = 2 - 2 * Qi[j]; if (qd <= 0) qd = TAU;
        delta = (G[i] - G[j]) / qd;
        var sum = ai + aj;
        ai -= delta; aj += delta;
        if (sum > C) { if (ai > C) { ai = C; aj = sum - C; } } else if (aj < 0) { aj = 0; ai = sum; }
        if (sum > C) { if (aj > C) { aj = C; ai = sum - C; } } else if (ai < 0) { ai = 0; aj = sum; }
      }
      alpha[i] = ai; alpha[j] = aj;
      var di = ai - oi, dj = aj - oj;
      for (k = 0; k < n; k++) G[k] += Qi[k] * di + Qj[k] * dj;
    }
    var free = 0, sumFree = 0, ub = Infinity, lb = -Infinity;
    for (k = 0; k < n; k++) {
      var yG = y[k] * G[k];
      if (alpha[k] >= C) { if (y[k] === -1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else if (alpha[k] <= 0) { if (y[k] === 1) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else { free++; sumFree += yG; }
    }
    this.rho = free > 0 ? sumFree / free : (ub + lb) / 2;
    this.gammaUsed = gamma;
    this.sv = []; this.coef = [];
    for (k = 0; k < n; k++) if (alpha[k] > 0) { this.sv.push(X[k]); this.coef.push(alpha[k] * y[k]); }
    return this;
  };
  SVC.prototype.decision = function (Q) {
    var sv = this.sv, coef = this.coef, g = this.gammaUsed, rho = this.rho;
    return Q.map(function (q) {
      var s = 0;
      for (var a = 0; a < sv.length; a++) {
        var x = sv[a], dist = 0;
        for (var j = 0; j < q.length; j++) { var t = q[j] - x[j]; dist += t * t; }
        s += coef[a] * Math.exp(-g * dist);
      }
      return s - rho;
    });
  };
  SVC.prototype.scores = SVC.prototype.decision;
  SVC.prototype.predict = function (Q) { return this.decision(Q).map(function (v) { return v > 0 ? 1 : 0; }); };

  /* Random forest as in scikit-learn: bootstrap samples, CART trees grown with the Gini criterion on
     sqrt(n_features) random metrics per split, exact thresholds (midpoints between sorted values).
     Each metric is sorted once per forest; nodes split these sorted lists (no re-sorting). */
  function Forest(p, seed) {
    this.nTrees = p.n_estimators; this.maxDepth = p.max_depth == null ? Infinity : p.max_depth;
    this.minLeaf = p.min_samples_leaf; this.bootstrap = p.bootstrap !== false; this.mtry = p.max_features;
    this.seed = seed || 0;
  }
  Forest.prototype.fit = function (X, y01) {
    var n = X.length, d = X[0].length, r = rng(this.seed * 7919 + 13), self = this, f, i, k;
    var mtry = this.mtry === "all" ? d : Math.max(1, Math.floor(Math.sqrt(d)));
    var V = new Float64Array(d * n), order = [];               // column-major values, one sorted order per metric
    for (f = 0; f < d; f++) {
      for (i = 0; i < n; i++) V[f * n + i] = X[i][f];
      var o = new Uint32Array(n), base0 = f * n;
      for (i = 0; i < n; i++) o[i] = i;
      order.push(o.sort(function (a, b) { return V[base0 + a] - V[base0 + b] || a - b; }));
    }
    var y = new Uint8Array(n);
    for (i = 0; i < n; i++) y[i] = y01[i] ? 1 : 0;
    // S[f*n + lo .. hi) = the rows of the current node, sorted by metric f; nodes split these ranges in place
    var S = new Uint32Array(d * n), tmp = new Uint32Array(n), goLeft = new Uint8Array(n), w = new Float64Array(n), feats = new Uint32Array(d);
    var T;                                                     // the tree being grown
    function build(lo, hi, depth) {
      var m = hi - lo, w0 = 0, w1 = 0, q;
      for (k = lo; k < hi; k++) { q = S[k]; if (y[q]) w1 += w[q]; else w0 += w[q]; }
      var id = T.f.length;
      T.f.push(-1); T.thr.push(0); T.l.push(0); T.r.push(0); T.p.push(w1 / (w0 + w1));
      if (depth >= self.maxDepth || m < 2 || m < 2 * self.minLeaf || w0 === 0 || w1 === 0) return id;
      for (k = 0; k < d; k++) feats[k] = k;
      var bestScore = -Infinity, bestF = -1, bestPos = -1, bestThr = 0, visited = 0, tot = w0 + w1;
      for (var drawn = 0; drawn < d && visited < mtry; drawn++) {      // draw metrics without replacement
        var pick = drawn + randint(r, d - drawn), t0 = feats[drawn];
        feats[drawn] = feats[pick]; feats[pick] = t0;
        f = feats[drawn];
        var base = f * n, vb = f * n;
        if (V[vb + S[base + hi - 1]] <= V[vb + S[base + lo]] + 1e-7) continue;   // constant here: does not count (as scikit-learn)
        visited++;
        var l0 = 0, l1 = 0;
        for (k = lo; k < hi - 1; k++) {
          var sk = S[base + k], vk = V[vb + sk], vn = V[vb + S[base + k + 1]];
          if (y[sk]) l1 += w[sk]; else l0 += w[sk];
          if (vn <= vk + 1e-7) continue;                              // scikit-learn's FEATURE_THRESHOLD
          var nl = k - lo + 1;
          if (nl < self.minLeaf || m - nl < self.minLeaf) continue;
          var wl = l0 + l1, wr = tot - wl, r0 = w0 - l0, r1 = w1 - l1;
          var score = (l0 * l0 + l1 * l1) / wl + (r0 * r0 + r1 * r1) / wr;   // = -(weighted Gini of the children) + const
          if (score > bestScore) {
            bestScore = score; bestF = f; bestPos = k;
            var th = vk / 2 + vn / 2;
            bestThr = (th === vn || th === Infinity) ? vk : th;
          }
        }
      }
      if (bestF < 0) return id;
      var bb = bestF * n, nLeft = bestPos - lo + 1;
      for (k = lo; k < hi; k++) goLeft[S[bb + k]] = k <= bestPos ? 1 : 0;
      for (f = 0; f < d; f++) {                                  // stable in-place partition of every sorted list
        var b2 = f * n, pl = b2 + lo, tz = 0;
        for (k = b2 + lo; k < b2 + hi; k++) { q = S[k]; if (goLeft[q]) S[pl++] = q; else tmp[tz++] = q; }
        S.set(tmp.subarray(0, tz), pl);
      }
      T.f[id] = bestF; T.thr[id] = bestThr;
      var left = build(lo, lo + nLeft, depth + 1);
      var right = build(lo + nLeft, hi, depth + 1);
      T.l[id] = left; T.r[id] = right;
      return id;
    }
    this.trees = [];
    for (var t = 0; t < this.nTrees; t++) {
      w.fill(0);
      if (this.bootstrap) for (i = 0; i < n; i++) w[randint(r, n)] += 1;
      else w.fill(1);
      var u = 0;
      for (f = 0; f < d; f++) {
        var of = order[f], c = 0;
        for (i = 0; i < n; i++) if (w[of[i]] > 0) S[f * n + c++] = of[i];
        u = c;
      }
      T = { f: [], thr: [], l: [], r: [], p: [] };
      build(0, u, 0);
      this.trees.push({ f: Int32Array.from(T.f), thr: Float64Array.from(T.thr), l: Int32Array.from(T.l), r: Int32Array.from(T.r), p: Float64Array.from(T.p) });
    }
    return this;
  };
  Forest.prototype.proba = function (Q) {
    var trees = this.trees;
    return Q.map(function (q) {
      var s = 0;
      for (var t = 0; t < trees.length; t++) {
        var T = trees[t], k = 0;
        while (T.f[k] >= 0) k = q[T.f[k]] <= T.thr[k] ? T.l[k] : T.r[k];
        s += T.p[k];
      }
      return s / trees.length;
    });
  };
  Forest.prototype.predict = function (Q) { return this.proba(Q).map(function (p) { return p > 0.5 ? 1 : 0; }); };

  [KNN, LogReg, GNB, Forest].forEach(function (M) { M.prototype.scores = M.prototype.proba; });

  function P(name, kind, low, high, choices) { return { name: name, kind: kind, low: low, high: high, choices: choices }; }
  var MODELS = {
    knn: { label: "K-nearest neighbours", make: function (p) { return new KNN(p); },
           default: { n_neighbors: 5, weights: "uniform", p: 2 },
           space: [P("n_neighbors", "int", 1, 30), P("weights", "choice", 0, 1, ["uniform", "distance"]), P("p", "choice", 0, 1, [1, 2])] },
    svm: { label: "Support vector machine (RBF)", make: function (p) { return new SVC(p); },
           default: { C: 1.0, gamma: "scale" }, space: [P("C", "log", 1e-2, 1e3), P("gamma", "log", 1e-4, 10.0)] },
    lr: { label: "Logistic regression", make: function (p) { return new LogReg(p); },
          default: { C: 1.0 }, space: [P("C", "log", 1e-3, 1e2)] },
    nb: { label: "Gaussian naive Bayes", make: function (p) { return new GNB(p); },
          default: { var_smoothing: 1e-9 }, space: [P("var_smoothing", "log", 1e-12, 1.0)] },
    rf: { label: "Random forest", make: function (p, seed) { return new Forest(p, seed); },
          default: { n_estimators: 100, max_depth: null, min_samples_leaf: 1 },
          space: [P("n_estimators", "int", 20, 200), P("max_depth", "int", 2, 20), P("min_samples_leaf", "int", 1, 10)] }
  };
  function decodeParam(p, u) {
    u = Math.min(Math.max(u, 0), 1);
    if (p.kind === "int") return Math.round(p.low + u * (p.high - p.low));
    if (p.kind === "log") return Math.pow(10, Math.log10(p.low) + u * (Math.log10(p.high) - Math.log10(p.low)));
    return p.choices[Math.min(Math.floor(u * p.choices.length), p.choices.length - 1)];
  }
  function encodeParam(p, v) {
    if (v == null || (typeof v === "string" && p.kind !== "choice")) return 0.5;   // gamma='scale', max_depth=None
    if (p.kind === "int") return (v - p.low) / (p.high - p.low);
    if (p.kind === "log") return (Math.log10(v) - Math.log10(p.low)) / (Math.log10(p.high) - Math.log10(p.low));
    return (p.choices.indexOf(v) + 0.5) / p.choices.length;
  }
  function withDefaults(model, params) { var o = {}, d = MODELS[model].default, k; for (k in d) o[k] = d[k]; for (k in params || {}) o[k] = params[k]; return o; }
  function describe(model, params) {
    var o = withDefaults(model, params);
    return Object.keys(o).map(function (k) {
      var v = o[k];
      return { name: k, value: v == null ? "None" : typeof v === "number" && !Number.isInteger(v) ? fmt3(v) : String(v) };
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
    var m = MODELS[model].make(withDefaults(model, params), seed).fit(res.X, res.y);
    return { pred: m.predict(B), scores: m.scores(B) };
  }
  /* A job = train on some rows of the dataset, predict others. Jobs are independent, so a pool of
     Web Workers can run them in parallel; runJob is what each worker executes. */
  function runJob(ds, job) {
    return trainPredict(job.model, job.params, take(ds.X, job.train), take(ds.y, job.train), take(ds.X, job.test), job.mask, job.seed);
  }
  function localPool(ds) { return { size: 1, run: function (job) { return Promise.resolve(runJob(ds, job)); } }; }

  /* ------------------------------------------------------------------ genetic algorithm */
  function GA(model, o) {
    o = o || {};
    this.model = model; this.space = MODELS[model].space;
    this.pop = o.population || 20; this.gens = o.generations == null ? 15 : o.generations;
    this.cx = 0.8; this.pm = 0.25; this.tour = 3; this.elite = 2; this.patience = 6;
    this.penalty = 0.005; this.inner = 5; this.seed = o.seed || 0;
  }
  GA.prototype.decode = function (genes) {
    var out = {};
    if (!genes) return withDefaults(this.model, {});
    this.space.forEach(function (p, i) { out[p.name] = decodeParam(p, genes[i]); });
    return out;
  };
  GA.prototype.defaultGenes = function () {
    var d = MODELS[this.model].default;
    return this.space.map(function (p) { return Math.min(1, Math.max(0, encodeParam(p, d[p.name]))); });
  };
  GA.prototype.folds = function (y) { return stratifiedFolds(y, this.inner, rng(this.seed + 17)); };
  /* The search itself, written as a generator: it yields the individuals it needs evaluated and
     receives their inner-CV F1, so the same code runs with local or parallel evaluation. */
  GA.prototype.search = function* (d, onGen) {
    var self = this, r = rng(this.seed * 2654435761 + 1), g = this.space.length, pbit = 1 / d, cache = {}, evals = 0;
    function repair(m) { if (!m.some(Boolean)) m[randint(r, d)] = true; return m; }
    function key(ind) { return ind.m.map(Number).join("") + JSON.stringify(self.decode(ind.g)); }
    function* score(pop) {
      var todo = [], seen = {};
      pop.forEach(function (ind) { var k = key(ind); if (!cache[k] && !seen[k]) { seen[k] = true; todo.push({ m: ind.m, g: ind.g, key: k, params: self.decode(ind.g) }); } });
      if (todo.length) {
        var f1s = yield todo;
        todo.forEach(function (t, i) {
          var nsel = t.m.filter(Boolean).length;
          cache[t.key] = { fit: f1s[i] - self.penalty * nsel, f1: f1s[i] };
          evals++;
        });
      }
      return pop.map(function (ind) { return cache[key(ind)]; });
    }
    var pop = [{ m: new Array(d).fill(true), g: null }];
    while (pop.length < this.pop) {
      var m = []; for (var j = 0; j < d; j++) m.push(r() < 0.5);
      var gn = []; for (j = 0; j < g; j++) gn.push(r());
      pop.push({ m: repair(m), g: gn });
    }
    var sc = yield* score(pop);
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
      sc = yield* score(pop);
    }
    var bi = 0; sc.forEach(function (s, i) { if (s.fit > sc[bi].fit) bi = i; });
    return { mask: pop[bi].m, params: this.decode(pop[bi].g), fitness: sc[bi].fit, cv_f1: sc[bi].f1, history: history, evaluations: evals };
  };
  /* Synchronous run on a feature matrix (used by the tests). */
  GA.prototype.run = function (X, y, onGen) {
    var self = this, folds = this.folds(y), it = this.search(X[0].length, onGen), step = it.next();
    while (!step.done) {
      step = it.next(step.value.map(function (ind) {
        var f1 = 0;
        folds.forEach(function (test, i) {
          var inT = {}; test.forEach(function (q) { inT[q] = true; });
          var tr = []; y.forEach(function (_, q) { if (!inT[q]) tr.push(q); });
          f1 += evaluate(take(y, test), trainPredict(self.model, ind.params, take(X, tr), take(y, tr), take(X, test), ind.m, self.seed + i).pred).f1;
        });
        return f1 / folds.length;
      }));
    }
    return step.value;
  };
  /* Asynchronous run on rows `rows` of the dataset held by the pool: one job per individual and inner fold. */
  GA.prototype.runAsync = async function (ds, rows, pool, onGen) {
    var self = this, y = take(ds.y, rows), folds = this.folds(y), d = ds.X[0].length;
    var splits = folds.map(function (test) {
      var inT = {}; test.forEach(function (q) { inT[q] = true; });
      var tr = []; y.forEach(function (_, q) { if (!inT[q]) tr.push(q); });
      return { train: take(rows, tr), test: take(rows, test), ytest: take(y, test) };
    });
    var it = this.search(d, onGen), step = it.next();
    while (!step.done) {
      var batch = step.value;
      var results = await Promise.all(batch.map(function (ind) {
        return Promise.all(splits.map(function (s, i) {
          return pool.run({ model: self.model, params: ind.params, train: s.train, test: s.test, mask: ind.m, seed: self.seed + i })
            .then(function (o) { return evaluate(s.ytest, o.pred).f1; });
        })).then(function (f) { return f.reduce(function (u, v) { return u + v; }, 0) / f.length; });
      }));
      step = it.next(results);
    }
    return step.value;
  };

  /* ------------------------------------------------------------------ the analysis of the web app */
  var KEYS = ["f1", "precision", "recall", "mcc", "auc", "pf", "accuracy"];
  function round4(v) { return v == null || !isFinite(v) ? null : Math.round(v * 1e4) / 1e4; }
  function pick(m) { var o = {}; KEYS.forEach(function (k) { o[k] = round4(m[k]); }); return o; }
  function conf(m) { return { tp: m.tp, fp: m.fp, fn: m.fn, tn: m.tn }; }
  function toUnit(s) {                     // decision values (SVM) -> (0, 1), as scores_to_unit in Python
    var inside = s.every(function (v) { return v >= 0 && v <= 1; });
    return inside ? s : s.map(sigmoid);
  }

  async function analyzeAsync(ds, opts, progress, pool) {
    opts = opts || {};
    pool = pool || localPool(ds);
    var model = opts.model || "knn", gens = opts.generations == null ? 15 : opts.generations, popN = opts.population || 20, seed = opts.seed || 0;
    var say = progress || function () {}, t0 = Date.now();
    if (!MODELS[model]) throw new Error("unknown model '" + model + "'");
    say({ stage: "processing", progress: 0.02 });
    var sp = stratifiedSplit(ds.y, 0.3, rng(seed + 101)), tr = sp.train, te = sp.test;
    var yte = take(ds.y, te), nTr1 = take(ds.y, tr).reduce(function (a, b) { return a + b; }, 0);
    if (nTr1 < 3 || yte.reduce(function (a, b) { return a + b; }, 0) < 1) throw new Error("not enough defective modules to train and test (need at least 4)");
    say({ stage: "default", progress: 0.08 });
    var o = await pool.run({ model: model, params: {}, train: tr, test: te, mask: null, seed: seed });
    var def = evaluate(yte, o.pred, o.scores);
    var ga = await new GA(model, { population: popN, generations: gens, seed: seed }).runAsync(ds, tr, pool, function (h) {
      say({ stage: "ga", progress: 0.1 + 0.8 * h.generation / Math.max(1, gens), generation: h.generation, best: h.best_f1, mean: h.mean, features: h.features });
    });
    say({ stage: "final", progress: 0.92 });
    var folds = stratifiedFolds(ds.y, 5, rng(seed + 303));
    var outs = await Promise.all([pool.run({ model: model, params: ga.params, train: tr, test: te, mask: ga.mask, seed: seed })].concat(folds.map(function (fold, f) {
      var inF = {}; fold.forEach(function (i) { inF[i] = true; });
      var trn = []; ds.y.forEach(function (_, i) { if (!inF[i]) trn.push(i); });
      return pool.run({ model: model, params: ga.params, train: trn, test: fold, mask: ga.mask, seed: seed + f });
    })));
    var opt = evaluate(yte, outs[0].pred, outs[0].scores);
    var raw = new Array(ds.y.length);
    folds.forEach(function (fold, f) { fold.forEach(function (i, q) { raw[i] = outs[f + 1].scores[q]; }); });
    var risk = toUnit(raw);
    var locs = ds.loc || ds.y.map(function () { return 0; });   // equal risk: the smaller class first (cheaper to review)
    var order = risk.map(function (_, i) { return i; }).sort(function (a, b) { return risk[b] - risk[a] || locs[a] - locs[b] || a - b; });
    var selected = ds.features.filter(function (_, j) { return ga.mask[j]; });
    say({ stage: "done", progress: 1 });
    return {
      engine: "browser", workers: pool.size, dataset: summary(ds), model: model, model_label: MODELS[model].label,
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
              SVC: SVC, Forest: Forest, MODELS: MODELS, evaluate: evaluate, auc: auc, recallAtEffort: recallAtEffort,
              stratifiedFolds: stratifiedFolds, stratifiedSplit: stratifiedSplit, trainPredict: trainPredict, runJob: runJob,
              localPool: localPool, GA: GA, analyzeAsync: analyzeAsync, describe: describe };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.BPML = API;
})(typeof self !== "undefined" ? self : this);
