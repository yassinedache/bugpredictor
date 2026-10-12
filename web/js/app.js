/* BugPredictor web app: routing, data choice, running an analysis (browser worker or Python API), results. */
(function () {
  "use strict";
  var I = window.I18N, t = I.t, $ = function (s) { return document.querySelector(s); };
  var BLUE = "#1a73e8", BLUE_BAR = "#a5c0f7", GREEN = "#4caf50", RED = "#d93025", GREY = "#9ca3af";
  var API = window.BP_API || null;
  var S = { engine: API ? "api" : "browser", manifest: [], source: "promise", selected: "ant-1.7", upload: null,
            model: "knn", gens: 15, pop: 20, seed: 0, pool: null, poll: null, result: null, runId: 0, charts: {}, research: null };
  var MODEL_LIST = ["knn", "lr", "nb", "svm", "rf"];
  var WORKERS = Math.max(1, Math.min(8, navigator.hardwareConcurrency || 2));

  /* A pool of Web Workers that run training jobs in parallel (see js/worker.js). */
  function Pool(n, ds) {
    var self = this;
    this.size = n; this.workers = []; this.idle = []; this.queue = []; this.pending = {}; this.next = 0;
    for (var i = 0; i < n; i++) {
      var w = new Worker("js/worker.js");
      w.postMessage({ type: "init", data: { X: ds.X, y: ds.y } });
      w.onmessage = function (e) {
        var p = self.pending[e.data.id];
        if (!p) return;
        delete self.pending[e.data.id];
        self.idle.push(p.worker);
        if (e.data.type === "result") p.resolve({ pred: e.data.pred, scores: e.data.scores });
        else p.reject(new Error(e.data.message));
        self.pump();
      };
      w.onerror = function (e) { Object.keys(self.pending).forEach(function (k) { self.pending[k].reject(new Error(e.message || "worker error")); }); };
      this.workers.push(w); this.idle.push(w);
    }
  }
  Pool.prototype.run = function (job) {
    var self = this;
    return new Promise(function (resolve, reject) { self.queue.push({ job: job, resolve: resolve, reject: reject }); self.pump(); });
  };
  Pool.prototype.pump = function () {
    while (this.idle.length && this.queue.length) {
      var w = this.idle.pop(), t = this.queue.shift(), id = ++this.next;
      this.pending[id] = { worker: w, resolve: t.resolve, reject: t.reject };
      w.postMessage({ type: "job", id: id, job: t.job });
    }
  };
  Pool.prototype.terminate = function () { this.workers.forEach(function (w) { w.terminate(); }); this.queue = []; this.pending = {}; };

  /* ---------------------------------------------------------------- helpers */
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function chart(id, cfg) {
    if (!window.Chart) return null;
    if (S.charts[id]) S.charts[id].destroy();
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.color = "#6b7280";
    cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: false }, cfg.options || {});
    return (S.charts[id] = new Chart(document.getElementById(id), cfg));
  }
  function modelName(m) { return t("m_" + m); }
  function fetchText(url) { return fetch(url).then(function (r) { if (!r.ok) throw new Error(url + ": " + r.status); return r.text(); }); }
  function fetchJSON(url, opts) {
    return fetch(url, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) { if (!r.ok) throw new Error(j.detail || (url + ": " + r.status)); return j; });
    });
  }

  /* ---------------------------------------------------------------- routing */
  function route() {
    var h = (location.hash || "#/").replace(/^#\/?/, "").split("?")[0] || "home";
    if (["home", "analyze", "research"].indexOf(h) < 0) h = "home";
    document.querySelectorAll("[data-view]").forEach(function (v) { v.hidden = v.getAttribute("data-view") !== h; });
    document.querySelectorAll("[data-route]").forEach(function (a) { a.classList.toggle("active", a.getAttribute("data-route") === h); });
    if (h === "research") renderResearch();
    window.scrollTo(0, 0);
  }

  /* ---------------------------------------------------------------- data choice */
  function renderDatasets() {
    $("#dsGrid").innerHTML = S.manifest.map(function (d) {
      return '<button type="button" class="ds" data-ds="' + esc(d.name) + '" aria-pressed="' + (d.name === S.selected) + '">' +
        "<b>" + esc(d.name) + "</b><small>" + t("dsCard", { n: I.int(d.modules), p: I.pct(d.defect_rate) }) + "</small>" +
        '<span class="bar"><i style="width:' + Math.round(100 * d.defect_rate) + '%"></i></span></button>';
    }).join("");
  }
  function setSource(src) {
    S.source = src;
    document.querySelectorAll("[data-src]").forEach(function (b) { b.setAttribute("aria-selected", b.getAttribute("data-src") === src ? "true" : "false"); });
    $("#src-promise").hidden = src !== "promise";
    $("#src-upload").hidden = src !== "upload";
    updateStart();
  }
  function showFile(name, sum) {
    $("#fileInfo").hidden = false;
    $("#drop").hidden = true;
    $("#fileName").textContent = name;
    $("#fileMeta").textContent = t("fileMeta", { n: I.int(sum.modules), f: sum.features, d: I.int(sum.defective), p: I.pct(sum.defect_rate), t: sum.target });
  }
  function clearFile() {
    S.upload = null;
    $("#file").value = "";
    $("#fileInfo").hidden = true;
    $("#drop").hidden = false;
    $("#fileError").hidden = true;
    updateStart();
  }
  function fileError(msg) { $("#fileError").textContent = msg; $("#fileError").hidden = false; }
  function readFile(file) {
    $("#fileError").hidden = true;
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) { fileError("10 MB max."); return; }
    var reader = new FileReader();
    reader.onload = function () {
      var text = String(reader.result), name = file.name.replace(/\.[^.]+$/, "");
      var sum;
      try { sum = BPML.summary(BPML.parse(text, name)); } catch (e) { fileError(e.message); return; }
      if (S.engine === "api") {
        var fd = new FormData(); fd.append("file", file);
        fetchJSON(API + "/upload", { method: "POST", body: fd }).then(function (j) {
          S.upload = { name: name, id: j.id, summary: j }; showFile(file.name, j); updateStart();
        }).catch(function (e) { fileError(e.message); });
      } else {
        S.upload = { name: name, text: text, summary: sum }; showFile(file.name, sum); updateStart();
      }
    };
    reader.onerror = function () { fileError("Could not read the file."); };
    reader.readAsText(file);
  }
  function renderModels() {
    $("#models").innerHTML = MODEL_LIST.map(function (m) {
      return '<button type="button" class="model" data-model="' + m + '" aria-pressed="' + (m === S.model) + '">' +
        "<b>" + esc(modelName(m)) + "</b><small>" + esc(t("m_" + m + "_d")) + "</small></button>";
    }).join("");
    $("#engineNote").textContent = t(S.engine === "api" ? "engineApi" : "engineBrowser", { n: WORKERS });
  }
  function updateStart() {
    var ready = S.source === "promise" ? !!S.selected : !!S.upload;
    $("#startBtn").disabled = !ready;
    $("#startHint").textContent = ready ? "" : t("pickData");
  }

  /* ---------------------------------------------------------------- running */
  var STAGES = ["processing", "default", "ga", "done"];
  function showProgress(ev) {
    var stage = ev.stage === "final" ? "ga" : ev.stage, idx = STAGES.indexOf(stage);
    document.querySelectorAll(".pstep").forEach(function (el, i) {
      el.classList.toggle("done", i < idx || stage === "done");
      el.classList.toggle("active", i === idx && stage !== "done");
    });
    var fill = stage === "processing" ? 0 : stage === "default" ? 33 : stage === "ga" ? 33 + 33 * Math.min(1, (ev.generation || 0) / S.gens) : 100;
    if (ev.stage === "final") fill = 80;
    $("#trackFill").style.width = fill + "%";
    var msg = ev.stage === "ga" ? (ev.generation ? t("pg_ga", { g: ev.generation, n: S.gens, b: I.num(ev.best, 3) }) : t("pg_ga0")) : t("pg_" + ev.stage);
    $("#progMsg").textContent = msg;
    $("#progPct").textContent = t("pct", { p: Math.round(100 * (ev.progress || 0)) });
  }
  function liveChart(hist) {
    var c = S.charts.liveChart;
    if (!c) {
      chart("liveChart", { type: "line", data: { labels: [], datasets: [
        { label: t("liveBest"), data: [], borderColor: GREEN, backgroundColor: GREEN, pointRadius: 3, borderWidth: 2.5, tension: 0.2 },
        { label: t("liveMean"), data: [], borderColor: BLUE_BAR, backgroundColor: BLUE_BAR, pointRadius: 0, borderWidth: 2, tension: 0.2 }] },
        options: { scales: { x: { title: { display: true, text: t("generation") } } },
                   plugins: { legend: { labels: { boxWidth: 10, usePointStyle: true } } } } });
      c = S.charts.liveChart;
    }
    if (!c) return;
    var labels = [], best = [], mean = [];          // fixed x axis: generations 0 .. n
    for (var g = 0; g <= S.gens; g++) { labels.push(g); best.push(null); mean.push(null); }
    hist.forEach(function (h) { best[h.generation] = h.best; mean[h.generation] = h.mean; });
    c.data.labels = labels; c.data.datasets[0].data = best; c.data.datasets[1].data = mean;
    c.update();
  }
  function stopRun() {
    S.runId++;
    if (S.pool) { S.pool.terminate(); S.pool = null; }
    if (S.poll) { clearTimeout(S.poll); S.poll = null; }
  }
  function start(newSeed) {
    stopRun();
    var run = S.runId;
    if (newSeed) { S.seed = (S.seed + 1) % 21; $("#seed").value = S.seed; $("#seedOut").textContent = S.seed; }
    var name = S.source === "promise" ? S.selected : S.upload.name;
    $("#setup").hidden = true; $("#results").hidden = true; $("#progress").hidden = false;
    $("#progName").textContent = name;
    $("#progMeta").textContent = t("progMeta", { model: modelName(S.model), gens: S.gens, pop: S.pop });
    if (S.charts.liveChart) { S.charts.liveChart.destroy(); delete S.charts.liveChart; }
    var hist = [];
    showProgress({ stage: "processing", progress: 0 });
    liveChart(hist);
    window.scrollTo(0, 0);
    var opts = { model: S.model, generations: S.gens, population: S.pop, seed: S.seed };
    function onEvent(ev) {
      if (run !== S.runId) return;
      showProgress(ev);
      if (ev.stage === "ga") { hist.push({ generation: ev.generation, best: ev.best, mean: ev.mean }); liveChart(hist); }
    }
    function onDone(res) {
      if (run !== S.runId) return;
      showProgress({ stage: "done", progress: 1 });
      S.result = res;
      setTimeout(function () { if (run === S.runId) showResults(); }, 700);
    }
    function onError(msg) {
      if (run !== S.runId) return;
      stopRun();
      $("#progress").hidden = true; $("#setup").hidden = false;
      $("#startHint").textContent = t("errPrefix") + msg;
    }
    if (S.engine === "api") {
      var body = { dataset: S.source === "promise" ? S.selected : S.upload.id, model: S.model, generations: S.gens, population: S.pop, seed: S.seed };
      fetchJSON(API + "/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(function (j) {
        var seen = 0;
        (function tick() {
          fetchJSON(API + "/jobs/" + j.job + "?ranking=0").then(function (st) {
            if (run !== S.runId) return;
            (st.history || []).slice(seen).forEach(function (h) { onEvent({ stage: "ga", progress: st.progress, generation: h.generation, best: h.best, mean: h.mean }); });
            seen = (st.history || []).length;
            if (st.stage !== "ga") onEvent({ stage: st.stage, progress: st.progress });
            if (st.status === "error") return onError(st.error);
            if (st.status === "done") return fetchJSON(API + "/jobs/" + j.job + "?ranking=1000000").then(function (f) { onDone(f.result); });
            S.poll = setTimeout(tick, 400);
          }).catch(function (e) { onError(e.message); });
        })();
      }).catch(function (e) { onError(e.message); });
      return;
    }
    var textP = S.source === "promise" ? fetchText("data/promise/" + S.selected + ".csv") : Promise.resolve(S.upload.text);
    textP.then(function (text) {
      if (run !== S.runId) return;
      var ds = BPML.parse(text, name);
      S.pool = new Pool(WORKERS, ds);
      return BPML.analyzeAsync(ds, opts, onEvent, S.pool).then(function (res) {
        if (S.pool) { S.pool.terminate(); S.pool = null; }
        onDone(res);
      });
    }).catch(function (e) { onError(e.message); });
  }

  /* ---------------------------------------------------------------- results */
  var BAR_KEYS = ["f1", "precision", "recall", "mcc", "auc"];
  var PARAM_LABEL = {
    en: { n_neighbors: "k (neighbours)", weights: "Weights", p: "Distance", C: "C (regularisation)", gamma: "Gamma", var_smoothing: "Smoothing",
          n_estimators: "Trees", max_depth: "Max depth", min_samples_leaf: "Min. leaf size", metrics: "Metrics used" },
    fr: { n_neighbors: "k (voisins)", weights: "Poids", p: "Distance", C: "C (régularisation)", gamma: "Gamma", var_smoothing: "Lissage",
          n_estimators: "Arbres", max_depth: "Profondeur max", min_samples_leaf: "Feuille min.", metrics: "Métriques utilisées" }
  };
  function paramValue(name, v) {
    if (name === "p") return v === "1" ? "Manhattan" : I.lang === "fr" ? "Euclidienne" : "Euclidean";
    if (name === "weights") return v === "uniform" ? (I.lang === "fr" ? "uniformes" : "uniform") : (I.lang === "fr" ? "inverse distance" : "1 / distance");
    if (/^[0-9.]+(e[-+]?\d+)?$/i.test(v) && I.lang === "fr") return v.replace(".", ",");
    return v;
  }
  function paramsHTML(params, nFeat, total) {
    var L = PARAM_LABEL[I.lang] || PARAM_LABEL.en;
    return params.map(function (p) { return "<div><small>" + esc(L[p.name] || p.name) + "</small><b>" + esc(paramValue(p.name, p.value)) + "</b></div>"; }).join("") +
      "<div><small>" + L.metrics + "</small><b>" + nFeat + " / " + total + "</b></div>";
  }
  function cmHTML(c) {
    return '<div class="good"><small>' + t("cmTP") + "</small><b>" + c.tp + '</b></div><div class="bad"><small>' + t("cmFN") + "</small><b>" + c.fn +
      '</b></div><div class="bad"><small>' + t("cmFP") + "</small><b>" + c.fp + "</b></div><div><small>" + t("cmTN") + "</small><b>" + c.tn + "</b></div>";
  }
  function barChart(id, metrics, color) {
    var vals = BAR_KEYS.map(function (k) { return metrics[k]; });
    var lo = Math.min(0, Math.min.apply(null, vals.map(function (v) { return v == null ? 0 : v; })));
    chart(id, { type: "bar", data: { labels: BAR_KEYS.map(function (k) { return t("metric_" + k); }),
      datasets: [{ data: vals, backgroundColor: color, borderRadius: 6, maxBarThickness: 56 }] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (c) { return " " + I.num(c.raw, 3); } } } },
                 scales: { y: { min: lo < 0 ? -0.2 : 0, max: 1, ticks: { stepSize: 0.2 } }, x: { grid: { display: false } } } } });
  }
  function shortModule(m) {
    var i = m.lastIndexOf(".");
    if (i < 0 || m.length < 28) return "<b>" + esc(m) + "</b>";
    return "<span>" + esc(m.slice(0, i + 1)) + "</span><b>" + esc(m.slice(i + 1)) + "</b>";
  }
  function showResults() {
    var r = S.result;
    $("#progress").hidden = true; $("#setup").hidden = true; $("#results").hidden = false;
    renderResults();
    window.scrollTo(0, 0);
  }
  function renderResults() {
    var r = S.result;
    if (!r) return;
    var d = r.default.metrics, o = r.optimized.metrics, nF = r.optimized.features.length, total = r.default.features.length;
    $("#resTitle").textContent = t("resTitle", { ds: r.dataset.name, model: modelName(r.model) });
    $("#resChips").innerHTML = [t("chipSplit", { tr: r.split.train, te: r.split.test, d: r.split.test_defective }),
      t("chipEngine", { e: r.engine === "python" ? t("engPython") : t("engBrowser", { n: r.workers || 1 }) }), t("chipEvals", { n: r.ga.evaluations }),
      t("chipTime", { s: I.num(r.seconds, 1) })].map(function (c) { return '<span class="chip">' + esc(c) + "</span>"; }).join("");
    var diff = o.f1 - d.f1, kind = Math.abs(diff) < 0.01 ? "same" : diff > 0 ? "good" : "bad";
    var vars = { n: r.split.test, o: I.num(o.f1, 3), d: I.num(d.f1, 3), diff: (diff >= 0 ? "+" : "−") + I.num(Math.abs(diff), 3), k: nF, m: total };
    var icon = kind === "bad" ? '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
      : '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>';
    $("#verdict").className = "verdict " + kind;
    $("#verdict").innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">' + icon + "</svg><p style=\"margin:0\">" +
      esc(t(kind === "good" ? "vGood" : kind === "bad" ? "vBad" : "vSame", vars)) + "</p>";
    var badge = $("#optBadge");
    badge.textContent = t(kind === "bad" ? "notBetter" : "recommended");
    badge.className = "badge" + (kind === "bad" ? " neutral" : "");
    barChart("defChart", d, BLUE_BAR);
    barChart("optChart", o, GREEN);
    $("#defParams").innerHTML = paramsHTML(r.default.params, total, total);
    $("#optParams").innerHTML = paramsHTML(r.optimized.params, nF, total);
    $("#defCM").innerHTML = cmHTML(r.default.confusion);
    $("#optCM").innerHTML = cmHTML(r.optimized.confusion);
    var h = r.ga.history;
    chart("convChart", { type: "line", data: { labels: h.map(function (x) { return x.generation; }), datasets: [
      { label: t("convF1"), data: h.map(function (x) { return x.best_f1; }), borderColor: GREEN, backgroundColor: GREEN, pointRadius: 3, borderWidth: 2.5 },
      { label: t("convBest"), data: h.map(function (x) { return x.best; }), borderColor: BLUE, backgroundColor: BLUE, pointRadius: 0, borderWidth: 2, borderDash: [5, 4] },
      { label: t("convMean"), data: h.map(function (x) { return x.mean; }), borderColor: BLUE_BAR, backgroundColor: BLUE_BAR, pointRadius: 0, borderWidth: 2 }] },
      options: { scales: { x: { title: { display: true, text: t("generation") } } }, plugins: { legend: { labels: { boxWidth: 10, usePointStyle: true } } } } });
    var sel = {}; r.optimized.features.forEach(function (f) { sel[f] = true; });
    $("#featSub").textContent = t("featSub", { k: nF, m: total });
    $("#featChips").innerHTML = r.default.features.map(function (f) {
      return '<span class="feat ' + (sel[f] ? "on" : "off") + '" title="' + esc(I.gloss(f) || f) + '">' + esc(f) + "</span>";
    }).join("");
    $("#effort").innerHTML = r.effort ? t("effort", { m: I.pct(r.effort.model), b: I.pct(r.effort.smallest_first) }) : t("noLoc");
    var rows = r.ranking.slice(0, 15);
    $("#rankTable").innerHTML = "<thead><tr><th>" + t("thRank") + "</th><th>" + t("thModule") + '</th><th class="num">' + t("thLoc") + "</th><th>" + t("thRisk") +
      "</th><th>" + t("thActual") + "</th></tr></thead><tbody>" + rows.map(function (x, i) {
        return "<tr><td>" + (i + 1) + '</td><td class="mod">' + shortModule(x.module) + '</td><td class="num">' + (x.loc == null ? "–" : I.int(x.loc)) +
          '</td><td><span class="riskbar"><i style="width:' + Math.round(100 * x.risk) + '%"></i></span>' + I.num(x.risk, 2) + "</td><td>" +
          (x.actual ? '<span class="yes">' + t("yes") + "</span>" : '<span class="no">' + t("no") + "</span>") + "</td></tr>";
      }).join("") + "</tbody>";
  }
  function downloadCSV() {
    var r = S.result;
    if (!r) return;
    var lines = ["rank,module,risk,loc,actually_defective"].concat(r.ranking.map(function (x, i) {
      return [i + 1, '"' + String(x.module).replace(/"/g, '""') + '"', x.risk, x.loc == null ? "" : x.loc, x.actual].join(",");
    }));
    var blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "bugpredictor-" + r.dataset.name + "-" + r.model + ".csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  /* ---------------------------------------------------------------- research page */
  function renderResearch() {
    var box = $("#research");
    if (S.research === null) {
      S.research = false;
      fetchJSON("data/research.json").then(function (j) { S.research = j; renderResearch(); })
        .catch(function () { box.innerHTML = '<p class="hint" style="margin-top:40px">' + t("rPending") + "</p>"; });
      return;
    }
    var R = S.research;
    if (!R) return;
    var L = R.leakage.mean, models = R.baseline.models;
    var f = function (x) { return I.num(x, 2); };
    var html = '<div class="research"><h1 class="page-title">' + t("rTitle") + '</h1><p class="page-sub">' + t("rLead") + "</p>" +
      "<h2>" + t("rLeakTitle") + "</h2><p>" + t("rLeakText") + "</p>" +
      '<div class="kpis">' +
      '<div class="kpi"><small>' + t("rKpiLeaky") + '</small><b class="red">' + f(L.leaky_ga) + "</b><span>" + t("rAvg") + "</span></div>" +
      '<div class="kpi"><small>' + t("rKpiHonest") + '</small><b class="green">' + f(L.honest_ga) + "</b><span>" + t("rAvg") + "</span></div>" +
      '<div class="kpi"><small>' + t("rKpiK1") + '</small><b class="red">' + f(L.knn1_leaky) + "</b><span>" + t("rAvg") + "</span></div>" +
      '<div class="kpi"><small>' + t("rKpiK1h") + '</small><b class="green">' + f(L.knn1_honest) + "</b><span>" + t("rAvg") + "</span></div></div>" +
      '<div class="card"><h2 style="margin:0 0 10px">' + t("rLeakChart") + '</h2><div class="chart-box tall"><canvas id="leakChart"></canvas></div></div>' +
      "<h2>" + t("rGaTitle") + "</h2><p>" + t("rGaText") + "</p>" +
      '<div class="card table-wrap"><table class="data"><thead><tr><th>' + t("thModel") + '</th><th class="num">' + t("thDefF1") + '</th><th class="num">' + t("thGaF1") +
      '</th><th class="num">' + t("thWins") + '</th><th class="num">' + t("thFeat") + '</th><th class="num">' + t("thInner") + "</th></tr></thead><tbody>" +
      R.ga.map(function (g) {
        var dlt = g.ga_f1 - g.default_f1;
        return "<tr><td>" + esc(modelName(g.model)) + '</td><td class="num">' + f(g.default_f1) + '</td><td class="num">' + f(g.ga_f1) +
          ' <span class="pill ' + (dlt >= 0 ? "up" : "down") + '">' + (dlt >= 0 ? "+" : "−") + I.num(Math.abs(dlt), 3) + '</span></td><td class="num">' + g.wins + " / " + g.n +
          '</td><td class="num">' + I.num(g.features, 1) + " / " + g.total + '</td><td class="num">' + f(g.inner_cv) + "</td></tr>";
      }).join("") + '</tbody></table><p class="hint">' + t("rInnerNote") + "</p></div>" +
      "<h2>" + t("rBaseTitle") + "</h2><p>" + t("rBaseText") + "</p>" +
      '<div class="card table-wrap"><table class="data"><thead><tr><th></th>' + models.map(function (m) { return '<th class="num">' + esc(modelName(m)) + "</th>"; }).join("") +
      "</tr></thead><tbody>" + ["f1", "mcc", "auc", "recall_20"].map(function (k) {
        var best = Math.max.apply(null, models.map(function (m) { return R.baseline.mean[k][m]; }));
        return "<tr><td>" + (k === "recall_20" ? "Recall@20%LOC" : t("metric_" + k)) + "</td>" + models.map(function (m) {
          var v = R.baseline.mean[k][m];
          return '<td class="num">' + (v === best ? "<b>" + f(v) + "</b>" : f(v)) + "</td>";
        }).join("") + "</tr>";
      }).join("") + '</tbody></table><p class="hint">' + t("rManual", { v: f(R.baseline.manual_up) }) + "</p></div>" +
      "<h2>" + t("rTransferTitle") + "</h2><p>" + t("rTransferText") + "</p>" +
      '<div class="card table-wrap"><table class="data"><thead><tr><th>' + t("thModel") + '</th><th class="num">' + t("thWithin") + '</th><th class="num">' + t("thCV") +
      '</th><th class="num">' + t("thCP") + "</th></tr></thead><tbody>" + R.transfer.map(function (x) {
        return "<tr><td>" + esc(modelName(x.model)) + '</td><td class="num">' + f(x.within) + '</td><td class="num">' + f(x.cross_version) + '</td><td class="num">' + f(x.cross_project) + "</td></tr>";
      }).join("") + '</tbody></table><p class="hint">F1, ' + t("rAvg") + "</p></div>" +
      '<p class="hint" style="margin-bottom:40px">' + t("rMore") + "</p></div>";
    box.innerHTML = html;
    var rows = R.leakage.rows;
    chart("leakChart", { type: "bar", data: { labels: rows.map(function (x) { return x.release; }), datasets: [
      { label: t("sLeaky"), data: rows.map(function (x) { return x.leaky_ga; }), backgroundColor: RED, borderRadius: 4 },
      { label: t("sHonestGA"), data: rows.map(function (x) { return x.honest_ga; }), backgroundColor: GREEN, borderRadius: 4 },
      { label: t("sHonestDef"), data: rows.map(function (x) { return x.honest_default; }), backgroundColor: BLUE_BAR, borderRadius: 4 }] },
      options: { scales: { y: { min: 0, max: 1, title: { display: true, text: "F1" } }, x: { grid: { display: false } } },
                 plugins: { legend: { labels: { boxWidth: 10, usePointStyle: true } } } } });
  }

  /* ---------------------------------------------------------------- wiring */
  function bind() {
    window.addEventListener("hashchange", route);
    document.addEventListener("click", function (e) {
      var b = e.target.closest("[data-lang]");
      if (b) I.set(b.getAttribute("data-lang"));
      var ds = e.target.closest("[data-ds]");
      if (ds) { S.selected = ds.getAttribute("data-ds"); renderDatasets(); updateStart(); }
      var m = e.target.closest("[data-model]");
      if (m && !m.disabled) { S.model = m.getAttribute("data-model"); renderModels(); }
      var src = e.target.closest("[data-src]");
      if (src) setSource(src.getAttribute("data-src"));
    });
    $("#file").addEventListener("change", function () { readFile(this.files[0]); });
    var drop = $("#drop");
    ["dragenter", "dragover"].forEach(function (n) { drop.addEventListener(n, function (e) { e.preventDefault(); drop.classList.add("hl"); }); });
    ["dragleave", "drop"].forEach(function (n) { drop.addEventListener(n, function (e) { e.preventDefault(); drop.classList.remove("hl"); }); });
    drop.addEventListener("drop", function (e) { readFile(e.dataTransfer.files[0]); });
    $("#fileRemove").addEventListener("click", clearFile);
    [["gens", "genOut", "gens"], ["pop", "popOut", "pop"], ["seed", "seedOut", "seed"]].forEach(function (x) {
      $("#" + x[0]).addEventListener("input", function () { S[x[2]] = +this.value; $("#" + x[1]).textContent = this.value; });
    });
    $("#startBtn").addEventListener("click", function () { start(false); });
    $("#againBtn").addEventListener("click", function () { start(true); });
    $("#newBtn").addEventListener("click", function () { stopRun(); $("#results").hidden = true; $("#setup").hidden = false; window.scrollTo(0, 0); });
    $("#cancelBtn").addEventListener("click", function () { stopRun(); $("#progress").hidden = true; $("#setup").hidden = false; });
    $("#csvBtn").addEventListener("click", downloadCSV);
    I.on(function () {
      renderDatasets(); renderModels(); updateStart();
      if (S.result && !$("#results").hidden) renderResults();
      if (S.research && !$("#view-research").hidden) renderResearch();
      if (S.upload) showFile($("#fileName").textContent, S.upload.summary);
    });
  }
  function init() {
    if (S.engine === "api") document.querySelector(".callout p").setAttribute("data-i18n-html", "homeNoteApi");
    I.apply();
    bind();
    renderModels();
    fetchJSON("data/manifest.json").then(function (j) { S.manifest = j.datasets; renderDatasets(); updateStart(); })
      .catch(function () { $("#dsGrid").innerHTML = '<p class="error">data/manifest.json missing</p>'; });
    route();
  }
  window.BP = { state: S, start: start };
  init();
})();
