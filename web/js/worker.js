/* One worker of the pool: holds the dataset and runs training jobs (fit on some rows, predict others).
   The genetic algorithm itself runs on the page and sends each (individual, fold) pair as a job,
   so several workers evaluate a generation in parallel. */
importScripts("ml.js");
var data = null;
self.onmessage = function (e) {
  var msg = e.data;
  if (msg.type === "init") { data = msg.data; return; }
  if (msg.type !== "job") return;
  try {
    var out = self.BPML.runJob(data, msg.job);
    self.postMessage({ type: "result", id: msg.id, pred: out.pred, scores: out.scores });
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: err && err.message ? err.message : String(err) });
  }
};
