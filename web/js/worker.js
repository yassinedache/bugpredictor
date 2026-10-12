/* Runs the browser engine off the main thread so the page stays responsive during the GA. */
importScripts("ml.js");
self.onmessage = function (e) {
  var msg = e.data;
  try {
    var ds = self.BPML.parse(msg.text, msg.name);
    var result = self.BPML.analyze(ds, msg.opts, function (ev) { self.postMessage({ type: "progress", ev: ev }); });
    self.postMessage({ type: "done", result: result });
  } catch (err) {
    self.postMessage({ type: "error", message: err && err.message ? err.message : String(err) });
  }
};
