// Checks docs/js/engine.js against the Python fixture (see web_parity.py).
//   node tools/web_parity.mjs parity.json
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const docs = resolve(here, "../../docs");
const E = await import(pathToFileURL(resolve(docs, "js/engine.js")));

const fixture = JSON.parse(readFileSync(process.argv[2] || "parity.json", "utf8"));
const fsFetch = async (url) => {
  const buf = readFileSync(fileURLToPath(url));
  return { json: async () => JSON.parse(buf.toString("utf8")),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
const clf = await E.loadClassifier(pathToFileURL(resolve(docs, "data") + "/"), fsFetch);

let failures = 0;
const fail = (msg) => { failures++; console.log("  FAIL " + msg); };
const maxDiff = (a, b) => a.reduce((m, x, i) => Math.max(m, Math.abs(x - b[i])), 0);

// 1. features + classifier on every recorded sample
let worstDesc = 0, worstScore = 0, agreeLoo = 0, correctLoo = 0, correctPyLoo = 0;
for (const s of fixture.samples) {
  const frames = s.frames;
  const tmpl = E.describe(frames, s.kind);
  worstDesc = Math.max(worstDesc, maxDiff(Array.from(tmpl.desc[0]), s.template0));
  const q = { static: E.describe(frames, "static"), dynamic: E.describe(frames, "dynamic", "query") };
  if (s.query_static) worstDesc = Math.max(worstDesc, maxDiff(Array.from(q.static.desc[0]), s.query_static));
  for (const [key, exclude] of [["scores", null], ["scores_loo", new Set([s.id])]]) {
    const js = clf.labelScores(q, true, exclude);
    const py = s[key];
    if (Object.keys(py).sort().join() !== [...js.keys()].sort().join()) fail(`${s.id} ${key}: label sets differ`);
    for (const [lbl, v] of Object.entries(py)) worstScore = Math.max(worstScore, Math.abs(v - js.get(lbl)));
  }
  const pyBest = Object.entries(s.scores_loo).sort((a, b) => a[1] - b[1])[0][0];
  const jsPred = E.scoresToPrediction(clf.labelScores(q, true, new Set([s.id])), 0.5);
  if (jsPred.label === pyBest) agreeLoo++;
  if (jsPred.label === s.label) correctLoo++;
  if (pyBest === s.label) correctPyLoo++;
}
console.log(`samples: ${fixture.samples.length}`);
console.log(`  max |descriptor JS - Python|   = ${worstDesc.toExponential(2)}`);
console.log(`  max |label score JS - Python|  = ${worstScore.toExponential(2)}`);
console.log(`  leave-one-out top label agrees: ${agreeLoo}/${fixture.samples.length}`);
console.log(`  leave-one-out accuracy: JS ${correctLoo}, Python ${correctPyLoo} of ${fixture.samples.length}`);
if (worstDesc > 1e-5) fail("descriptors differ");
if (worstScore > 1e-4) fail("scores differ");
if (agreeLoo !== fixture.samples.length) fail("leave-one-out labels differ");

// 2. segmenter + session on the simulated live stream (both modes)
for (const mode of ["words", "spelling"]) {
const rec = new E.Recognizer(clf, { spelling: mode === "spelling" });
const preds = [];
fixture.stream.forEach((f, i) => {
  const r = rec.process(f);
  if (r.prediction) preds.push({ i, ...r.prediction });
});
const py = fixture.stream_predictions[mode];
console.log(`stream (${mode}): ${fixture.stream.length} frames, Python ${py.length} predictions, JS ${preds.length}`);
for (let k = 0; k < Math.max(py.length, preds.length); k++) {
  const a = py[k], b = preds[k];
  const line = (p) => p ? `${String(p.i).padStart(4)} ${String(p.label).padEnd(7)} ${p.confidence.toFixed(3)} ${p.recognized ? "OK" : "--"} ${p.reason}${(p.replaces_previous || p.replacesPrevious) ? " (replaces)" : ""}` : "(none)";
  const same = a && b && a.i === b.i && a.label === b.label && a.recognized === b.recognized
    && a.reason === b.reason && Math.abs(a.confidence - b.confidence) < 2e-3
    && a.replaces_previous === b.replacesPrevious;
  console.log(`  py ${line(a)}   js ${line(b)}   ${same ? "same" : "DIFFERENT"}`);
  if (!same) fail(`stream (${mode}) prediction ${k} differs`);
}
}

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PARITY CHECKS PASSED");
process.exit(failures ? 1 : 0);
