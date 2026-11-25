// Siri Sign recognition engine: a line-by-line port of the Python app
// (python/app/landmarks.py, features.py, segmenter.py, classifier.py and the
// recognize part of session.py). No DOM here, so it also runs in Node for the
// parity tests (python/tools/web_parity.mjs).
//
// A Frame is { t, pose, left, right, face, aspect } where each landmark set is
// an array of [x, y, z(, visibility)] in MediaPipe's normalized image coords,
// or null when not detected. Coordinates come from the UNMIRRORED camera image,
// exactly like the frames the Python app received.

// ---------------------------------------------------------------- landmarks.py

const POSE_PAIRS = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14],
  [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
export const POSE_MIRROR = Array.from({ length: 33 }, (_, i) => i);
for (const [a, b] of POSE_PAIRS) { POSE_MIRROR[a] = b; POSE_MIRROR[b] = a; }

const FACE_PAIRS = [
  [70, 300], [63, 293], [105, 334], [66, 296], [107, 336],                 // eyebrows
  [33, 263], [160, 387], [158, 385], [133, 362], [153, 380], [144, 373],   // eyes
  [61, 291], [40, 270], [37, 267], [91, 321], [84, 314], [78, 308],        // mouth
];
const FACE_CENTER = [1, 0, 17, 13, 14, 152];
export const FACE_IDX = [...FACE_PAIRS.flat(), ...FACE_CENTER];
export const FACE_NOSE = FACE_IDX.indexOf(1);
export const FACE_MIRROR = FACE_IDX.map((_, i) => i);
{
  const pos = new Map(FACE_IDX.map((idx, k) => [idx, k]));
  for (const [a, b] of FACE_PAIRS) {
    FACE_MIRROR[pos.get(a)] = pos.get(b);
    FACE_MIRROR[pos.get(b)] = pos.get(a);
  }
}

/** MediaPipe HolisticLandmarkerResult (first person) -> Frame. */
export function frameFromResult(r, t, aspect) {
  const arr = (lms, vis) => (lms && lms.length && lms[0] && lms[0].length)
    ? lms[0].map((p) => vis ? [p.x, p.y, p.z, p.visibility ?? 1.0] : [p.x, p.y, p.z]) : null;
  let face = arr(r.faceLandmarks, false);
  if (face) face = FACE_IDX.map((i) => face[i]);
  return { t, pose: arr(r.poseLandmarks, true), left: arr(r.leftHandLandmarks, false),
    right: arr(r.rightHandLandmarks, false), face, aspect };
}

export function mirrorFrame(f) {
  const flip = (a) => a ? a.map((p) => { const q = p.slice(); q[0] = 1.0 - q[0]; return q; }) : null;
  let pose = flip(f.pose);
  if (pose) pose = POSE_MIRROR.map((j) => pose[j]);
  let face = flip(f.face);
  if (face) face = FACE_MIRROR.map((j) => face[j]);
  return { t: f.t, pose, left: flip(f.right), right: flip(f.left), face, aspect: f.aspect };
}

// ---------------------------------------------------------------- numpy helpers

/** np.interp (clamped ends). xp strictly increasing. */
function interp1(x, xp, fp) {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  if (x === xp[lo]) return fp[lo];
  const slope = (fp[hi] - fp[lo]) / (xp[hi] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

/** np.linspace(a, b, n) */
function linspace(a, b, n) {
  if (n === 1) return [a];
  const step = (b - a) / (n - 1);
  const y = Array.from({ length: n }, (_, i) => i * step + a);
  y[n - 1] = b;
  return y;
}

/** Python 3 round() (ties to even). */
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const norm2 = (dx, dy) => Math.sqrt(dx * dx + dy * dy);

/** 3rd largest value of an array (np.sort(a)[-3]). */
function thirdLargest(a) {
  const s = Float64Array.from(a).sort();
  return s[s.length - 3];
}

/** Interpolate each column of X (T x D) at the times in grid -> (n x D). */
function interpRows(grid, ts, X) {
  const D = X[0].length;
  const col = new Float64Array(ts.length);
  const out = grid.map(() => new Float64Array(D));
  for (let k = 0; k < D; k++) {
    for (let i = 0; i < ts.length; i++) col[i] = X[i][k];
    for (let g = 0; g < grid.length; g++) out[g][k] = interp1(grid[g], ts, col);
  }
  return out;
}

// ---------------------------------------------------------------- features.py

export const N_FRAMES = 32;
export const PRESENT_LINE = 1.3;
export const MIN_HAND_PRESENCE = 0.3;

const GROUPS = [
  ["left_shape", 60, 1.0], ["left_pos", 2, 0.8], ["left_present", 1, 0.5],
  ["right_shape", 60, 1.0], ["right_pos", 2, 0.8], ["right_present", 1, 0.5],
  ["face", 2 * FACE_IDX.length, 2.0], ["elbows", 4, 0.3],
];
export const FRAME_DIM = GROUPS.reduce((s, g) => s + g[1], 0);
export const WEIGHTS = new Float64Array(FRAME_DIM);
{
  let o = 0;
  for (const [, size, w] of GROUPS) for (let i = 0; i < size; i++) WEIGHTS[o++] = w / Math.sqrt(size);
}

/** Normalized image coords -> isotropic units (1.0 = image height). */
function toUnits(points, aspect) {
  return points.map((p) => [p[0] * aspect, p[1], p[2] * aspect]);
}

export function bodyRef(frame) {
  if (!frame.pose) return null;
  const a = frame.aspect;
  const ls = [frame.pose[11][0] * a, frame.pose[11][1]];
  const rs = [frame.pose[12][0] * a, frame.pose[12][1]];
  const scale = norm2(ls[0] - rs[0], ls[1] - rs[1]);
  if (scale < 1e-3) return null;
  return { center: [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2], scale };
}

/** (21, 2) hand points relative to the shoulders, in shoulder widths. */
export function handInBody(hand, aspect, ref) {
  return hand.map((p) => [(p[0] * aspect - ref.center[0]) / ref.scale, (p[1] - ref.center[1]) / ref.scale]);
}

export function wristHeight(hand, ref) {
  return (hand[0][1] - ref.center[1]) / ref.scale;
}

function fillRefs(frames) {
  const refs = frames.map(bodyRef);
  const idx = [];
  refs.forEach((r, i) => { if (r) idx.push(i); });
  if (!idx.length) return null;
  return refs.map((r, i) => {
    if (r) return r;
    let best = idx[0];
    for (const k of idx) if (Math.abs(k - i) < Math.abs(best - i)) best = k;
    return refs[best];
  });
}

function interpTrack(ts, arrays) {
  const mask = arrays.map((a) => a !== null);
  const nKnown = mask.filter(Boolean).length;
  if (nKnown === 0 || nKnown === arrays.length) return arrays;
  const first = arrays.find((a) => a !== null);
  const rows = first.length, cols = first[0].length;
  const kt = [], known = [];
  arrays.forEach((a, i) => { if (a) { kt.push(ts[i]); known.push(a.flat()); } });
  const fp = new Float64Array(known.length);
  const out = arrays.map(() => Array.from({ length: rows }, () => new Array(cols)));
  for (let k = 0; k < rows * cols; k++) {
    for (let j = 0; j < known.length; j++) fp[j] = known[j][k];
    const r = Math.floor(k / cols), c = k % cols;
    for (let i = 0; i < arrays.length; i++) out[i][r][c] = interp1(ts[i], kt, fp);
  }
  return out;
}

function timesOf(frames) {
  let ts = frames.map((f) => f.t);
  for (let i = 1; i < ts.length; i++) {
    if (ts[i] - ts[i - 1] <= 0) { ts = ts.map((_, j) => j); break; }
  }
  return ts;
}

export function cleanSequence(frames) {
  if (!frames.length) return null;
  const refs = fillRefs(frames);
  if (!refs) return null;
  const ts = timesOf(frames);
  const hands = {};
  for (const side of ["left", "right"]) {
    let track = frames.map((f, i) => {
      const h = f[side];
      return (h && wristHeight(h, refs[i]) > PRESENT_LINE) ? null : h;
    });
    const present = track.filter((h) => h !== null).length / track.length;
    if (present < MIN_HAND_PRESENCE) track = track.map(() => null);
    else track = interpTrack(ts, track);
    hands[side] = track;
  }
  const faces = interpTrack(ts, frames.map((f) => f.face));
  const poses = interpTrack(ts, frames.map((f) => f.pose));
  const clean = frames.map((f, i) => ({ t: f.t, pose: poses[i], left: hands.left[i],
    right: hands.right[i], face: faces[i], aspect: f.aspect }));
  return { clean, refs };
}

export function frameVector(f, ref) {
  const v = new Float64Array(FRAME_DIM);
  let o = 0;
  for (const hand of [f.left, f.right]) {
    if (!hand) { o += 63; continue; }
    const h = toUnits(hand, f.aspect);
    const w = h[0];
    const size = Math.max(Math.hypot(h[9][0] - w[0], h[9][1] - w[1], h[9][2] - w[2]), 1e-6);
    for (let i = 1; i < 21; i++) for (let c = 0; c < 3; c++) v[o++] = (h[i][c] - w[c]) / size;
    v[o++] = (w[0] - ref.center[0]) / ref.scale;
    v[o++] = (w[1] - ref.center[1]) / ref.scale;
    v[o++] = 1;
  }
  if (f.face) {
    const face = toUnits(f.face, f.aspect);
    const nose = face[FACE_NOSE];
    for (const p of face) { v[o++] = (p[0] - nose[0]) / ref.scale; v[o++] = (p[1] - nose[1]) / ref.scale; }
  } else o += 2 * FACE_IDX.length;
  if (f.pose) {
    const p = toUnits(f.pose, f.aspect);
    for (const j of [13, 14]) {
      v[o++] = (p[j][0] - ref.center[0]) / ref.scale;
      v[o++] = (p[j][1] - ref.center[1]) / ref.scale;
    }
  }
  for (let i = 0; i < FRAME_DIM; i++) v[i] *= WEIGHTS[i];
  return v;
}

function prepare(frames) {
  const cleaned = cleanSequence(frames);
  if (!cleaned) return null;
  const { clean, refs } = cleaned;
  const X = clean.map((f, i) => frameVector(f, refs[i]));
  const ts = timesOf(clean);
  const P = clean.map((f, i) => {
    const hs = [f.left, f.right].filter(Boolean).map((h) => handInBody(h, f.aspect, refs[i]));
    return hs.length ? hs.flat() : null;
  });
  return { ts, X, P };
}

function meanRows(X) {
  const m = new Float64Array(X[0].length);
  for (const r of X) for (let k = 0; k < m.length; k++) m[k] += r[k];
  for (let k = 0; k < m.length; k++) m[k] /= X.length;
  return m;
}

export function resample(ts, X, n = N_FRAMES) {
  if (X.length === 1 || ts[ts.length - 1] - ts[0] <= 0) {
    const m = meanRows(X);
    return Array.from({ length: n }, () => m);
  }
  return interpRows(linspace(ts[0], ts[ts.length - 1], n), ts, X);
}

const STILL_TRIM = 0.08, KEEP_STILL = 2;

function stillRun(P, order) {
  const anchor = P[order[0]];
  let n = 1;
  for (const i of order.slice(1)) {
    if (!anchor || !P[i] || P[i].length !== anchor.length) break;
    const moved = thirdLargest(P[i].map((p, k) => norm2(p[0] - anchor[k][0], p[1] - anchor[k][1])));
    if (moved > STILL_TRIM) break;
    n++;
  }
  return n;
}

function movementPart(ts, X, P) {
  const T = ts.length;
  if (T <= 2 * KEEP_STILL + 1) return { ts, X };
  const fwd = [...Array(T).keys()];
  const lead = stillRun(P, fwd);
  const trail = stillRun(P, fwd.slice().reverse());
  let a = Math.max(0, lead - KEEP_STILL);
  let b = Math.min(T, T - trail + KEEP_STILL);
  if (b - a < 2) { a = Math.max(0, T - 2 * KEEP_STILL); b = T; }
  return { ts: ts.slice(a, b), X: X.slice(a, b) };
}

const QUERY_STARTS = [0.0, 0.15, 0.3, 0.45], QUERY_ENDS = [0.0, 0.2, 0.35];
const QUERY_MIN_KEEP = 0.45, QUERY_CUT_COST = 0.5;
const TEMPLATE_STARTS = [0.0, 0.25], TEMPLATE_CUT_COST = 1.2;

function crops(ts, X, starts, ends, minKeep, cost) {
  const out = [], pens = [];
  const dur = ts[ts.length - 1] - ts[0];
  const sq = Math.sqrt(N_FRAMES);
  for (const a of starts) {
    for (const b of ends) {
      if (1 - a - b < minKeep - 1e-9) continue;
      let segT = ts, segX = X;
      if (!(dur <= 0)) {
        const t0 = ts[0] + a * dur, t1 = ts[ts.length - 1] - b * dur;
        const grid = linspace(t0, t1, Math.max(2, pyRound(ts.length * (1 - a - b)) + 1));
        segT = grid;
        segX = interpRows(grid, ts, X);
      }
      const R = resample(segT, segX);
      const flat = new Float64Array(N_FRAMES * FRAME_DIM);
      R.forEach((row, i) => { for (let k = 0; k < FRAME_DIM; k++) flat[i * FRAME_DIM + k] = row[k] / sq; });
      out.push(flat);
      pens.push(cost * (a + b));
    }
  }
  return { variants: out, penalties: pens };
}

function staticDescriptor(frames) {
  const m = prepare(frames);
  return m ? meanRows(m.X) : null;
}

function dynamicDescriptor(frames, role) {
  const m = prepare(frames);
  if (!m) return null;
  const { ts, X } = movementPart(m.ts, m.X, m.P);
  if (role === "query") return crops(ts, X, QUERY_STARTS, QUERY_ENDS, QUERY_MIN_KEEP, QUERY_CUT_COST);
  return crops(ts, X, TEMPLATE_STARTS, [0.0], 0.0, TEMPLATE_CUT_COST);
}

/** -> { desc: [variants], descMirror: [variants], penalty: [..] } or null */
export function describe(frames, kind, role = "template") {
  const mirrored = frames.map(mirrorFrame);
  if (kind === "static") {
    const d = staticDescriptor(frames);
    if (!d) return null;
    return { desc: [d], descMirror: [staticDescriptor(mirrored)], penalty: [0] };
  }
  const d = dynamicDescriptor(frames, role);
  if (!d) return null;
  const dm = dynamicDescriptor(mirrored, role);
  return { desc: d.variants, descMirror: dm.variants, penalty: d.penalties };
}

// ---------------------------------------------------------------- segmenter.py

export const REST_LINE = 1.2, HOLD_TIME = 0.4, MAX_DURATION = 3.0, MIN_DURATION = 0.25;
export const STILL_DIST = 0.08, MOVE_DIST = 0.15, DOWN_TIME = 0.12, PREROLL = 0.15;

const robustMax = (dists) => thirdLargest(dists);
const sides = (hands) => Object.keys(hands);

function wobble(entries) {
  let worst = 0;
  for (const side of sides(entries[0].hands)) {
    const n = entries.length;
    const mean = entries[0].hands[side].map(() => [0, 0]);
    for (const e of entries) e.hands[side].forEach((p, k) => { mean[k][0] += p[0]; mean[k][1] += p[1]; });
    mean.forEach((m) => { m[0] /= n; m[1] /= n; });
    for (const e of entries) {
      const dev = robustMax(e.hands[side].map((p, k) => norm2(p[0] - mean[k][0], p[1] - mean[k][1])));
      if (dev > worst) worst = dev;
    }
  }
  return worst;
}

function sameHands(entries) {
  const keys = sides(entries[0].hands).sort().join();
  return keys.length > 0 && entries.every((e) => sides(e.hands).sort().join() === keys);
}

export function findStillWindow(entries, length = 0.3) {
  let best = null, bestW = null, j = 0;
  for (let i = 0; i < entries.length; i++) {
    while (j < entries.length && entries[j].frame.t - entries[i].frame.t < length) j++;
    if (j >= entries.length) break;
    const win = entries.slice(i, j + 1);
    if (win.length < 3 || !sameHands(win)) continue;
    const w = wobble(win);
    if (bestW === null || w < bestW) { best = [i, j + 1]; bestW = w; }
  }
  if (best === null || bestW > STILL_DIST * 1.5) return null;
  return best;
}

function makeSegment(frames, reason, still) {
  return {
    frames, reason,
    stillFrom: still ? still[0] : null, stillTo: still ? still[1] : null,
    get duration() { return frames.length ? frames[frames.length - 1].t - frames[0].t : 0; },
    get stillFrames() { return this.stillFrom === null ? null : frames.slice(this.stillFrom, this.stillTo); },
  };
}

export class Segmenter {
  constructor(restLine = REST_LINE) { this.restLine = restLine; this.reset(); }

  reset() {
    this.state = "idle";
    this.recent = [];
    this.seg = [];
    this.holdRef = null;
    this.downSince = null;
    this.movedCount = 0;
    this.lastRef = null;
  }

  entry(frame) {
    let ref = bodyRef(frame);
    if (ref) this.lastRef = ref; else ref = this.lastRef;
    const hands = {};
    if (ref) {
      for (const side of ["left", "right"]) {
        const h = frame[side];
        if (h && wristHeight(h, ref) < this.restLine) hands[side] = handInBody(h, frame.aspect, ref);
      }
    }
    return { frame, hands };
  }

  stillFor(entries, holdTime) {
    if (!entries.length) return null;
    const now = entries[entries.length - 1].frame.t;
    const win = entries.filter((e) => e.frame.t >= now - holdTime);
    if (win.length < 3 || now - win[0].frame.t < holdTime * 0.8) return null;
    if (!sameHands(win) || wobble(win) > STILL_DIST) return null;
    return win;
  }

  movedFromHold(e) {
    const ref = this.holdRef || {};
    if (sides(e.hands).sort().join() !== sides(ref).sort().join()) return true;
    return sides(ref).some((s) =>
      robustMax(e.hands[s].map((p, k) => norm2(p[0] - ref[s][k][0], p[1] - ref[s][k][1]))) > MOVE_DIST);
  }

  setHeld(window) {
    this.state = "held";
    const last = window[window.length - 1];
    this.holdRef = {};
    for (const s of sides(last.hands)) {
      const m = last.hands[s].map(() => [0, 0]);
      for (const e of window) e.hands[s].forEach((p, k) => { m[k][0] += p[0]; m[k][1] += p[1]; });
      m.forEach((q) => { q[0] /= window.length; q[1] /= window.length; });
      this.holdRef[s] = m;
    }
    this.movedCount = 0;
  }

  emit(entries, reason, still) {
    if (still === null) still = findStillWindow(entries);
    return makeSegment(entries.map((e) => e.frame), reason, still);
  }

  get publicState() { return this.state === "signing" ? "signing" : "idle"; }

  /** -> Segment or null */
  update(frame) {
    const e = this.entry(frame);
    const t = frame.t;
    this.recent.push(e);
    while (this.recent.length && this.recent[0].frame.t < t - 1.0) this.recent.shift();

    const up = sides(e.hands).length > 0;
    if (up) this.downSince = null;
    else if (this.downSince === null) this.downSince = t;
    const down = this.downSince !== null && t - this.downSince >= DOWN_TIME - 1e-9;

    if (this.state === "idle") {
      if (up) { this.state = "signing"; this.seg = [e]; }
      return null;
    }

    if (this.state === "held" || this.state === "wait") {
      if (down) { this.state = "idle"; return null; }
      if (this.state === "held") {
        if (up && this.movedFromHold(e)) this.movedCount++;
        else this.movedCount = 0;
        if (this.movedCount >= 2) {
          this.state = "signing";
          this.seg = this.recent.filter((x) => x.frame.t >= t - PREROLL && sides(x.hands).length);
        }
        return null;
      }
      const win = this.stillFor(this.recent.slice(), HOLD_TIME);
      if (win) this.setHeld(win);
      return null;
    }

    // --- signing ---
    this.seg.push(e);
    const start = this.seg[0].frame.t;

    if (down) {
      const upEntries = this.seg.filter((x) => sides(x.hands).length);
      this.state = "idle";
      this.seg = [];
      if (!upEntries.length || upEntries[upEntries.length - 1].frame.t - upEntries[0].frame.t < MIN_DURATION) return null;
      return this.emit(upEntries, "down", null);
    }

    if (up) {
      const win = this.stillFor(this.seg, HOLD_TIME);
      if (win) {
        const seg = this.seg;
        this.seg = [];
        this.setHeld(win);
        return this.emit(seg, "hold", [seg.length - win.length, seg.length]);
      }
    }

    if (t - start >= MAX_DURATION) {
      const seg = this.seg.filter((x) => sides(x.hands).length);
      this.seg = [];
      this.state = "wait";
      return seg.length ? this.emit(seg, "timeout", null) : null;
    }
    return null;
  }
}

// ---------------------------------------------------------------- classifier.py

export const K = 3, UNKNOWN_SCORE = 2.6, N_RETRIEVE = 150;

class Group {
  constructor(kind, meta, floats) {
    this.kind = kind;
    this.dim = meta.dim;
    this.spreads = meta.spreads;
    this.examples = meta.examples;
    this.rows = []; this.rowEx = []; this.rowPen = []; this.rowSpread = [];
    meta.examples.forEach((ex, i) => {
      for (let v = 0; v < ex.variants; v++) {
        this.rows.push(floats.subarray(ex.offset + v * this.dim, ex.offset + (v + 1) * this.dim));
        this.rowEx.push(i);
        this.rowPen.push(ex.penalty[v]);
        this.rowSpread.push(meta.spreads[ex.label]);
      }
    });
  }

  /** example index -> best score (Map keeps Python's dict insertion order). */
  scores(q, mirror) {
    const V = mirror ? [...q.desc, ...q.descMirror] : q.desc;
    const Vpen = mirror ? [...q.penalty, ...q.penalty] : q.penalty;
    const n = Math.min(N_RETRIEVE, this.rows.length);
    const out = new Map();
    for (let v = 0; v < V.length; v++) {
      const vec = V[v];
      const dist = this.rows.map((row, r) => {
        let s = 0;
        for (let k = 0; k < this.dim; k++) { const d = vec[k] - row[k]; s += d * d; }
        return [Math.sqrt(s), r];
      });
      dist.sort((a, b) => a[0] - b[0]);
      for (const [dd, r] of dist.slice(0, n)) {
        const ex = this.rowEx[r];
        const s = dd / this.rowSpread[r] + this.rowPen[r] + Vpen[v];
        if (s < (out.has(ex) ? out.get(ex) : Infinity)) out.set(ex, s);
      }
    }
    return out;
  }
}

export class Classifier {
  /** meta = templates.json, floats = Float32Array of templates.bin */
  constructor(meta, floats) {
    this.kinds = meta.kinds;
    this.groups = {};
    for (const [kind, g] of Object.entries(meta.groups)) {
      if (g.dim !== (kind === "static" ? FRAME_DIM : FRAME_DIM * N_FRAMES)) {
        throw new Error(`templates.json: ${kind} dim ${g.dim} does not match this engine`);
      }
      this.groups[kind] = new Group(kind, g, floats);
    }
  }

  get labels() { return Object.keys(this.kinds); }
  kindOf(label) { return this.kinds[label] || null; }

  /** allowed: Set of labels to consider (Python's spelling mode), or null for all. */
  labelScores(query, mirror = true, excludeIds = null, allowed = null) {
    const perLabel = new Map();
    for (const [kind, g] of Object.entries(this.groups)) {
      const q = query[kind];
      if (!q) continue;
      for (const [i, score] of g.scores(q, mirror)) {
        const ex = g.examples[i];
        if (excludeIds && excludeIds.has(ex.id)) continue;
        if (allowed && !allowed.has(ex.label)) continue;
        if (!perLabel.has(ex.label)) perLabel.set(ex.label, []);
        perLabel.get(ex.label).push(score);
      }
    }
    const out = new Map();
    for (const [lbl, v] of perLabel) {
      const top = v.slice().sort((a, b) => a - b).slice(0, K);
      out.set(lbl, top.reduce((s, x) => s + x, 0) / top.length);
    }
    return out;
  }

  predict(query, threshold = 0.5, mirror = true, excludeIds = null, allowed = null) {
    return scoresToPrediction(this.labelScores(query, mirror, excludeIds, allowed), threshold);
  }
}

export function scoresToPrediction(labelScores, threshold) {
  if (!labelScores.size) return { label: null, confidence: 0, alternatives: [], recognized: false, score: null };
  const labels = [...labelScores.keys()];
  const s = labels.map((l) => labelScores.get(l));
  const w = s.map((x) => Math.exp(-0.5 * x * x));
  const total = w.reduce((a, b) => a + b, 0) + Math.exp(-0.5 * UNKNOWN_SCORE ** 2);
  const p = w.map((x) => x / total);
  const order = labels.map((_, i) => i).sort((a, b) => p[b] - p[a]);
  const top = order[0];
  return {
    label: labels[top], confidence: p[top],
    alternatives: order.slice(1, 3).map((i) => [labels[i], p[i]]),
    recognized: p[top] >= threshold, score: s[top],
  };
}

export function queryFromSegment(seg) {
  const still = seg.stillFrames;
  return {
    static: (still && still.length) ? describe(still, "static") : null,
    dynamic: describe(seg.frames, "dynamic", "query"),
  };
}

/** Load docs/data/templates.json + templates.bin. fetchFn for Node tests. */
export async function loadClassifier(baseUrl, fetchFn = fetch) {
  const meta = await (await fetchFn(new URL("templates.json", baseUrl))).json();
  const buf = await (await fetchFn(new URL("templates.bin", baseUrl))).arrayBuffer();
  const floats = new Float32Array(buf);
  if (floats.length !== meta.floats) throw new Error("templates.bin is incomplete");
  return new Classifier(meta, floats);
}

// ---------------------------------------------------------------- session.py (recognize only)

export const REPLACE_WINDOW = 1.0;

export class Recognizer {
  constructor(classifier, { threshold = 0.5, mirror = true, spelling = false } = {}) {
    this.clf = classifier;
    this.threshold = threshold;
    this.mirror = mirror;
    this.spelling = spelling;   // Python's "spelling" setting: only single letters
    this.segmenter = new Segmenter();
    this.prev = null;
  }

  reset() { this.segmenter.reset(); this.prev = null; }

  allowed() {
    if (!this.spelling) return null;
    const letters = this.clf.labels.filter((l) => [...l].length === 1);
    return letters.length ? new Set(letters) : null;
  }

  /** Feed one frame. -> { state, up: {left,right}, prediction|null } */
  process(frame) {
    let prediction = null;
    const seg = this.segmenter.update(frame);
    if (seg) prediction = this.recognize(seg);
    if (this.segmenter.state === "idle") this.prev = null;
    return { state: this.segmenter.publicState, prediction };
  }

  recognize(seg) {
    const q = queryFromSegment(seg);
    const pred = this.clf.predict(q, this.threshold, this.mirror, null, this.allowed());
    const msg = { ...pred, reason: seg.reason, duration: seg.duration, replacesPrevious: false };
    const kind = msg.label ? this.clf.kindOf(msg.label) : null;
    const prev = this.prev;
    if (msg.recognized && kind === "dynamic" && prev
        && seg.frames[0].t - prev.t <= REPLACE_WINDOW
        && (prev.kind === "static" || prev.label === msg.label)) {
      msg.replacesPrevious = true;
    }
    this.prev = msg.recognized ? { label: msg.label, kind, t: seg.frames[seg.frames.length - 1].t } : null;
    return msg;
  }
}
