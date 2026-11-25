// Siri Sign: camera + MediaPipe Holistic (in the browser) + the recognition
// engine ported from the Python app (engine.js).
import {
  Recognizer, loadClassifier, frameFromResult, bodyRef, wristHeight, REST_LINE,
} from "./engine.js";
import { Orb } from "./orb.js";

const MP_VERSION = "1.2.2";   // same MediaPipe version as the Python app
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL = new URL("../models/holistic_landmarker.task", import.meta.url);
const DATA_URL = new URL("../data/", import.meta.url);
const MAX_FPS = 20;          // the Python page sent at most 20 frames per second
const CAPTURE_W = 640;       // ...scaled to 640 px wide
const WORD_BREAK_MS = 1500;  // hands down this long after a letter ends the word
const NAME = "JULIA";

const DISPLAY = { ARROZ: "Arroz", "FEIJÃO": "Feijão", NOME: "Nome", SINAL: "Sinal", CASA: "Casa" };
const display = (label) => DISPLAY[label] ||
  (label.length === 1 ? label : label.charAt(0) + label.slice(1).toLowerCase());
const isLetter = (label) => [...label].length === 1;

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
const capture = document.createElement("canvas");
const cctx = capture.getContext("2d", { willReadFrequently: false });
const orb = new Orb($("orb"), $("orbGlow"), document.querySelector(".orb-wrap"));
const debug = new URLSearchParams(location.search).has("debug");
if (debug) $("debug").hidden = false;

let landmarker = null, recognizer = null, stream = null;
let running = false, lastProc = 0, lastTs = 0, fps = 0, inferMs = 0;
let mode = "words";
let spell = "", lastLetterAt = 0, lastHandsUpAt = 0;
let history = [];       // [{ text, letter }]
let dimTimer = null;
let lastFrame = null;

// ------------------------------------------------------------------ loading

let modelProgress = 0;
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} em ${url}`);
  const total = Number(res.headers.get("content-length")) || 13.7e6;   // gzip may hide the size
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(1, got / total));
  }
  onProgress(1);
  return new Uint8Array(await new Blob(chunks).arrayBuffer());
}

async function createLandmarker() {
  const [{ FilesetResolver, HolisticLandmarker }, model] = await Promise.all([
    import(`${MP_BASE}/vision_bundle.mjs`),
    fetchWithProgress(MODEL_URL, (p) => { modelProgress = p; renderLoading(); }),
  ]);
  const files = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  const opts = (delegate) => ({
    baseOptions: { modelAssetBuffer: model, delegate },
    runningMode: "VIDEO",
    // same thresholds as python/app/landmarks.py
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5,
  });
  try {
    return { lm: await HolisticLandmarker.createFromOptions(files, opts("GPU")), delegate: "GPU" };
  } catch (e) {
    console.warn("GPU indisponível, usando CPU", e);
    return { lm: await HolisticLandmarker.createFromOptions(files, opts("CPU")), delegate: "CPU" };
  }
}

// Start downloading right away so it's (mostly) ready when the camera is on.
const ready = Promise.all([createLandmarker(), loadClassifier(DATA_URL)]);
ready.catch(() => {});

// ------------------------------------------------------------------ camera gate UI

let gateState = "intro";
function setGate(state, opts = {}) {
  gateState = state;
  const gate = $("gate"), inner = $("gateInner");
  gate.classList.toggle("show", state !== "hidden");
  if (state === "requesting") {
    inner.innerHTML = `<div class="spinner"></div><p class="gate-title">Aguardando permissão</p>
      <p class="gate-text">Toque em <b>Permitir</b> quando o navegador pedir acesso à câmera.</p>`;
  } else if (state === "loading") {
    inner.innerHTML = `<div class="spinner"></div><p class="gate-title">Preparando o reconhecimento</p>
      <p class="gate-text" id="loadText">Baixando o modelo de visão (cerca de 28 MB, só na primeira vez)…</p>
      <div class="progress"><div id="loadBar"></div></div>`;
    renderLoading();
  } else if (state === "error") {
    inner.innerHTML = `<div class="gate-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5M12 16.5v.01"/></svg></div>
      <p class="gate-title">${opts.title}</p><p class="gate-text">${opts.text}</p>
      ${opts.retry === false ? "" : '<button class="btn" type="button" id="btnRetry">Tentar de novo</button>'}`;
    const b = $("btnRetry");
    if (b) b.onclick = () => (opts.onRetry || start)();
  }
}
function renderLoading() {
  const bar = $("loadBar");
  if (bar) bar.style.width = Math.round(Math.max(0.04, modelProgress) * 100) + "%";
  const txt = $("loadText");
  if (txt && modelProgress >= 1) txt.textContent = "Iniciando o modelo…";
}

function cameraError(e) {
  const name = e && e.name;
  const ios = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      title: "Acesso à câmera bloqueado",
      text: ios
        ? "Para liberar no iPhone: toque em <b>aA</b> na barra de endereço › <b>Ajustes do Site</b> › <b>Câmera</b> › <b>Permitir</b>. Depois toque em Tentar de novo."
        : "Clique no ícone de câmera ou cadeado na barra de endereço, permita a câmera para este site e toque em Tentar de novo.",
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError" || name === "DevicesNotFoundError") {
    return { title: "Nenhuma câmera encontrada", text: "Este aparelho não tem uma câmera disponível. Tente em um celular ou notebook com câmera." };
  }
  if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") {
    return { title: "A câmera está ocupada", text: "Outro app ou aba está usando a câmera. Feche-o e toque em Tentar de novo." };
  }
  return { title: "Não foi possível abrir a câmera", text: "Tente recarregar a página ou abrir o link no Safari ou no Chrome." };
}

// ------------------------------------------------------------------ start

async function start() {
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setGate("error", {
      title: "Câmera indisponível neste navegador",
      text: window.isSecureContext
        ? "Abra o link diretamente no <b>Safari</b> ou no <b>Chrome</b> (navegadores dentro de apps, como Instagram ou WhatsApp, costumam bloquear a câmera)."
        : "A câmera só funciona em páginas seguras (https). Abra o endereço com https://.",
      retry: false,
    });
    return;
  }
  setGate("requesting");
  setStatus("Aguardando a câmera…");
  try {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e) {
    console.error(e);
    setGate("error", cameraError(e));
    setStatus("A câmera não está ativa");
    return;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  if (!video.videoWidth) await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
  layout();

  setGate("loading");
  setStatus("Preparando o reconhecimento…");
  let clf, delegate;
  try {
    const [lmInfo, c] = await ready;
    landmarker = lmInfo.lm; delegate = lmInfo.delegate; clf = c;
  } catch (e) {
    console.error(e);
    setGate("error", {
      title: "Não consegui carregar o reconhecimento",
      text: "Verifique a conexão com a internet e recarregue a página.",
      onRetry: () => location.reload(),
    });
    setStatus("Erro ao carregar");
    return;
  }
  recognizer = new Recognizer(clf, { spelling: mode === "spelling" });
  window.__siriSign = { delegate };   // for debugging / automated tests
  if (debug) window.__siriSign.say = onPrediction;
  setGate("hidden");
  $("camPill").hidden = false;
  running = true;
  setStatus("Levante as mãos para sinalizar");
  showResponse("Pode sinalizar", "Estou olhando. Faça um sinal da lista abaixo.", { quiet: true });
}

$("btnStart").onclick = start;

// ------------------------------------------------------------------ layout & overlay

function layout() {
  const wrap = $("stageWrap"), stage = $("stage");
  const ar = (video.videoWidth && video.videoHeight) ? video.videoWidth / video.videoHeight : 4 / 3;
  const W = wrap.clientWidth;
  const maxH = Math.min(window.innerHeight * (window.innerWidth < 820 ? 0.58 : 0.68), 720);
  let w = W, h = W / ar;
  if (h > maxH) { h = maxH; w = Math.min(W, h * ar); }
  stage.style.aspectRatio = "auto";
  stage.style.width = Math.floor(w) + "px";
  stage.style.height = Math.floor(h) + "px";
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  overlay.width = Math.floor(w * dpr);
  overlay.height = Math.floor(h * dpr);
}
window.addEventListener("resize", () => { if (video.videoWidth) layout(); });

const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

function drawOverlay(frame, ref, upSides) {
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);
  if (!frame) return;
  // video is shown with object-fit: cover and mirrored
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(W / vw, H / vh), dw = vw * s, dh = vh * s;
  const ox = (W - dw) / 2, oy = (H - dh) / 2;
  const X = (x) => ox + (1 - x) * dw, Y = (y) => oy + y * dh;
  const unit = W / 400;

  if (ref && !upSides.left && !upSides.right) {
    const y = Y(ref.center[1] + REST_LINE * ref.scale);
    if (y > 0 && y < H) {
      octx.save();
      octx.setLineDash([6 * unit, 6 * unit]);
      octx.strokeStyle = "rgba(255,255,255,0.55)";
      octx.lineWidth = 1.5 * unit;
      octx.beginPath(); octx.moveTo(0, y); octx.lineTo(W, y); octx.stroke();
      octx.setLineDash([]);
      octx.font = `500 ${Math.round(11 * unit)}px -apple-system, BlinkMacSystemFont, Inter, sans-serif`;
      octx.fillStyle = "rgba(255,255,255,0.85)";
      octx.textAlign = "center";
      octx.fillText("levante as mãos acima da linha", W / 2, y - 8 * unit);
      octx.restore();
    }
  }
  for (const side of ["left", "right"]) {
    const h = frame[side];
    if (!h) continue;
    const up = upSides[side];
    octx.strokeStyle = up ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.35)";
    octx.lineWidth = 2 * unit;
    octx.lineCap = "round";
    octx.beginPath();
    for (const [a, b] of HAND_EDGES) { octx.moveTo(X(h[a][0]), Y(h[a][1])); octx.lineTo(X(h[b][0]), Y(h[b][1])); }
    octx.stroke();
    octx.fillStyle = up ? "#bf5af2" : "rgba(255,255,255,0.5)";
    for (const p of h) { octx.beginPath(); octx.arc(X(p[0]), Y(p[1]), 2.6 * unit, 0, Math.PI * 2); octx.fill(); }
  }
}

// ------------------------------------------------------------------ main loop

function tick(now) {
  requestAnimationFrame(tick);
  orb.draw(now);
  if (!running || video.readyState < 2 || !video.videoWidth) return;
  if (now - lastProc < 1000 / MAX_FPS - 2) return;
  if (lastProc) { const inst = 1000 / (now - lastProc); fps = fps ? 0.85 * fps + 0.15 * inst : inst; }
  lastProc = now;

  const scale = Math.min(1, CAPTURE_W / video.videoWidth);
  const w = Math.round(video.videoWidth * scale), h = Math.round(video.videoHeight * scale);
  if (capture.width !== w || capture.height !== h) { capture.width = w; capture.height = h; }
  cctx.drawImage(video, 0, 0, w, h);

  const ts = Math.max(lastTs + 1, Math.round(now));
  lastTs = ts;
  let result;
  const t0 = performance.now();
  try {
    result = landmarker.detectForVideo(capture, ts);
  } catch (e) {
    console.error(e);
    return;
  }
  inferMs = performance.now() - t0;
  const frame = frameFromResult(result, now / 1000, w / h);
  lastFrame = frame;
  const out = recognizer.process(frame);

  // live status (same "hand up" rule as the Python segmenter)
  const ref = bodyRef(frame);
  const upSides = { left: false, right: false };
  if (ref) for (const s of ["left", "right"]) upSides[s] = !!(frame[s] && wristHeight(frame[s], ref) < REST_LINE);
  const anyUp = upSides.left || upSides.right;
  if (anyUp) lastHandsUpAt = now;
  drawOverlay(frame, ref, upSides);
  updateLive(frame, ref, anyUp, out.state);
  if (out.prediction) onPrediction(out.prediction);

  if (debug) {
    window.__siriSign.frame = frame;
    window.__siriSign.fps = fps;
    window.__siriSign.ms = inferMs;
    $("debug").textContent = `${fps.toFixed(1)} fps · ${inferMs.toFixed(0)} ms · ${window.__siriSign?.delegate}\n` +
      `${out.state} · pose ${frame.pose ? "✓" : "–"} face ${frame.face ? "✓" : "–"} L ${frame.left ? "✓" : "–"} R ${frame.right ? "✓" : "–"}`;
  }
}
requestAnimationFrame(tick);

let liveText = "";
function updateLive(frame, ref, anyUp, state) {
  let text, live = false, energy = 0.18;
  if (!ref) {
    text = frame.face ? "Afaste-se um pouco: preciso ver seus ombros" : "Não estou vendo você";
    energy = 0.12;
  } else if (state === "signing") {
    text = "Estou vendo…"; live = true; energy = 0.7;
  } else if (anyUp) {
    text = "Pode sinalizar"; live = true; energy = 0.45;
  } else {
    text = "Levante as mãos para sinalizar";
  }
  orb.setEnergy(energy);
  if (text !== liveText) { liveText = text; setStatus(text, live); }
  const pill = $("camPill");
  pill.classList.toggle("signing", state === "signing");
  $("camPillText").textContent = state === "signing" ? "Observando" : "Câmera ativa";
}

function setStatus(text, live = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("live", live);
}

// ------------------------------------------------------------------ responses

function showResponse(word, caption, { spelling = false, unknown = false, quiet = false } = {}) {
  const w = $("word");
  w.textContent = word;
  w.className = "word" + (spelling ? " spelling" : "") + (unknown ? " unknown" : "");
  void w.offsetWidth;
  if (!quiet) w.classList.add("enter");
  if (typeof caption === "string") $("caption").textContent = caption;
  else { $("caption").replaceChildren(...caption); }
  clearTimeout(dimTimer);
  if (!quiet) dimTimer = setTimeout(() => w.classList.add("dim"), 7000);
}

function confCaption(prefix, p) {
  const c = document.createElement("span");
  c.className = "conf";
  c.textContent = Math.round(p * 100) + "%";
  return [document.createTextNode(prefix + " · "), c, document.createTextNode(" de confiança")];
}

function onPrediction(p) {
  if (!p.recognized) {
    const near = p.label ? `Parecido com ${display(p.label)} (${Math.round(p.confidence * 100)}%)` : "";
    showResponse("Não entendi. Tente de novo.", near, { unknown: true });
    orb.shake();
    orb.pulse(0.25);
    return;
  }

  if (p.replacesPrevious) {
    // e.g. the "U" handshape at the start of NOME was shown first: drop it
    if (spell) spell = spell.slice(0, -1);
    else if (history.length && !history[history.length - 1].letter) history.pop();
    renderHistory();
  }

  if (isLetter(p.label)) {
    spell += p.label;
    lastLetterAt = performance.now();
    highlightName();
    if (spell.endsWith(NAME)) {
      showResponse("Julia", "Nome soletrado: J · U · L · I · A");
      addHistory("Julia");
      spell = "";
      orb.pulse(1.2);
      setTimeout(highlightName, 1800);
      return;
    }
    showResponse(spell, confCaption(`Letra ${p.label}`, p.confidence), { spelling: true });
    orb.pulse(0.6);
    return;
  }

  finishSpell();
  showResponse(display(p.label), confCaption("Sinal reconhecido", p.confidence));
  addHistory(display(p.label));
  orb.pulse(1);
}

function finishSpell() {
  if (!spell) return;
  addHistory(spell, true);
  spell = "";
  highlightName();
}

// word break: hands down for a moment after a letter
setInterval(() => {
  const now = performance.now();
  if (spell && now - lastHandsUpAt > WORD_BREAK_MS && now - lastLetterAt > WORD_BREAK_MS) finishSpell();
}, 250);

function addHistory(text, letter = false) {
  history.push({ text, letter });
  if (history.length > 14) history.shift();
  renderHistory();
}
function renderHistory() {
  const chips = $("chips");
  if (!history.length) {
    chips.innerHTML = '<span class="empty">Os sinais reconhecidos aparecem aqui.</span>';
    return;
  }
  chips.replaceChildren(...history.map((h) => {
    const s = document.createElement("span");
    s.className = "chip";
    s.textContent = h.text;
    return s;
  }));
}

/** Light up J U L I A in the signs card as the name is spelled. */
function highlightName() {
  const boxes = document.querySelectorAll(".letters b");
  let n = 0;
  for (let k = Math.min(spell.length, NAME.length); k > 0; k--) {
    if (spell.endsWith(NAME.slice(0, k))) { n = k; break; }
  }
  boxes.forEach((b, i) => b.classList.toggle("hit", i < n));
}

// ------------------------------------------------------------------ mode

const HINTS = {
  words: "Reconhece palavras e letras soltas.",
  spelling: "Só letras, juntadas numa palavra. Abaixe as mãos para terminar.",
};
function setMode(m) {
  mode = m;
  document.querySelectorAll(".segmented button").forEach((b) => {
    const on = b.dataset.mode === m;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
  });
  $("modeHint").textContent = HINTS[m];
  finishSpell();
  if (recognizer) { recognizer.spelling = m === "spelling"; recognizer.reset(); }
}
document.querySelectorAll(".segmented button").forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });
$("linkSpell").onclick = () => {
  setMode("spelling");
  document.querySelector(".stage-card").scrollIntoView({ behavior: "smooth", block: "center" });
};

// pause cleanly when the tab is hidden
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { if (recognizer) recognizer.reset(); finishSpell(); }
});
