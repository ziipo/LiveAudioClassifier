// app.js — page orchestration. Wires file upload -> audio -> AST -> probe ->
// rendering. Keeps DOM access in one place and the ML logic in the modules.

import { AST, detectAccelerator } from "./ast.js";
import { Probe } from "./probe.js";
import { decodeTo16kMono, extractWindows } from "./audio.js";

// --- DOM handles, all looked up once on load ---------------------------------
const $ = (id) => document.getElementById(id);
const els = {
  dropzone: $("dropzone"),
  fileInput: $("fileInput"),
  stageLabel: $("stageLabel"),
  progressBar: $("progressBar"),
  progressFill: $("progressFill"),
  results: $("results"),
  fileNameLabel: $("fileNameLabel"),
  verdictLabel: $("verdictLabel"),
  verdictDetail: $("verdictDetail"),
  windowsContainer: $("windowsContainer"),
  windowInfo: $("windowInfo"),
  errorBox: $("errorBox"),
  acceleratorHint: $("acceleratorHint"),
  metricsHint: $("metricsHint"),
};

// --- One-time setup: load probe weights and AST in parallel ------------------
let astInstance = null;
let probeInstance = null;

async function init() {
  // Detect WebGPU vs WASM up front so we can warn the user if WASM-only
  // (inference will be ~10x slower).
  detectAccelerator().then((kind) => {
    els.acceleratorHint.textContent = kind === "webgpu"
      ? "Browser accelerator: WebGPU (fast)"
      : "Browser accelerator: WASM (slower — try Chrome or Edge for WebGPU)";
  });

  // Probe loads instantly (~42 KB JSON). Show its metrics as soon as we have them.
  probeInstance = await Probe.load("probe-weights.json");
  els.metricsHint.textContent =
    `Probe v${probeInstance.version} — ` +
    `${(probeInstance.metrics.trainedTestAcc * 100).toFixed(1)}% on auto-split test, ` +
    `${(probeInstance.metrics.sourceQualityTestAcc * 100).toFixed(1)}% on adversarial test`;

  // AST is the slow one (~87 MB int8 download). Defer until the user actually
  // picks a file — no point making a passive visitor pay 87 MB of bandwidth.
}

// --- Lazy AST load. Called on the first file the user picks. ------------------
async function ensureAstLoaded() {
  if (astInstance) return astInstance;
  setStage("loading", 0, "Downloading AST model (~87 MB, one time)…");
  astInstance = await AST.load(({ stage, progress, file }) => {
    if (stage === "downloading" && progress != null) {
      const fileLabel = file ? ` (${file})` : "";
      setStage("downloading", progress, `Downloading AST model${fileLabel}…`);
    } else if (stage === "ready") {
      setStage("loading", 1, "Model loaded. Initializing…");
    }
  });
  return astInstance;
}

// --- Wiring: drag-and-drop + file picker -------------------------------------
function setupDropzone() {
  ["dragenter", "dragover"].forEach((ev) => {
    els.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      els.dropzone.classList.add("drag-active");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    els.dropzone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      els.dropzone.classList.remove("drag-active");
    });
  });
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });
  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });
  els.fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  });
}

// --- The pipeline ------------------------------------------------------------
async function handleFile(file) {
  hideError();
  hideResults();
  els.fileNameLabel.textContent = file.name;
  try {
    setStage("decoding", 0, "Decoding audio…");
    const samples = await decodeTo16kMono(file);
    const durationSec = samples.length / 16000;

    setStage("windowing", 0, "Extracting 30-second windows…");
    const { windows, info } = extractWindows(samples);
    if (windows.length === 0) {
      throw new Error(info || "Audio is too short to classify (need at least 5 seconds).");
    }

    const ast = await ensureAstLoaded();
    setStage("inference", 0, "Running AST on each window…");
    const perWindow = [];
    for (let i = 0; i < windows.length; i++) {
      setStage("inference", i / windows.length,
               `Window ${i + 1} of ${windows.length}…`);
      const emb = await ast.embed(windows[i]);
      const out = probeInstance.predict(emb);
      perWindow.push(out);
    }

    setStage("done", 1, "Done.");
    renderResults({ file, durationSec, info, perWindow });
  } catch (err) {
    console.error(err);
    showError(err.message || String(err));
    setStage("idle", 0, "");
  }
}

// --- Rendering ---------------------------------------------------------------
function renderResults({ file, durationSec, info, perWindow }) {
  // Majority vote + average confidence — same as the Python CLI.
  let nLive = 0, sumPLive = 0;
  for (const w of perWindow) {
    if (w.label === "live") nLive++;
    sumPLive += w.pLive;
  }
  const finalLabel = nLive >= Math.ceil(perWindow.length / 2) ? "live" : "studio";
  const avgPLive = sumPLive / perWindow.length;
  const finalConfidence = finalLabel === "live" ? avgPLive : 1 - avgPLive;

  els.windowInfo.textContent =
    `Duration ${durationSec.toFixed(1)}s · ${info}`;

  els.verdictLabel.textContent = finalLabel.toUpperCase();
  els.verdictLabel.dataset.label = finalLabel;
  els.verdictDetail.textContent =
    `${(finalConfidence * 100).toFixed(1)}% confidence ` +
    `(${nLive} of ${perWindow.length} windows agree)`;

  els.windowsContainer.innerHTML = "";
  perWindow.forEach((w, i) => {
    const card = document.createElement("div");
    card.className = `window-card window-${w.label}`;
    const pct = (w.pLive * 100).toFixed(1);
    card.innerHTML = `
      <div class="window-title">Window ${i + 1}</div>
      <div class="window-label">${w.label.toUpperCase()}</div>
      <div class="window-bar"><div class="window-bar-fill" style="width: ${pct}%"></div></div>
      <div class="window-prob">p(live) = ${pct}%</div>
    `;
    els.windowsContainer.appendChild(card);
  });

  els.results.classList.remove("hidden");
}

function hideResults()  { els.results.classList.add("hidden"); }
function showError(m)   { els.errorBox.textContent = m; els.errorBox.classList.remove("hidden"); }
function hideError()    { els.errorBox.classList.add("hidden"); }

function setStage(stage, progress, label) {
  els.stageLabel.textContent = label || "";
  const visible = stage !== "idle";
  els.progressBar.classList.toggle("hidden", !visible);
  els.progressFill.style.width = `${Math.round(progress * 100)}%`;
}

// --- Boot --------------------------------------------------------------------
setupDropzone();
init().catch((err) => {
  console.error(err);
  showError(`Failed to initialize: ${err.message || err}`);
});
