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
  // We run AST on WASM (see ast.js for why WebGPU is disabled).
  // ~1-2 s per 30-second window on a recent laptop; one file = ~3-6 s of inference.
  els.acceleratorHint.textContent =
    "Browser accelerator: WASM (CPU). Inference takes a few seconds per file.";

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
// The onLoadProgress callback receives a fraction [0..1] of the *download* phase.
async function ensureAstLoaded(onLoadProgress) {
  if (astInstance) return astInstance;
  astInstance = await AST.load(({ stage, progress, file }) => {
    if (stage === "downloading" && progress != null) {
      const fileLabel = file ? ` (${file})` : "";
      onLoadProgress(progress, `Downloading AST model${fileLabel}…`);
    } else if (stage === "ready") {
      onLoadProgress(1, "Model loaded. Initializing…");
    } else {
      onLoadProgress(0, "Downloading AST model (~87 MB, one time)…");
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
//
// End-to-end progress design:
//   We map each stage onto a sub-segment [start..end] of the OVERALL bar so the
//   user sees one bar that fills monotonically from 0% -> 100% across the
//   whole pipeline — not a bar that resets per task. The text label still
//   tells them which stage is running (good context), but the bar tells them
//   how far through the whole job they are.
//
//   Segment weights (rough, calibrated to wall-clock):
//     - First-time run (model download dominates):
//         decode 0.00..0.05  windows 0.05..0.07  download 0.07..0.55
//         inference 0.55..1.00
//     - Cached run (no download):
//         decode 0.00..0.05  windows 0.05..0.07  inference 0.07..1.00
//
//   Inference itself is N windows; each window claims an equal slice of the
//   inference segment. The fill bar also has a CSS shimmer overlay so even
//   when a slice is mid-window (no movement for ~1.5 s) the bar still looks
//   alive ("something is happening").
function planSegments(modelAlreadyLoaded) {
  if (modelAlreadyLoaded) {
    return { decode: [0, 0.05], windowing: [0.05, 0.07], download: null, inference: [0.07, 1.0] };
  }
  return { decode: [0, 0.05], windowing: [0.05, 0.07], download: [0.07, 0.55], inference: [0.55, 1.0] };
}

function lerp([a, b], t) { return a + (b - a) * Math.max(0, Math.min(1, t)); }

async function handleFile(file) {
  hideError();
  hideResults();
  els.fileNameLabel.textContent = file.name;
  const seg = planSegments(astInstance !== null);
  try {
    setStage("decoding", lerp(seg.decode, 0), "Decoding audio…");
    const samples = await decodeTo16kMono(file);
    const durationSec = samples.length / 16000;
    setStage("decoding", lerp(seg.decode, 1), "Decoding audio…");

    setStage("windowing", lerp(seg.windowing, 0), "Extracting 30-second windows…");
    const { windows, info } = extractWindows(samples);
    if (windows.length === 0) {
      throw new Error(info || "Audio is too short to classify (need at least 5 seconds).");
    }
    setStage("windowing", lerp(seg.windowing, 1), "Extracting 30-second windows…");

    const ast = await ensureAstLoaded((p, label) => {
      if (seg.download) setStage("downloading", lerp(seg.download, p), label);
    });

    // Inference loop.
    //
    // Each ast.embed() call blocks the main thread synchronously for ~1.5-2 s
    // (WASM does not yield). The browser cannot paint during that block —
    // not the bar, not a spinner, not even a CSS shimmer animation reliably.
    // So we set expectations honestly:
    //   * Label tells the user *up front* roughly how long this will take.
    //   * Bar advances cleanly between windows (visible jumps after each).
    //   * We yield to the browser BEFORE each WASM call with setTimeout so
    //     the new state actually paints before the freeze.
    // No fake animation that would look broken when it freezes.
    const infSeg = seg.inference;
    const perWindow = [];
    // estWindowMs starts unknown — we don't show an ETA on the first window
    // because a bad guess up-front ("5s left") looks worse than no number when
    // the actual answer turns out to be 30s. After window 1 we have real data.
    let estWindowMs = null;

    for (let i = 0; i < windows.length; i++) {
      const remaining = windows.length - i;
      const baseLabel = `Analyzing window ${i + 1} of ${windows.length}`;
      const etaSuffix = estWindowMs
        ? ` · about ${Math.max(1, Math.round((remaining * estWindowMs) / 1000))}s left…`
        : "…";
      setStage("inference", lerp(infSeg, i / windows.length),
               baseLabel + etaSuffix);
      // Macrotask yield so the label + bar actually paint before WASM blocks.
      // rAF alone isn't enough: its callback runs BEFORE the next commit, but
      // a synchronous WASM call on the next line would prevent the commit
      // from happening, deferring the callback until after WASM finishes.
      // setTimeout puts us in the macrotask queue AFTER the browser has had a
      // chance to paint.
      await new Promise(r => setTimeout(r, 32));

      const winStart = performance.now();
      const emb = await ast.embed(windows[i]);
      const out = probeInstance.predict(emb);
      if (new URLSearchParams(window.location.search).get("debug") === "1") {
        console.log(`[DEBUG window ${i + 1} probe]`, {
          logits: out.logits.map((v) => v.toFixed(4)),
          p_studio: out.pStudio.toFixed(4),
          p_live: out.pLive.toFixed(4),
          label: out.label,
        });
      }
      perWindow.push(out);
      // Calibrate from the actual first window so the ETA on subsequent
      // windows matches the user's hardware.
      const actual = performance.now() - winStart;
      estWindowMs = i === 0 ? actual : 0.6 * estWindowMs + 0.4 * actual;
    }
    setStage("inference", 1, "Finishing up…");

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
