// ast.js — runs AST (Audio Spectrogram Transformer) in the browser and returns
// the (768,) pooler_output.
//
// Why this is wired the way it is:
//
//   * For PREPROCESSING (mel-spectrogram), we still use transformers.js's
//     AutoFeatureExtractor — its JS port of ASTFeatureExtractor matches the
//     Python version bit-for-bit, including the model-specific normalisation
//     (mean=-4.2677, std=4.5689). Writing this from scratch and matching it
//     numerically takes days; using theirs is one call.
//
//   * For INFERENCE, we run our OWN ONNX file with onnxruntime-web.
//
//     We tried transformers.js's AutoModel pointing at Xenova's ONNX, but
//     that export is the FULL ASTForAudioClassification model — its only
//     output is 527-class AudioSet logits, not pooler_output. Our linear
//     probe was trained on pooler_output, so we need a backbone-only ONNX
//     that exposes that vector. We export it ourselves (see
//     _internal/scripts/12_export_ast_backbone_onnx.py) and ship it from
//     the same origin as the page (web/model.onnx, ~85 MB).
//
// Public API:
//   const ast = await AST.load(onProgress);  // loads feature extractor + ONNX
//   const emb768 = await ast.embed(samples16k);

import {
  AutoFeatureExtractor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.mjs";

// Tell onnxruntime-web where to find its WASM workers. Pinning to the same
// version's CDN URL avoids version skew.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

// Preprocessor: the original MIT repo (Xenova mirrors its preprocessor config too).
const PREPROCESSOR_ID = "Xenova/ast-finetuned-audioset-10-10-0.4593";

// Our self-hosted backbone-only ONNX, served same-origin from web/.
const MODEL_URL = "./model.onnx";

export class AST {
  constructor(featureExtractor, session) {
    this.featureExtractor = featureExtractor;
    this.session = session;
  }

  /**
   * Load the AST feature extractor (mel preprocessing) and our ONNX backbone.
   *
   * @param {(status: {stage: string, progress?: number, file?: string}) => void}
   *   [onProgress] called repeatedly during loading so the UI can show a bar.
   */
  static async load(onProgress) {
    onProgress?.({ stage: "loading", progress: 0 });

    // Feature extractor: tiny (<100 KB), fetches preprocessor_config.json.
    // We map transformers.js's progress callback into our shape.
    const wrap = (raw) => {
      if (!onProgress) return;
      const stage = raw.status === "progress" ? "downloading"
                  : raw.status === "ready"    ? "ready"
                  : raw.status;
      const progress = typeof raw.progress === "number"
        ? Math.min(1, raw.progress / 100)
        : undefined;
      onProgress({ stage, progress, file: raw.file });
    };
    const featureExtractor = await AutoFeatureExtractor.from_pretrained(
      PREPROCESSOR_ID,
      { progress_callback: wrap }
    );

    // Backbone ONNX: 85 MB, served same-origin from this Pages site.
    // We fetch it ourselves so we can report download progress (an
    // InferenceSession.create with a URL has no progress hook).
    const modelBytes = await fetchWithProgress(MODEL_URL, (loaded, total) => {
      onProgress?.({
        stage: "downloading",
        progress: total ? loaded / total : undefined,
        file: "model.onnx",
      });
    });

    // IMPORTANT: WASM-only, no WebGPU.
    //
    // We previously preferred WebGPU and fell back to WASM. But for THIS
    // int8-quantized backbone, the onnxruntime-web WebGPU EP silently
    // produces incorrect numerical output — the 768-d pooler_output L2
    // norm shrinks by ~17% (34.5 -> 28.6 on a known reference clip),
    // which is enough to flip nearly every studio prediction to "live"
    // with high confidence. Likely cause: a per-op MatMulInteger /
    // DynamicQuantizeLinear path on WebGPU that disagrees with the
    // CPU/WASM implementation. There is no error thrown — just bad numbers.
    //
    // WASM-only matches Python's onnxruntime to 4 decimal places on the
    // same files. Inference is ~1-2 s per 30 s window instead of ~300 ms,
    // which is fine for a single-click demo.
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    onProgress?.({ stage: "ready", progress: 1 });
    return new AST(featureExtractor, session);
  }

  /**
   * Run a forward pass on a 30s, 16 kHz mono Float32Array and return the
   * (768,) pooler_output as a plain JS array.
   *
   * @param {Float32Array} samples16k expected length 30 * 16000 = 480_000
   * @returns {Promise<number[]>} length 768
   */
  async embed(samples16k) {
    // Mel preprocessing -> shape (1, 1024, 128) float32 tensor.
    const features = await this.featureExtractor(samples16k, {
      sampling_rate: 16000,
    });
    const input = features.input_values;

    // [DEBUG] When ?debug=1 is in the URL, log stats so we can compare to
    // Python's ASTFeatureExtractor output. On a 30s studio clip, Python
    // produces mel mean≈0.47, std≈0.31, min≈-1.28, max≈1.26 (post-normalization
    // with model-specific mean=-4.27, std=4.57). If the JS port produces
    // raw log-mels (mean≈-4, std≈4) we've found the bug.
    if (typeof window !== "undefined" &&
        new URLSearchParams(window.location.search).get("debug") === "1") {
      const d = input.data;
      let s = 0, sq = 0, mn = Infinity, mx = -Infinity;
      for (let i = 0; i < d.length; i++) {
        const v = d[i]; s += v; sq += v * v;
        if (v < mn) mn = v; if (v > mx) mx = v;
      }
      const mean = s / d.length;
      const std = Math.sqrt(sq / d.length - mean * mean);
      const sampMin = Math.min(...samples16k.slice(0, 100000));
      const sampMax = Math.max(...samples16k.slice(0, 100000));
      console.log("[DEBUG audio→mel]", {
        samples_len: samples16k.length,
        samples_first100k_min: sampMin.toFixed(4),
        samples_first100k_max: sampMax.toFixed(4),
        mel_dims: input.dims,
        mel_mean: mean.toFixed(4),
        mel_std: std.toFixed(4),
        mel_min: mn.toFixed(4),
        mel_max: mx.toFixed(4),
        expected_mean_studio: "≈0.47 (normalized) — if you see ≈-4 the normalization is missing",
      });
    }

    // transformers.js returns its own Tensor; ORT wants ort.Tensor. The data
    // array is compatible (Float32Array).
    const ortInput = new ort.Tensor("float32", input.data, input.dims);

    const outputs = await this.session.run({ input_values: ortInput });
    const pooler = outputs.pooler_output;
    if (!pooler) {
      throw new Error(
        "ONNX session returned no pooler_output. Available outputs: " +
        Object.keys(outputs).join(", ")
      );
    }
    if (pooler.data.length !== 768) {
      throw new Error(
        `pooler_output length ${pooler.data.length} != 768`
      );
    }

    if (typeof window !== "undefined" &&
        new URLSearchParams(window.location.search).get("debug") === "1") {
      const p = pooler.data;
      let s = 0, sq = 0, l2 = 0;
      for (let i = 0; i < p.length; i++) { s += p[i]; sq += p[i] * p[i]; l2 += p[i] * p[i]; }
      const mean = s / p.length;
      console.log("[DEBUG ONNX pooler_output]", {
        len: p.length,
        mean: mean.toFixed(4),
        std: Math.sqrt(sq / p.length - mean * mean).toFixed(4),
        L2: Math.sqrt(l2).toFixed(4),
        expected_L2: "≈34.5 (Python ref on Tigerlily)",
      });
    }

    return Array.from(pooler.data);
  }
}

/**
 * fetch() that streams the response body and reports progress as it goes.
 * Used for the 85 MB model download so the UI can show a real progress bar
 * instead of a spinner. Falls back to non-progress fetch if streaming isn't
 * supported (older browsers).
 */
async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to load ${url}: HTTP ${response.status}`);
  }
  const total = Number(response.headers.get("content-length")) || 0;

  // If streaming isn't available, fall back to .arrayBuffer() (no progress).
  if (!response.body || !response.body.getReader) {
    onProgress(0, total);
    const buf = await response.arrayBuffer();
    onProgress(buf.byteLength, total || buf.byteLength);
    return new Uint8Array(buf);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  // Concatenate chunks into a single Uint8Array.
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out;
}

// Useful for the UI: predict whether the browser has WebGPU available.
// We show the user a hint based on this — "WebGPU (fast)" vs "WASM (slower)".
export async function detectAccelerator() {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch (_) {}
  }
  return "wasm";
}
