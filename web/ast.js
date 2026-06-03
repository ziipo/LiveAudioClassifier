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

    // Try WebGPU first; fall back to wasm if it's not available or fails.
    let session;
    try {
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["webgpu", "wasm"],
        graphOptimizationLevel: "all",
      });
    } catch (e) {
      console.warn("ONNX session create failed with WebGPU; falling back to WASM", e);
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    }

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
