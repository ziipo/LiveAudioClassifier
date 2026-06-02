// ast.js — wraps transformers.js to run AST (Audio Spectrogram Transformer)
// in the browser and return the (768,) pooler_output.
//
// We use transformers.js for the *whole* AST forward pass because:
//   1. Their ASTFeatureExtractor JS implementation matches the Python one
//      bit-for-bit (including mean=-4.2677, std=4.5689 normalization).
//   2. Writing a mel-spectrogram implementation by hand and matching it
//      numerically takes days; using theirs is one line.
//   3. They wrap onnxruntime-web internally, so we still get WebGPU when
//      available and WASM as a fallback — same runtime behavior.
//
// The model: MIT/ast-finetuned-audioset-10-10-0.4593, int8-quantized version
// hosted at Xenova/ast-finetuned-audioset-10-10-0.4593. ~87 MB download,
// cached by the browser after first visit.
//
// Public API:
//   const ast = await AST.load();                  // loads model + feature extractor
//   const emb768 = await ast.embed(samples16k);   // samples = Float32Array, 16 kHz mono
//   ast.warmupStatus()                              // "downloading" | "loading" | "ready"

import {
  AutoModel,
  AutoFeatureExtractor,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";

const MODEL_ID = "Xenova/ast-finetuned-audioset-10-10-0.4593";

// Default transformers.js will pick model_quantized.onnx (~87 MB) when dtype
// isn't specified for an int8-quantized repo. Setting it explicitly documents
// intent and guards against the library's default changing.
const MODEL_OPTS = { dtype: "q8" };

export class AST {
  constructor(featureExtractor, model) {
    this.featureExtractor = featureExtractor;
    this.model = model;
  }

  /**
   * Load AST + its feature extractor. The model file (~87 MB) is fetched from
   * HuggingFace's CDN; the browser caches it so subsequent visits are instant.
   * @param {(status: {stage: string, progress?: number}) => void} [onProgress]
   *   Called repeatedly during the model download so the UI can show a bar.
   */
  static async load(onProgress) {
    // transformers.js's progress_callback receives `{ status, file, progress, total }`.
    // We map that down to something the UI can use directly.
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
    onProgress?.({ stage: "loading", progress: 0 });

    const featureExtractor = await AutoFeatureExtractor.from_pretrained(
      MODEL_ID,
      { progress_callback: wrap }
    );
    const model = await AutoModel.from_pretrained(
      MODEL_ID,
      { ...MODEL_OPTS, progress_callback: wrap }
    );
    onProgress?.({ stage: "ready", progress: 1 });
    return new AST(featureExtractor, model);
  }

  /**
   * Run a forward pass on a 30s, 16 kHz mono Float32Array and return the
   * (768,) pooler_output as a plain JS array of numbers.
   *
   * @param {Float32Array} samples16k expected length 30 * 16000 = 480_000
   * @returns {Promise<number[]>} length 768
   */
  async embed(samples16k) {
    // The feature extractor takes raw audio + sample rate, returns input_values
    // tensor of shape [1, 1024, 128] — the log-mel spectrogram AST expects.
    const features = await this.featureExtractor(samples16k, {
      sampling_rate: 16000,
    });
    // Model forward pass. ASTModel returns last_hidden_state + pooler_output.
    // We only need pooler_output (the [CLS] projection).
    const outputs = await this.model(features);
    if (!outputs.pooler_output) {
      throw new Error(
        "AST forward returned no pooler_output — model variant mismatch?"
      );
    }
    // pooler_output is a Tensor with shape [1, 768]. Extract the flat data.
    const tensor = outputs.pooler_output;
    const data = tensor.data; // typed array (Float32Array or similar)
    if (data.length !== 768) {
      throw new Error(`pooler_output length ${data.length} != 768`);
    }
    return Array.from(data);
  }
}

// Useful for the UI: predict whether the browser has WebGPU available.
// transformers.js auto-detects this, but we want to show the user a hint
// like "your browser is using WebGPU (fast)" vs "WASM (slow)".
export async function detectAccelerator() {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch (_) {}
  }
  return "wasm";
}
