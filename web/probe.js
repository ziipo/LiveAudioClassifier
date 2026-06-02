// probe.js — the trained linear-probe head, in 30 lines of plain JS.
//
// We have a (2, 768) weight matrix and a (2,) bias trained in Python (see
// scripts/07_train_linear_probe.py). Inference is one matmul + softmax;
// no need for any matrix library at this size — saves ~50 KB of deps.
//
// Usage:
//   const probe = await Probe.load("probe-weights.json");
//   const { pStudio, pLive, label, confidence } = probe.predict(embedding768);

export class Probe {
  /**
   * @param {object} cfg parsed probe-weights.json
   */
  constructor(cfg) {
    this.W = cfg.weight;         // [2][768]
    this.b = cfg.bias;           // [2]
    this.inDim = cfg.in_dim;     // 768
    this.id2label = cfg.id2label; // {"0": "studio", "1": "live"}
    this.metrics = {
      trainedValAcc: cfg.trained_val_acc,
      trainedTestAcc: cfg.trained_test_acc,
      sourceQualityTestAcc: cfg.source_quality_test_acc,
    };
    this.version = cfg.version;
    if (!this.W || this.W.length !== 2 || this.W[0].length !== this.inDim) {
      throw new Error(`probe weight shape mismatch: expected (2, ${this.inDim})`);
    }
  }

  /**
   * Fetch the probe weights from a URL and construct the Probe.
   * @param {string} url e.g. "probe-weights.json"
   */
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`failed to load probe weights from ${url}: ${r.status}`);
    return new Probe(await r.json());
  }

  /**
   * Run the probe on one 768-d embedding (the AST pooler_output).
   * @param {Float32Array|number[]} embedding length 768
   * @returns {{pStudio: number, pLive: number, label: string, confidence: number,
   *           logits: [number, number]}}
   */
  predict(embedding) {
    if (embedding.length !== this.inDim) {
      throw new Error(`embedding length ${embedding.length} != expected ${this.inDim}`);
    }
    // Logits = W·x + b. Manually unrolled for clarity; 2*768 = 1,536 multiplies,
    // V8 inlines this in ~tens of microseconds.
    const logits = [this.b[0], this.b[1]];
    const W0 = this.W[0], W1 = this.W[1];
    for (let i = 0; i < this.inDim; i++) {
      logits[0] += W0[i] * embedding[i];
      logits[1] += W1[i] * embedding[i];
    }
    // Softmax with max-subtraction for numerical stability.
    const m = Math.max(logits[0], logits[1]);
    const e0 = Math.exp(logits[0] - m);
    const e1 = Math.exp(logits[1] - m);
    const sum = e0 + e1;
    const pStudio = e0 / sum;
    const pLive = e1 / sum;
    // label2id was {"studio": 0, "live": 1}, so id 1 = live.
    const winner = pLive >= pStudio ? 1 : 0;
    return {
      pStudio,
      pLive,
      label: this.id2label[String(winner)],
      confidence: winner === 1 ? pLive : pStudio,
      logits,
    };
  }
}
