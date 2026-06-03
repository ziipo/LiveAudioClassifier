# LiveAudioClassifier

A neural network that listens to a piece of music and predicts whether the recording is **a live recording** or **not**. Try it in your browser, or download the classifier and run it locally.

> 🎙️ **[Try the demo →](https://ziipo.github.io/LiveAudioClassifier/)** *(link will work after first deploy)*

My testing shows the classifier achieves about **94.5%** accuracy on a held-out test set and **93.9%** on an adversarial test set of deliberately-hard recordings (clean soundboard mixes from the Internet Archive's Live Music Archive). The web demo runs entirely client-side via [transformers.js](https://huggingface.co/docs/transformers.js) and [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) — your audio never leaves your device.

## How it works

The classifier is a two-part model:

1. **AST backbone** ([MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)) — 86 million parameters, pretrained on AudioSet to understand audio at a general level. Frozen during my training.
2. **Linear probe** — 1,538 parameters. A single `nn.Linear(768, 2)` layer trained on the AST embeddings to make the live-vs-studio decision.

The probe weights live at [`web/probe-weights.json`](web/probe-weights.json) (~42 KB). The AST backbone is shipped as an int8-quantized ONNX file (~85 MB) served from the same GitHub Pages origin, downloaded once on first visit and browser-cached. For each input clip, the pipeline takes three 30-second windows at 25/50/75% of the usable span, runs each through AST + the probe, and takes the majority vote.

## What it was trained on

24,227 30-second audio clips, split ~80/10/10 into train/val/test at the *track* level (no clip leakage between splits):

| Source | Clips | Label |
|---|---|---|
| Internet Archive Live Music Archive | 10,149 | live |
| Free Music Archive | 7,994 | studio |
| Personal collection | 6,084 | live |
| **Total** | **24,227** | 67% live / 33% studio |

## How well it does

| Test set | Accuracy |
|---|---|
| Auto-split test (2,401 clips drawn from the same distribution as training) | **94.46%** |
| Source-quality test (2,043 clips from 150 SBD/AUD/MTX-tagged Internet Archive shows the model has never seen, picked to be deliberately adversarial) | **93.93%** |

The 0.5-pp drop on the harder test set is small — the model generalizes well. Specifically, it doesn't rely on crowd-noise shortcuts: soundboard-mixed live recordings (no crowd) still score 93.1%.

Both tests use a held-out FMA studio split and various live sources.

## Running the local CLI

If you'd rather classify a folder of audio without a browser — or work offline once the model is cached — there's a Python command-line version at `scripts/predict.py`.

```bash
# One-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Classify one or more files
python scripts/predict.py path/to/song.mp3
python scripts/predict.py *.mp3

# Machine-readable output
python scripts/predict.py --json song.mp3
```

The CLI uses the same probe weights as the browser demo (`web/probe-weights.json`) and the full PyTorch AST backbone (downloaded automatically from HuggingFace on first run, ~350 MB). On Apple Silicon and CUDA machines it runs on GPU/MPS automatically; pass `--device cpu` to force CPU.

The install footprint is large (~1 GB) because PyTorch is. The browser demo is the lighter option if you don't want a Python environment.

Example output:

```
song.mp3
  prediction: live
  confidence: 0.913  (3/3 live)
  per-window p(live): 0.872 0.940 0.928
  (three windows at 25/50/75% of usable span (240.0s clip))
```

## Privacy

Your audio never leaves your device. The web demo's only network requests are:

- One-time download of the AST backbone (~85 MB ONNX file from the same GitHub Pages origin, cached forever afterward)
- One-time download of `transformers.js` and `onnxruntime-web` from jsDelivr
- The static page assets from GitHub Pages

You can verify this in your browser's DevTools Network tab. There is no server, no analytics, no telemetry. The local CLI similarly fetches the model from HuggingFace once and then runs offline.

## Limitations

- **First load is slow.** The AST backbone is ~85 MB (browser) or ~350 MB (CLI). Both are cached after the first run.
- **WebAssembly fallback is noticeably slower** than WebGPU. The page detects which is available and shows a hint. WebGPU works in recent Chrome/Edge/Safari; Firefox falls back to WASM.
- **Short audio.** Anything under 5 seconds is rejected; 5-30s is padded to 30s and gets a single window; 60s+ gets the full three-window treatment.
- **Studio recordings made to sound live** (reverb-heavy production, room mics, single-take sessions) can be misclassified as live. Very clean soundboard-style live recordings can occasionally go the other way.

## Acknowledgements

- [MIT CSAIL](https://huggingface.co/MIT) for the [Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- [Xenova / HuggingFace](https://huggingface.co/Xenova) for the pre-converted ONNX variants used by the browser demo
- [transformers.js](https://huggingface.co/docs/transformers.js) for browser inference machinery
- The [Internet Archive Live Music Archive](https://archive.org/details/etree) for ~10,000 live clips
- The [Free Music Archive](https://github.com/mdeff/fma) for the studio counterpart

## License

MIT. See [LICENSE](LICENSE).
