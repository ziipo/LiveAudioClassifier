# LiveAudioClassifier

A neural network that listens to a piece of music and predicts whether the recording is **a live recording** or **not**. Try it in your browser or download the classifier and run it locally. 

> 🎙️ **[Try the demo →](https://ziipo.github.io/LiveAudioClassifier/)** *(link will work after first deploy)*

My testing shows the classifier achieves about **94.5%** accuracy on a held-out test set and **93.9%** on an adversarial test set of deliberately-hard recordings (clean soundboard mixes from the Internet Archive's Live Music Archive). For web demo, all inference happens client-side via [transformers.js](https://huggingface.co/docs/transformers.js) and [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).

## How it works

The classifier is a two-part model:

1. **AST backbone** ([MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)) — 86 million parameters, pretrained on AudioSet to understand audio at a general level. Frozen during my training.
2. **Linear probe** — 1,538 parameters. A single `nn.Linear(768, 2)` layer trained on the AST embeddings to make the live-vs-studio decision.

The probe weights live at [`web/probe-weights.json`](web/probe-weights.json) (~42 KB). The AST backbone is downloaded once from HuggingFace's CDN (~87 MB, int8-quantized) and browser-cached.

## What it was trained on

24,227 30-second audio clips, split ~80/10/10 into train/val/test at the *track* level (no clip leakage between splits):

| Source | Clips | Label |
|---|---|---|
| Internet Archive Live Music Archive | 10,149 | live |
| Free Music Archive  | 7,994 | studio |
| personal collection | 6,084 | live |
| **Total** | **24,227** | 67% live / 33% studio |

## How well it does

| Test set | Accuracy |
|---|---|
| Auto-split test (2,401 clips drawn from the same distribution as training) | **94.46%** |
| Source-quality test (2,043 clips from 150 SBD/AUD/MTX-tagged Internet Archive shows the model has never seen, picked to be deliberately adversarial) | **93.93%** |

The 0.5-pp drop on the harder test set is small — the model generalizes well. Specifically, it doesn't rely on crowd-noise shortcuts: soundboard-mixed live recordings (no crowd) still score 93.1%.

Both tests use a held-out FMA studio split and various live sources.

## Repository layout

```
web/                              # the GitHub Pages demo
  index.html                      # the page itself
  app.js                          # orchestration: upload → audio → AST → probe → render
  ast.js                          # transformers.js wrapper for AST inference
  audio.js                        # decode → 16 kHz mono → 30s windows
  probe.js                        # 25-line JS implementation of the linear probe
  probe-weights.json              # trained weights (42 KB)
  styles.css

scripts/                          # Python training & data pipeline
  01_fetch_internet_archive.py    # pull live shows from IA
  02_sort_manual.py               # sort hand-collected MP3s into subfolders
  04_extract_clips.py             # WAV → 30s clips at 16 kHz mono
  05_extract_features.py          # WAV clips → AST mel features cache
  06_extract_embeddings.py        # AST forward pass once → embeddings cache
  07_train_linear_probe.py        # train the head
  08_fetch_source_quality_test.py # pull adversarial SBD/AUD/MTX test set
  09_extract_test_set.py          # process adversarial test audio
  10_evaluate_test_set.py         # score a checkpoint on the adversarial test
  11_export_probe_to_json.py      # export trained weights for the browser

data/
  metadata.csv                    # one row per source track (committed)
  clips.csv                       # one row per clip (committed)
  embeddings.csv                  # one row per embedding (committed)
  raw/                            # downloaded audio (gitignored)
  clips/, features/, embeddings/  # processed artifacts (gitignored)

models/                           # trained checkpoints (gitignored)
live_vs_studio_classifier_plan.md # original project plan
falsePositiveInvestigation.md     # investigation plan for the 64 hardest errors
parallelNextSteps.md              # roadmap for shipping v1 + full fine-tune
webDemoPlan.md                    # this demo's design doc
```

## Running the web demo locally

The web demo is plain static files — no build step.

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

The AST model (~87 MB) downloads from HuggingFace on first visit and is cached by the browser thereafter.

## Privacy

Your audio never leaves your device. The only network requests the demo makes are:

- One-time download of the AST backbone from HuggingFace's CDN (cached forever afterward)
- One-time download of `transformers.js` and `onnxruntime-web` from jsDelivr
- The static page assets from GitHub Pages

You can verify this in your browser's DevTools Network tab. There is no server, no analytics, no telemetry.

## Acknowledgements

- [MIT CSAIL](https://huggingface.co/MIT) for the [Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- [Xenova / HuggingFace](https://huggingface.co/Xenova) for the pre-converted ONNX variants
- [transformers.js](https://huggingface.co/docs/transformers.js) for browser inference machinery
- The [Internet Archive Live Music Archive](https://archive.org/details/etree) for ~10,000 live clips
- The [Free Music Archive](https://github.com/mdeff/fma) for the studio counterpart

## License

MIT. See [LICENSE](LICENSE).
