#!/usr/bin/env python3
"""
predict.py — local command-line version of the LiveAudioClassifier web demo.

Same model, same pipeline, same accuracy as the web demo at
https://ziipo.github.io/LiveAudioClassifier/ — runs locally if you'd rather
not (or can't) use the browser version. Useful for batch processing folders
of audio, or for working offline after the model is cached.

Pipeline (per audio file):
  1. Resample to 16 kHz mono
  2. Extract three 30-second windows at 25/50/75% of the usable span
  3. Compute AST log-mel features for each window
  4. AST forward pass -> 768-dim pooler_output
  5. Linear probe -> (p_studio, p_live)
  6. Majority vote across windows

Usage:
  python scripts/predict.py path/to/song.mp3
  python scripts/predict.py *.mp3
  python scripts/predict.py --json song.mp3       # machine-readable output
  python scripts/predict.py --device cpu song.mp3 # force CPU instead of MPS/CUDA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTModel

# The AST checkpoint we use. Same one the browser demo loads (the int8 ONNX
# variant lives at Xenova/<same id>; here we use HuggingFace's reference
# PyTorch weights since this is a local CLI, not the browser).
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Clip-windowing constants — must match web/audio.js for prediction parity.
SR = 16000
CLIP_SEC = 30
CLIP_LEN = SR * CLIP_SEC
HEAD_SKIP = 10
TAIL_SKIP = 10
POSITIONS = [0.25, 0.5, 0.75]

DEFAULT_PROBE = Path(__file__).resolve().parent.parent / "web" / "probe-weights.json"


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_audio(path: Path) -> np.ndarray:
    """Decode any common audio file as 16 kHz mono float32."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


def extract_windows(y: np.ndarray) -> tuple[list[np.ndarray], str]:
    """Return up to three 30s windows from 16 kHz mono audio.

    Mirrors web/audio.js:extractWindows() so the CLI and browser produce
    bit-identical input tensors for the same source audio.
    """
    total_sec = len(y) / SR
    if len(y) < CLIP_LEN:
        if len(y) < SR * 5:
            return [], "audio is under 5 seconds — too short to classify"
        padded = np.zeros(CLIP_LEN, dtype=np.float32)
        padded[: len(y)] = y
        return [padded], f"padded {total_sec:.1f}s to 30s"

    if total_sec < CLIP_SEC * 2:
        start = max(0, (len(y) - CLIP_LEN) // 2)
        return [y[start:start + CLIP_LEN]], (
            f"single middle window from {total_sec:.1f}s clip"
        )

    usable = total_sec - HEAD_SKIP - TAIL_SKIP
    if usable < CLIP_SEC:
        start = max(0, (len(y) - CLIP_LEN) // 2)
        return [y[start:start + CLIP_LEN]], (
            f"single window from {total_sec:.1f}s clip"
        )

    windows = []
    for p in POSITIONS:
        center_sec = HEAD_SKIP + usable * p
        start = int((center_sec - CLIP_SEC / 2) * SR)
        start = max(0, min(start, len(y) - CLIP_LEN))
        windows.append(y[start:start + CLIP_LEN])
    return windows, f"three windows at 25/50/75% of usable span ({total_sec:.1f}s clip)"


class Probe(nn.Module):
    """Linear-probe head, loaded from the same JSON the browser demo uses.

    Keeping the weights in JSON (rather than a .pt) means the CLI and the web
    demo can share a single artifact — one source of truth.
    """

    def __init__(self, weights_path: Path):
        super().__init__()
        cfg = json.loads(weights_path.read_text())
        self.fc = nn.Linear(cfg["in_dim"], cfg["num_classes"])
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor(cfg["weight"], dtype=torch.float32))
            self.fc.bias.copy_(torch.tensor(cfg["bias"], dtype=torch.float32))
        self.id2label = cfg["id2label"]
        self.version = cfg.get("version", "?")
        self.metrics = {
            "trained_val_acc": cfg.get("trained_val_acc"),
            "trained_test_acc": cfg.get("trained_test_acc"),
            "source_quality_test_acc": cfg.get("source_quality_test_acc"),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def predict_one(path: Path, fe: ASTFeatureExtractor, model: ASTModel,
                probe: Probe, device: torch.device) -> dict:
    y = load_audio(path)
    windows, info = extract_windows(y)
    if not windows:
        return {"file": str(path), "error": info, "windows": []}

    per_window = []
    for w in windows:
        # Feature extractor produces shape (1, 1024, 128); .to(device) is fine.
        features = fe(w, sampling_rate=SR, return_tensors="pt")
        x = features["input_values"].to(device)
        with torch.no_grad():
            pooled = model(x).pooler_output  # (1, 768)
            logits = probe(pooled)            # (1, 2)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # label2id was {"studio": 0, "live": 1}
        per_window.append({
            "p_studio": float(probs[0]),
            "p_live": float(probs[1]),
            "label": "live" if probs[1] >= probs[0] else "studio",
        })

    # Majority vote + average confidence on the winning class.
    n_live = sum(1 for w in per_window if w["label"] == "live")
    final_label = "live" if n_live >= (len(per_window) + 1) // 2 else "studio"
    avg_p_live = sum(w["p_live"] for w in per_window) / len(per_window)
    final_conf = avg_p_live if final_label == "live" else 1 - avg_p_live

    return {
        "file": str(path),
        "info": info,
        "label": final_label,
        "confidence": float(final_conf),
        "windows": per_window,
        "agreement": f"{n_live}/{len(per_window)} live",
    }


def render_text(result: dict) -> str:
    if "error" in result:
        return f"{result['file']}\n  ERROR: {result['error']}\n"
    lines = [
        result["file"],
        f"  prediction: {result['label']}",
        f"  confidence: {result['confidence']:.3f}  ({result['agreement']})",
        f"  per-window p(live): "
        + " ".join(f"{w['p_live']:.3f}" for w in result["windows"]),
        f"  ({result['info']})",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path,
                   help="audio file(s) to classify (mp3, wav, flac, m4a, ogg …)")
    p.add_argument("--probe-weights", type=Path, default=DEFAULT_PROBE,
                   help=f"path to probe weights JSON (default: {DEFAULT_PROBE.name} "
                        "in the web/ folder)")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"],
                   help="force a specific torch device (default: auto-detect)")
    p.add_argument("--json", action="store_true",
                   help="print one JSON object per file on stdout instead of text")
    args = p.parse_args()

    if not args.probe_weights.exists():
        sys.exit(f"probe weights not found: {args.probe_weights}")

    device = pick_device(args.device)
    if not args.json:
        print(f"Loading {MODEL_ID} on {device} …", file=sys.stderr)
    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTModel.from_pretrained(MODEL_ID).to(device).eval()
    probe = Probe(args.probe_weights).to(device).eval()
    if not args.json:
        print(f"Probe {probe.version} "
              f"(trained test acc: {probe.metrics['trained_test_acc']:.3f})",
              file=sys.stderr)

    for path in args.files:
        if not path.exists():
            print(json.dumps({"file": str(path), "error": "file not found"})
                  if args.json else f"{path}\n  ERROR: file not found\n")
            continue
        try:
            result = predict_one(path, fe, model, probe, device)
        except Exception as e:
            result = {"file": str(path), "error": str(e)}
        if args.json:
            print(json.dumps(result))
        else:
            print(render_text(result))


if __name__ == "__main__":
    main()
