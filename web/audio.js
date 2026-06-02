// audio.js — turn a user-supplied audio file into 30-second 16 kHz mono
// windows ready for AST.
//
// Pipeline (all in the browser, no network):
//   1. File -> ArrayBuffer
//   2. AudioContext.decodeAudioData -> AudioBuffer (browser handles MP3/WAV/FLAC)
//   3. Downmix to mono (average channels)
//   4. Resample to 16 kHz via OfflineAudioContext
//   5. Extract three 30s windows at 25/50/75% positions, mirroring
//      scripts/04_extract_clips.py:live_clip_windows()
//
// Why OfflineAudioContext for resampling: it's the Web Audio API's standard
// resampler, available in every browser since ~2018, and gives ~librosa-quality
// output for typical music audio. Hand-rolling a polyphase resampler in JS is
// unnecessary precision-chasing.

const TARGET_SR = 16000;
const CLIP_SEC = 30;
const HEAD_SKIP = 10;
const TAIL_SKIP = 10;
const POSITIONS = [0.25, 0.5, 0.75];

/**
 * Decode an audio File or Blob into 16 kHz mono samples.
 * @param {File|Blob} file
 * @returns {Promise<Float32Array>}
 */
export async function decodeTo16kMono(file) {
  const arrayBuf = await file.arrayBuffer();

  // Phase 1: decode in the browser's native sample rate. Some browsers (Safari)
  // refuse to decode if the AudioContext sample rate doesn't match the file's,
  // so we decode at native first, then resample.
  const decoder = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await decoder.decodeAudioData(arrayBuf.slice(0));
  } finally {
    // Free the decode context promptly — we don't need it anymore.
    decoder.close?.();
  }

  // Phase 2: downmix to mono by averaging channels.
  const monoBuf = downmixToMono(decoded);

  // Phase 3: resample mono buffer to 16 kHz via OfflineAudioContext.
  return await resampleTo16k(monoBuf, decoded.sampleRate);
}

/**
 * Average all channels of an AudioBuffer into a single Float32Array.
 */
function downmixToMono(audioBuffer) {
  const nCh = audioBuffer.numberOfChannels;
  const len = audioBuffer.length;
  if (nCh === 1) {
    // Copy so we own the buffer (avoids holding the AudioBuffer alive).
    return audioBuffer.getChannelData(0).slice();
  }
  const mono = new Float32Array(len);
  const channels = [];
  for (let c = 0; c < nCh; c++) channels.push(audioBuffer.getChannelData(c));
  for (let i = 0; i < len; i++) {
    let sum = 0;
    for (let c = 0; c < nCh; c++) sum += channels[c][i];
    mono[i] = sum / nCh;
  }
  return mono;
}

/**
 * Resample a mono Float32Array from `inputSr` to TARGET_SR using
 * OfflineAudioContext.
 */
async function resampleTo16k(monoSamples, inputSr) {
  if (inputSr === TARGET_SR) return monoSamples.slice();
  // Allocate a tiny AudioBuffer at the source rate to feed the offline graph.
  const ctx = new OfflineAudioContext({
    numberOfChannels: 1,
    length: Math.ceil(monoSamples.length * TARGET_SR / inputSr),
    sampleRate: TARGET_SR,
  });
  const src = ctx.createBufferSource();
  const srcBuf = new AudioBuffer({
    numberOfChannels: 1,
    length: monoSamples.length,
    sampleRate: inputSr,
  });
  srcBuf.copyToChannel(monoSamples, 0);
  src.buffer = srcBuf;
  src.connect(ctx.destination);
  src.start(0);
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0).slice();
}

/**
 * Extract up to three 30-second windows from 16 kHz mono samples, mirroring
 * the Python live_clip_windows() (scripts/04_extract_clips.py).
 *
 * For audio between 30-60s: pads or returns no windows depending on length.
 * For audio < 30s: pads with zeros to 30s. Matches studio_clip() in the
 * Python pipeline for short clips.
 *
 * @param {Float32Array} samples16k
 * @returns {{windows: Float32Array[], info: string}}
 */
export function extractWindows(samples16k) {
  const clipLen = TARGET_SR * CLIP_SEC;
  const totalSec = samples16k.length / TARGET_SR;

  // Very short: pad to 30s and return one window. This matches studio_clip()
  // behavior on FMA tracks shorter than 30s.
  if (samples16k.length < clipLen) {
    if (samples16k.length < TARGET_SR * 5) {
      return { windows: [], info: "audio is under 5 seconds — too short to classify" };
    }
    const padded = new Float32Array(clipLen);
    padded.set(samples16k);
    return { windows: [padded], info: `padded ${totalSec.toFixed(1)}s to 30s` };
  }

  // Long-enough audio: extract three 30s windows at 25/50/75% of the usable
  // span, skipping HEAD_SKIP at the start and TAIL_SKIP at the end.
  if (totalSec < CLIP_SEC * 2) {
    // Between 30-60s: take a single middle window (don't try for 3, they'd overlap heavily)
    const start = Math.max(0, Math.floor((samples16k.length - clipLen) / 2));
    return {
      windows: [samples16k.slice(start, start + clipLen)],
      info: `single middle window from ${totalSec.toFixed(1)}s clip`,
    };
  }

  const usable = totalSec - HEAD_SKIP - TAIL_SKIP;
  if (usable < CLIP_SEC) {
    // Edge case: 60-70s clips can fall here. Fall back to single middle window.
    const start = Math.max(0, Math.floor((samples16k.length - clipLen) / 2));
    return {
      windows: [samples16k.slice(start, start + clipLen)],
      info: `single window from ${totalSec.toFixed(1)}s clip`,
    };
  }

  const windows = POSITIONS.map((p) => {
    const centerSec = HEAD_SKIP + usable * p;
    let start = Math.floor((centerSec - CLIP_SEC / 2) * TARGET_SR);
    start = Math.max(0, Math.min(start, samples16k.length - clipLen));
    return samples16k.slice(start, start + clipLen);
  });
  return {
    windows,
    info: `three windows at 25/50/75% of usable span (${totalSec.toFixed(1)}s clip)`,
  };
}
