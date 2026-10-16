#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Objective DSP Baseline for the Music Understanding Benchmark (MUB).

Estimates BPM, key, and time signature directly from audio waveforms
using classical DSP methods (no machine learning, no semantics).

This baseline provides simple waveform-based reference predictions. Agreement
with existing labels is a consistency observation, not proof of annotation
provenance or correctness. It implements no structural segmentation and is
neither a state-of-the-art system nor a theoretical performance bound.

Methods (numpy and ffmpeg, no heavy ML dependencies):
  - Decode: ffmpeg → 22050 Hz mono PCM
  - BPM: spectral flux → autocorrelation over 60–180 BPM
  - Key: aggregate pitch-class energy → major/minor key profiles
  - Time signature: test periodicities at 2, 3, and 4 beats; denominator 4

Usage:
    python baselines/dsp_baseline.py \
        --benchmark data/benchmark.json \
        --audio-dir /path/to/audio/files \
        --output results/dsp_baseline.json

Note: Audio files are NOT distributed with the benchmark. Participants
must supply their own audio. The audio_hash field in benchmark.json
can be used to verify file identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
HOP = 512
N_FFT = 2048

# Krumhansl-Kessler key profiles (classic perceptual key-finding weights)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                          5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                          4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

PITCH_CLASS = {name: i for i, name in enumerate(PITCH_NAMES)}
PITCH_CLASS.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


# ── Audio decoding ────────────────────────────────────────────────────

def decode_audio(path: str, max_seconds: float = 600.0) -> np.ndarray:
    """Decode any audio file to mono float32 PCM via ffmpeg."""
    cmd = [
        FFMPEG, "-v", "quiet", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-t", str(max_seconds), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# ── Spectral analysis ─────────────────────────────────────────────────

def compute_spectrum(y: np.ndarray) -> np.ndarray:
    """STFT → magnitude spectrum."""
    win = np.hanning(N_FFT).astype(np.float32)
    n_frames = 1 + max(0, (len(y) - N_FFT) // HOP)
    if n_frames < 2:
        raise ValueError("Audio too short for spectral analysis")
    shape = (n_frames, N_FFT)
    strides = (y.strides[0] * HOP, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    return np.abs(np.fft.rfft(frames * win, axis=1))


def spectral_flux(spec: np.ndarray) -> np.ndarray:
    """Spectral flux: sum of half-wave-rectified frame-to-frame differences."""
    diff = np.diff(spec, axis=0)
    return np.maximum(diff, 0).sum(axis=1)


# ── BPM estimation ───────────────────────────────────────────────────

def estimate_bpm(flux: np.ndarray, sr: int = SAMPLE_RATE, hop: int = HOP):
    """Autocorrelation-based BPM estimation, folded to 60–180 BPM range."""
    fps = sr / hop
    x = flux - flux.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]

    lo = int(round(fps * 60.0 / 180.0))  # 180 BPM → shortest period
    hi = int(round(fps * 60.0 / 60.0))   # 60 BPM  → longest period
    hi = min(hi, len(ac) - 1)
    if lo >= hi:
        return None, 0.0

    window = ac[lo:hi]
    if window.size == 0:
        return None, 0.0
    lag = lo + int(np.argmax(window))
    strength = float(ac[lag])
    bpm = 60.0 / (lag * hop / sr)
    return round(bpm, 1), round(strength, 4)


# ── Time signature estimation ─────────────────────────────────────────

def estimate_timesig(flux: np.ndarray, bpm: float | None,
                     sr: int = SAMPLE_RATE, hop: int = HOP):
    """Estimate time signature from beat periodicity."""
    if not bpm:
        return None, 0.0
    fps = sr / hop
    beat_period = 60.0 / bpm * fps
    x = flux - flux.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]

    scores = {}
    for beats in (2, 3, 4):
        lag = int(round(beat_period * beats))
        if lag < len(ac) - 4:
            lo, hi = max(0, lag - 2), min(len(ac), lag + 3)
            scores[beats] = float(np.max(ac[lo:hi]))
    if not scores:
        return None, 0.0
    best = max(scores, key=scores.get)
    return f"{best}/4", round(scores[best], 4)


# ── Key estimation ───────────────────────────────────────────────────

def estimate_key(y: np.ndarray, spec: np.ndarray, sr: int = SAMPLE_RATE):
    """Chroma → Krumhansl-Schmuckler key correlation."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    valid = (freqs > 55) & (freqs < 5000)
    idx = np.where(valid)[0]
    if idx.size == 0:
        return None, None, None, 0.0

    midi = 69 + 12 * np.log2(freqs[idx] / 440.0)
    pitch_class = np.round(midi).astype(int) % 12
    weights = spec[:, idx].mean(axis=0)
    chroma = np.zeros(12)
    np.add.at(chroma, pitch_class, weights)
    total = chroma.sum()
    if total <= 0:
        return None, None, None, 0.0
    chroma = chroma / total

    best = (None, None, 0.0)
    for pc in range(12):
        for name, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = np.roll(profile, pc)
            corr = float(np.corrcoef(chroma, rotated)[0, 1])
            if not np.isnan(corr) and corr > best[2]:
                best = (PITCH_NAMES[pc], name, corr)

    label, mode, corr = best
    pc_idx = PITCH_NAMES.index(label)
    return f"{label} {mode}", pc_idx, mode, round(corr, 4)


# ── Full analysis ─────────────────────────────────────────────────────

def analyze_one(audio_path: str) -> dict:
    """Run full DSP analysis on a single audio file."""
    result = {"audio_path": audio_path, "ok": False}
    try:
        y = decode_audio(audio_path)
    except Exception as exc:
        result["error"] = f"decode_failed: {exc}"
        return result

    if len(y) < SAMPLE_RATE * 10:
        result["error"] = "too_short"
        return result

    y = y - y.mean()
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    try:
        spec = compute_spectrum(y)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    flux = spectral_flux(spec)
    bpm, bpm_strength = estimate_bpm(flux)
    ts, ts_strength = estimate_timesig(flux, bpm)
    key_label, key_pc, key_mode, key_corr = estimate_key(y, spec)

    result.update({
        "ok": True,
        "bpm": bpm,
        "bpm_strength": bpm_strength,
        "time_signature": ts,
        "time_signature_strength": ts_strength,
        "key": key_label,
        "key_pitch_class": key_pc,
        "key_mode": key_mode,
        "key_correlation": key_corr,
    })
    return result


# ── Main ──────────────────────────────────────────────────────────────

def build_hash_index(audio_dir: str) -> dict[str, str]:
    """Pre-scan audio_dir once, returning {hash16: filepath} for all audio files."""
    index = {}
    for root, _, files in os.walk(audio_dir):
        for fname in files:
            if fname.endswith((".m4a", ".mp3", ".wav", ".flac", ".ogg")):
                fpath = os.path.join(root, fname)
                try:
                    h = hashlib.sha256()
                    with open(fpath, "rb") as f:
                        h.update(f.read(1_048_576))
                    index[h.hexdigest()[:16]] = fpath
                except Exception:
                    continue
    return index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmark.json")
    ap.add_argument("--audio-dir", required=True,
                    help="Directory containing audio files (will be searched by hash)")
    ap.add_argument("--output", default="results/dsp_baseline.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max entries to process (0 = all)")
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    entries = benchmark["entries"]
    if args.limit:
        entries = entries[:args.limit]

    print(f"[dsp_baseline] Building hash index of audio files in {args.audio_dir}...",
          flush=True)
    hash_index = build_hash_index(args.audio_dir)
    print(f"[dsp_baseline] Indexed {len(hash_index)} audio files", flush=True)

    predictions = []
    for i, entry in enumerate(entries, 1):
        eid = entry["id"]
        audio_hash = entry.get("audio_hash", "")

        audio_path = hash_index.get(audio_hash)
        if not audio_path:
            predictions.append({"id": eid, "bpm": None, "key": None,
                                "time_signature": None, "structure": None})
            print(f"  [{i}/{len(entries)}] {eid}: audio not found", flush=True)
            continue

        r = analyze_one(audio_path)
        if r["ok"]:
            predictions.append({
                "id": eid,
                "bpm": r["bpm"],
                "key": r["key"],
                "time_signature": r["time_signature"],
                "structure": None,
            })
        else:
            predictions.append({"id": eid, "bpm": None, "key": None,
                                "time_signature": None, "structure": None})

        status = "✓" if r["ok"] else "✗"
        print(f"  [{i}/{len(entries)}] {eid}: {status}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": "dsp_baseline", "predictions": predictions}, f,
                  ensure_ascii=False, indent=2)
    ok = sum(1 for p in predictions if p.get("bpm") is not None)
    print(f"\n[dsp_baseline] {ok}/{len(predictions)} succeeded → {args.output}")


if __name__ == "__main__":
    main()
