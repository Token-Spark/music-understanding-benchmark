#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Majority / Prior Baseline for the Music Understanding Benchmark (MUB).

This baseline uses the benchmark's own answer distribution and has no audio
access. It predicts the same answer for every recording:

  - BPM: middle sorted BPM across all entries
  - Key: most frequent key
  - Time signature: most frequent time signature
  - Structure: no intervals (returns None)

This is a descriptive test-distribution prior, not a trained text-only model
or a theoretical lower bound on audio-informed performance.

Usage:
    python baselines/majority_baseline.py \
        --benchmark data/benchmark.json \
        --output results/majority_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def compute_majority(entries: list[dict]) -> dict:
    """Compute the majority (modal) prediction for each task."""
    bpms = [e["gold"]["bpm"] for e in entries if e["gold"].get("bpm")]
    keys = [e["gold"]["tonality"]["label"]
            for e in entries if e["gold"].get("tonality", {}).get("label")]
    tss = [e["gold"]["time_signature"]
           for e in entries if e["gold"].get("time_signature")]

    bpm_median = sorted(bpms)[len(bpms) // 2] if bpms else None
    key_mode = Counter(keys).most_common(1)[0][0] if keys else None
    ts_mode = Counter(tss).most_common(1)[0][0] if tss else None

    return {
        "bpm": bpm_median,
        "key": key_mode,
        "time_signature": ts_mode,
        "structure": None,
    }


def generate_predictions(entries: list[dict], majority: dict) -> list[dict]:
    """Generate identical majority predictions for every entry."""
    preds = []
    for entry in entries:
        preds.append({
            "id": entry["id"],
            "bpm": majority["bpm"],
            "key": majority["key"],
            "time_signature": majority["time_signature"],
            "structure": None,
        })
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmark.json")
    ap.add_argument("--output", default="results/majority_baseline.json")
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    entries = benchmark["entries"]
    majority = compute_majority(entries)

    print(f"[majority] Modal predictions:")
    print(f"  BPM: {majority['bpm']}")
    print(f"  Key: {majority['key']}")
    print(f"  Time signature: {majority['time_signature']}")
    print(f"  Structure: (none — majority baseline cannot segment)")

    preds = generate_predictions(entries, majority)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": "majority_baseline", "predictions": preds}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[majority] {len(preds)} predictions → {args.output}")


if __name__ == "__main__":
    main()
