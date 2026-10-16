#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a small demo subset from the real MUB benchmark data.

Extracts 5 diverse entries from data/benchmark.json so users can test the
full pipeline (prompts -> majority baseline -> evaluate) without needing the
audio files.  The DSP baseline requires audio and is optional.

Usage:
    python demo/generate_demo_subset.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "benchmark.json"
DEMO_BENCHMARK = ROOT / "demo" / "demo_benchmark.json"

# 5 entries chosen for diversity from the real benchmark data
DEMO_IDS = ["MUB-00001", "MUB-00016", "MUB-00006", "MUB-00013", "MUB-00002"]


def main():
    with open(SOURCE, encoding="utf-8") as f:
        full = json.load(f)

    entries = {e["id"]: e for e in full["entries"]}
    subset = [entries[i] for i in DEMO_IDS if i in entries]

    demo = {
        "name": "MUB Demo Benchmark",
        "version": "demo-1.0",
        "description": (
            "A 5-entry subset of the full MUB benchmark for pipeline "
            "demonstration. Not a substitute for the full 500-piece benchmark."
        ),
        "generated_at": full.get("generated_at", ""),
        "source": "data/benchmark.json",
        "tasks": full["tasks"],
        "entries": subset,
    }

    DEMO_BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    with open(DEMO_BENCHMARK, "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=2)

    print(f"[demo] {len(subset)} entries -> {DEMO_BENCHMARK}")
    for e in subset:
        g = e["gold"]
        print(f"  {e['id']}: BPM={g['bpm']}, "
              f"key={g['tonality']['label']}, "
              f"ts={g['time_signature']}, "
              f"segs={e['segment_count']}")


if __name__ == "__main__":
    main()
