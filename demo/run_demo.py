#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-command demo: extract a 5-entry subset from the real benchmark data,
generate prompts, run the majority-vote baseline, and evaluate.

No audio files or external dependencies (beyond numpy) are needed.

Usage:
    python demo/run_demo.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = Path(__file__).resolve().parent
DEMO_BENCHMARK = DEMO_DIR / "demo_benchmark.json"
DEMO_PROMPTS = DEMO_DIR / "demo_prompts"
DEMO_RESULTS = DEMO_DIR / "demo_results"


def run(cmd: list[str], desc: str) -> int:
    """Run a subprocess, streaming output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print(f"\n  [FAILED] exit code {rc}")
    return rc


def main():
    python = sys.executable or "python"

    # Step 1: Generate demo benchmark subset
    rc = run([python, str(DEMO_DIR / "generate_demo_subset.py")],
             "Step 1/4: Generate 5-entry demo subset from real benchmark")
    if rc != 0:
        return rc

    # Step 2: Generate prompts from the demo benchmark
    rc = run([
        python, str(ROOT / "src" / "prepare_prompts.py"),
        "--benchmark", str(DEMO_BENCHMARK),
        "--output-dir", str(DEMO_PROMPTS),
    ], "Step 2/4: Generate evaluation prompts")
    if rc != 0:
        return rc

    # Step 3: Run majority-vote baseline (no audio needed)
    os.makedirs(str(DEMO_RESULTS), exist_ok=True)
    rc = run([
        python, str(ROOT / "baselines" / "majority_baseline.py"),
        "--benchmark", str(DEMO_BENCHMARK),
        "--output", str(DEMO_RESULTS / "majority_baseline.json"),
    ], "Step 3/4: Run majority-vote baseline (no audio needed)")
    if rc != 0:
        return rc

    # Step 4: Evaluate
    rc = run([
        python, str(ROOT / "src" / "evaluate.py"),
        "--benchmark", str(DEMO_BENCHMARK),
        "--predictions", str(DEMO_RESULTS / "majority_baseline.json"),
        "--output", str(DEMO_RESULTS / "report_majority.json"),
        "--model-name", "majority_baseline_demo",
    ], "Step 4/4: Evaluate predictions against gold labels")
    if rc != 0:
        return rc

    # Summary
    print(f"\n{'='*60}")
    print("  Demo complete!")
    print(f"{'='*60}")
    print(f"  Demo benchmark:  {DEMO_BENCHMARK}")
    print(f"  Prompts:         {DEMO_PROMPTS}/")
    print(f"  Predictions:     {DEMO_RESULTS}/majority_baseline.json")
    print(f"  Evaluation:       {DEMO_RESULTS}/report_majority.json")
    print()
    print("  To run the DSP baseline (requires audio + ffmpeg):")
    print(f"    python baselines/dsp_baseline.py \\")
    print(f"      --benchmark {DEMO_BENCHMARK} \\")
    print(f"      --audio-dir /path/to/audio \\")
    print(f"      --output {DEMO_RESULTS}/dsp_baseline.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
