#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate model predictions against the Music Understanding Benchmark (MUB).

Reads:
  - benchmark.json (gold answers, anonymized)
  - predictions.json (model outputs)

Produces:
  - report.json (aggregate + per-entry metrics)

The predictions file format is a list of objects:
  [
    {
      "id": "MUB-00001",
      "bpm": 120,
      "key": "C major",           # or {"pitch_class": 0, "mode": "major"}
      "time_signature": "4/4",
      "structure": [
        {"start": 0.0, "end": 15.0, "label": "intro"},
        ...
      ]
    },
    ...
  ]

Missing fields are treated as no-prediction (not a hit).
Fields may be omitted entirely for tasks a model doesn't support.

Usage:
    python src/evaluate.py \
        --benchmark data/benchmark.json \
        --predictions results/model_xxx.json \
        --output results/report_model_xxx.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Add src to path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import evaluate_predictions, parse_key


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_predictions(raw: list[dict]) -> list[dict]:
    """
    Normalize prediction format variations:
      - key as string → parse to (pitchClass, mode)
      - key as dict → keep
      - BPM as string → float
      - structure missing → None
    """
    normalized = []
    for item in raw:
        entry = {"id": item.get("id")}

        # BPM
        bpm = item.get("bpm")
        if bpm is not None:
            try:
                entry["bpm"] = float(bpm)
            except (ValueError, TypeError):
                entry["bpm"] = None

        # Key
        key = item.get("key")
        if key is not None:
            if isinstance(key, dict):
                pc = key.get("pitch_class", key.get("pitchClass"))
                mode = key.get("mode")
                if pc is not None and mode is not None:
                    entry["key"] = (int(pc), mode)
                else:
                    entry["key"] = None
            elif isinstance(key, str):
                entry["key"] = parse_key(key)
            elif isinstance(key, list) and len(key) >= 2:
                entry["key"] = (int(key[0]), str(key[1]))
            else:
                entry["key"] = None
        else:
            entry["key"] = None

        # Time signature
        ts = item.get("time_signature") or item.get("timesig") or item.get("timeSignature")
        entry["time_signature"] = str(ts) if ts else None

        # Structure
        struct = item.get("structure") or item.get("segments")
        if struct and isinstance(struct, list):
            entry["structure"] = [
                {
                    "start": float(s.get("start", s.get("begin", 0))),
                    "end": float(s.get("end", s.get("stop", 0))),
                    "label": s.get("label", s.get("name", "")),
                }
                for s in struct
            ]
        else:
            entry["structure"] = None

        normalized.append(entry)
    return normalized


def format_report(results: dict, model_name: str) -> dict:
    """Format results into a human-readable report structure."""
    return {
        "model": model_name,
        "benchmark_version": "1.0.0",
        "summary": {
            "bpm_accuracy": results["bpm"]["accuracy"],
            "bpm_median_relative_error": results["bpm"]["median_relative_error"],
            "tonality_accuracy": results["tonality"]["accuracy"],
            "time_signature_accuracy": results["time_signature"]["accuracy"],
            "structure_mean_f1": results["structure"]["mean_f1"],
            "structure_mean_precision": results["structure"]["mean_precision"],
            "structure_mean_recall": results["structure"]["mean_recall"],
        },
        "detail": {
            "bpm": results["bpm"],
            "tonality": results["tonality"],
            "time_signature": results["time_signature"],
            "structure": {k: v for k, v in results["structure"].items()
                          if k != "per_entry"},
        },
        "per_entry": results.get("per_entry", []),
    }


def print_summary(report: dict) -> None:
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"  Music Understanding Benchmark — Evaluation Report")
    print(f"  Model: {report['model']}")
    print(f"{'='*60}")
    print(f"  Entries evaluated:     {report['detail']['bpm']['total']}")
    print(f"")
    print(f"  BPM Estimation")
    print(f"    Accuracy (±4%):      {s['bpm_accuracy']:.1%}")
    print(f"    Median rel. error:   {s['bpm_median_relative_error']:.1%}")
    print(f"")
    print(f"  Tonality Identification")
    print(f"    Accuracy:            {s['tonality_accuracy']:.1%}")
    print(f"")
    print(f"  Time Signature Detection")
    print(f"    Accuracy:            {s['time_signature_accuracy']:.1%}")
    print(f"")
    print(f"  Structure Segmentation")
    print(f"    Mean F1:             {s['structure_mean_f1']:.3f}")
    print(f"    Mean Precision:      {s['structure_mean_precision']:.3f}")
    print(f"    Mean Recall:         {s['structure_mean_recall']:.3f}")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmark.json",
                    help="Path to benchmark.json")
    ap.add_argument("--predictions", required=True,
                    help="Path to model predictions JSON")
    ap.add_argument("--output", default=None,
                    help="Output report path (default: stdout + <predictions_dir>/report.json)")
    ap.add_argument("--model-name", default="unknown",
                    help="Name of the model being evaluated")
    args = ap.parse_args()

    benchmark = load_json(args.benchmark)
    gold_entries = benchmark["entries"]

    raw_preds = load_json(args.predictions)
    if isinstance(raw_preds, dict) and "predictions" in raw_preds:
        raw_preds = raw_preds["predictions"]

    predictions = normalize_predictions(raw_preds)
    print(f"[evaluate] {len(predictions)} predictions, {len(gold_entries)} gold entries",
          file=sys.stderr)

    results = evaluate_predictions(predictions, gold_entries)
    report = format_report(results, args.model_name)

    print_summary(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[evaluate] Report → {args.output}")
    else:
        print(f"\n[evaluate] (No output file specified; use --output to save.)")


if __name__ == "__main__":
    main()
