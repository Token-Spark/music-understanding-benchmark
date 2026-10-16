#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare evaluation prompts for the Music Understanding Benchmark (MUB).

This script reads the benchmark JSON and produces a set of task-specific
prompt files that can be fed to any multimodal LLM. Each prompt asks the
model to perform one of four tasks on an audio input:

  1. BPM estimation — "What is the tempo (BPM) of this music?"
  2. Tonality identification — "What is the key of this music?"
  3. Time signature detection — "What is the time signature?"
  4. Structure segmentation — "Segment this music into structural parts."

The audio is referenced by an opaque hash — the model never sees the title,
artist, or any identifying information. Participants must supply their own
audio files matching the hashes in the benchmark.

Usage:
    python src/prepare_prompts.py \
        --benchmark data/benchmark.json \
        --output-dir data/prompts/
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Shared constraint prefix prepended to every task prompt.
# These rules are essential to the benchmark's validity: without them, a model
# with internet access could use audio fingerprinting, music recognition APIs,
# or web search to identify the recording and retrieve its metadata — bypassing
# the anonymization entirely and turning the task into a lookup rather than a
# genuine audio-understanding test.
CONSTRAINT_PREFIX = (
    "CONSTRAINTS — Read before answering:\n"
    "1. Base your answer solely on what you hear in the provided audio.\n"
    "2. Do NOT use web search, internet access, or any external tool to identify the recording.\n"
    "3. Do NOT attempt to use audio fingerprinting, music recognition services, "
    "or any external database lookup.\n"
    "4. Do NOT use any prior knowledge about what this recording might be — "
    "the piece has been deliberately anonymized.\n"
    "5. Respond based only on your own audio perception and musical analysis.\n"
    "\n"
)

TASK_PROMPTS = {
    "bpm": (
        CONSTRAINT_PREFIX
        + "You are listening to a piece of music.\n"
        "Estimate the tempo in beats per minute (BPM).\n\n"
        "Respond with a single integer between 40 and 220.\n"
        "Format: {\"bpm\": <integer>}"
    ),
    "tonality": (
        CONSTRAINT_PREFIX
        + "You are listening to a piece of music.\n"
        "Identify the musical key (tonality) of this piece.\n\n"
        "Respond with the tonic pitch class (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) "
        "and the mode (major or minor).\n"
        "Format: {\"key\": \"<pitch> <mode>\"}"
    ),
    "time_signature": (
        CONSTRAINT_PREFIX
        + "You are listening to a piece of music.\n"
        "Identify the time signature of this piece.\n\n"
        "Respond with the time signature as a fraction (e.g., \"4/4\", \"3/4\", \"6/8\").\n"
        "Format: {\"time_signature\": \"<numerator>/<denominator>\"}"
    ),
    "structure": (
        CONSTRAINT_PREFIX
        + "You are listening to a piece of music.\n"
        "Segment this music into its structural parts (e.g., intro, verse, chorus, "
        "bridge, outro, solo, etc.).\n\n"
        "For each segment, provide:\n"
        "- start: start time in seconds (float)\n"
        "- end: end time in seconds (float)\n"
        "- label: a short descriptive label for the segment\n\n"
        "Segments must be contiguous and cover the entire duration with no gaps or overlaps.\n"
        "Format: {\"structure\": [{\"start\": 0.0, \"end\": 15.0, \"label\": \"intro\"}, ...]}"
    ),
}


def prepare_prompts(benchmark_path: str, output_dir: str) -> None:
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = data["entries"]
    os.makedirs(output_dir, exist_ok=True)

    # Write one combined prompt file per task
    for task_name, prompt_text in TASK_PROMPTS.items():
        task_file = os.path.join(output_dir, f"prompts_{task_name}.jsonl")
        with open(task_file, "w", encoding="utf-8") as f:
            for entry in entries:
                record = {
                    "id": entry["id"],
                    "audio_hash": entry["audio_hash"],
                    "duration_sec": entry["duration_sec"],
                    "prompt": prompt_text,
                    # Gold answers are NOT included in the prompt file.
                    # They live only in benchmark.json for scoring.
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[prepare_prompts] → {task_file} ({len(entries)} prompts)")

    # Write a combined multi-task prompt file
    combined_file = os.path.join(output_dir, "prompts_all.jsonl")
    with open(combined_file, "w", encoding="utf-8") as f:
        for entry in entries:
            record = {
                "id": entry["id"],
                "audio_hash": entry["audio_hash"],
                "duration_sec": entry["duration_sec"],
                "prompt": (
                    CONSTRAINT_PREFIX
                    + "You are listening to a piece of music.\n"
                    "Please analyze this music and provide the following:\n\n"
                    "1. Tempo (BPM): a single integer (40–220)\n"
                    "2. Key/tonality: tonic pitch class and mode (major/minor)\n"
                    "3. Time signature: as a fraction (e.g., 4/4)\n"
                    "4. Structure segmentation: list of {start, end, label} segments\n\n"
                    "Respond in JSON format:\n"
                    '{"bpm": <int>, "key": "<pitch> <mode>", '
                    '"time_signature": "<n>/<d>", '
                    '"structure": [{"start": <float>, "end": <float>, "label": "<str>"}, ...]}'
                ),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[prepare_prompts] → {combined_file} ({len(entries)} prompts)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmark.json")
    ap.add_argument("--output-dir", default="data/prompts")
    args = ap.parse_args()
    prepare_prompts(args.benchmark, args.output_dir)


if __name__ == "__main__":
    main()
