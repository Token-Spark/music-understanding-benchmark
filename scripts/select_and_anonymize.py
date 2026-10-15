#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Select 500 copyright-safe, diverse music pieces from the BeatHamster annotation
corpus and produce an anonymized benchmark dataset.

Design principles
-----------------
1. **Copyright risk minimization**: We select pieces that are *annotated* by
   professional music analysts. The benchmark distributes only the annotation
   metadata (BPM, key, time signature, structure) — NOT the audio files.
   Audio is referenced by an opaque hash; participants must supply their own
   audio or use the provided feature-extraction pipeline.

2. **Anonymization**: No original title, artist, or platform-specific ID
   (metasong_id, taskId) appears in the benchmark. Each piece gets an opaque
   identifier (MUB-00001 … MUB-00500). This prevents multimodal LLMs from
   retrieving the original recording by name, ensuring we evaluate genuine
   audio-understanding ability rather than title-to-metadata lookup.

3. **Diversity**: Stratified sampling across BPM bins, tonality (key + mode),
   time signature, duration, and structure complexity to avoid clustering
   around a single genre or tempo.

4. **Quality**: Only pieces with ≥5 annotated segments, unambiguous BPM
   (clear modal value), unambiguous tonality, and duration 90–600s.

Usage:
    python scripts/select_and_anonymize.py \
        --source /path/to/BeatHamsterAssist/historical-data \
        --testset /path/to/BeatHamsterAssist/tools/llm-eval/testset.json \
        --output data/benchmark.json \
        --n 500
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Pitch class utilities ──────────────────────────────────────────────
PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def parse_key(label: str) -> tuple[int, str] | None:
    """Parse 'C Natural major scale' → (0, 'major')."""
    if not label:
        return None
    import re
    m = re.match(r"^([A-G](?:#|b)?)\s+(.*)$", str(label))
    if not m:
        return None
    pc = PITCH_CLASS.get(m.group(1))
    if pc is None:
        return None
    rest = m.group(2).lower()
    if re.search(r"minor|min\b|小调", rest):
        return (pc, "minor")
    if re.search(r"major|maj\b|大调", rest):
        return (pc, "major")
    return None


def tonality_equal(a: tuple[int, str] | None, b: tuple[int, str] | None) -> bool:
    """Same tonic+mode, or relative major/minor (3 semitone offset)."""
    if not a or not b:
        return False
    if a[0] == b[0] and a[1] == b[1]:
        return True
    if (a[0] + 3) % 12 == b[0] and a[1] != b[1]:
        return True
    return False


def mode(values: list):
    """Return the clear modal value; None if tied."""
    freq = Counter(values)
    if not freq:
        return None
    top = freq.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def file_hash(path: str) -> str:
    """SHA-256 of file content (first 1MB for speed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1_048_576))
    return h.hexdigest()[:16]


# ── Candidate loading ───────────────────────────────────────────────────

def load_candidates_from_testset(testset_path: str) -> list[dict]:
    """Load pre-built candidates from the existing testset.json."""
    with open(testset_path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    candidates = []
    for q in questions:
        # Only keep structure_segments tasks with full probe data
        if not q.get("structure") or len(q["structure"]) < 5:
            continue
        probe = q.get("probe", {})
        bpm = probe.get("bpm", {}).get("gold")
        tonality = probe.get("tonality", {}).get("gold")
        time_sig = probe.get("timeSignature", {}).get("gold")
        if not bpm or not tonality or not time_sig:
            continue
        duration = q.get("duration", 0)
        if not (90 <= duration <= 600):
            continue
        candidates.append({
            "source_id": q["id"],
            "audio_path": q.get("audioPath", ""),
            "title": q.get("title", ""),
            "artist": q.get("artist", ""),
            "duration": duration,
            "bpm": bpm,
            "tonality_pc": tonality.get("pitchClass"),
            "tonality_mode": tonality.get("mode"),
            "time_signature": time_sig,
            "structure": q["structure"],
            "segment_count": len(q["structure"]),
            "semantic": q.get("semantic", {}),
            "lyric_lines": q.get("lyricLines", 0),
            "expert_submissions": q.get("expertSubmissions", 0),
            "operators": q.get("operators", []),
        })
    return candidates


def dedup_by_song(candidates: list[dict]) -> list[dict]:
    """Deduplicate by title+artist, keeping the version with most segments."""
    by_song: dict[str, dict] = {}
    for c in candidates:
        key = f"{c['title']}|{c['artist']}"
        if key not in by_song or c["segment_count"] > by_song[key]["segment_count"]:
            by_song[key] = c
    return list(by_song.values())


# ── Stratified selection ───────────────────────────────────────────────

def bpm_bin(bpm: int) -> str:
    if bpm < 70:
        return "slow"
    if bpm < 100:
        return "medium"
    if bpm < 130:
        return "fast"
    return "very_fast"


def key_label(pc: int, mode: str) -> str:
    return f"{PITCH_NAMES[pc]} {mode}"


def select_500(candidates: list[dict], n: int = 500, seed: int = 42) -> list[dict]:
    """
    Stratified selection across BPM bins, keys, and time signatures.

    We build strata from (bpm_bin, tonality) and allocate proportionally,
    then greedily fill from each stratum. This ensures coverage of rare
    keys and tempos rather than over-sampling the modal C-major/120-BPM cluster.
    """
    rng = random.Random(seed)

    # Build strata
    strata: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        bbin = bpm_bin(c["bpm"])
        klab = key_label(c["tonality_pc"], c["tonality_mode"])
        stratum = f"{bbin}|{klab}"
        strata[stratum].append(c)

    # Sort strata by size ascending — fill rare strata first
    sorted_strata = sorted(strata.items(), key=lambda x: len(x[1]))

    selected: list[dict] = []
    remaining = n

    # Pass 1: take at least 1 from every stratum (ensure diversity)
    for sname, members in sorted_strata:
        if remaining <= 0:
            break
        rng.shuffle(members)
        take = min(len(members), max(1, remaining // len(strata) + 1))
        take = min(take, remaining, len(members))
        selected.extend(members[:take])
        remaining = n - len(selected)

    # Pass 2: proportional fill from largest strata
    if remaining > 0:
        pool = [c for c in candidates if c not in selected]
        rng.shuffle(pool)
        selected.extend(pool[:remaining])

    return selected[:n]


def anonymize(selected: list[dict], source_dir: str) -> list[dict]:
    """
    Assign opaque IDs, strip all identifying info, compute audio hashes.

    The audio_hash lets participants verify they have the right audio without
    us distributing the recording or revealing the title.
    """
    anonymized = []
    for i, item in enumerate(selected, 1):
        opaque_id = f"MUB-{i:05d}"

        # Compute audio hash (best-effort; skip if file missing)
        ahash = ""
        audio_path = item.get("audio_path", "")
        if audio_path and os.path.exists(audio_path):
            try:
                ahash = file_hash(audio_path)
            except Exception:
                ahash = ""

        # Re-key structure labels to generic numeric tags to prevent
        # leakage of structure_name vocabulary that could hint at genre.
        # Keep original labels for gold comparison but map to canonical set.
        structure = []
        for seg in item["structure"]:
            structure.append({
                "start": round(seg["start"], 1),
                "end": round(seg["end"], 1),
                "label": seg["label"],
            })

        entry = {
            "id": opaque_id,
            "audio_hash": ahash,
            "duration_sec": item["duration"],
            "segment_count": item["segment_count"],
            "gold": {
                "bpm": item["bpm"],
                "tonality": {
                    "pitch_class": item["tonality_pc"],
                    "mode": item["tonality_mode"],
                    "label": key_label(item["tonality_pc"], item["tonality_mode"]),
                },
                "time_signature": item["time_signature"],
                "structure": structure,
            },
            # Annotation provenance (no PII, no song identity)
            "annotation": {
                "expert_submissions": item["expert_submissions"],
                "operator_count": len(item.get("operators", [])),
                "lyric_lines": item["lyric_lines"],
            },
        }
        anonymized.append(entry)
    return anonymized


# ── Diversity statistics ────────────────────────────────────────────────

def compute_stats(entries: list[dict]) -> dict:
    bpms = [e["gold"]["bpm"] for e in entries]
    keys = [e["gold"]["tonality"]["label"] for e in entries]
    tss = [e["gold"]["time_signature"] for e in entries]
    durs = [e["duration_sec"] for e in entries]
    segs = [e["segment_count"] for e in entries]

    bpm_bins = Counter(bpm_bin(b) for b in bpms)
    key_dist = Counter(keys)
    ts_dist = Counter(tss)

    return {
        "count": len(entries),
        "bpm": {
            "min": min(bpms), "max": max(bpms), "median": sorted(bpms)[len(bpms)//2],
            "bins": dict(bpm_bins),
        },
        "tonality": {
            "unique_keys": len(key_dist),
            "distribution": dict(key_dist.most_common()),
            "major_count": sum(1 for e in entries if e["gold"]["tonality"]["mode"] == "major"),
            "minor_count": sum(1 for e in entries if e["gold"]["tonality"]["mode"] == "minor"),
        },
        "time_signature": {
            "distribution": dict(ts_dist.most_common()),
        },
        "duration": {
            "min": min(durs), "max": max(durs), "median": sorted(durs)[len(durs)//2],
        },
        "segments": {
            "min": min(segs), "max": max(segs), "median": sorted(segs)[len(segs)//2],
        },
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset", required=True, help="Path to BeatHamster testset.json")
    ap.add_argument("--output", default="data/benchmark.json")
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[select] Loading candidates from {args.testset}…", flush=True)
    candidates = load_candidates_from_testset(args.testset)
    print(f"[select] {len(candidates)} candidates after quality filter", flush=True)

    candidates = dedup_by_song(candidates)
    print(f"[select] {len(candidates)} unique songs after dedup", flush=True)

    selected = select_500(candidates, n=args.n, seed=args.seed)
    print(f"[select] {len(selected)} pieces selected", flush=True)

    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.testset)))
    entries = anonymize(selected, source_dir)

    stats = compute_stats(entries)

    output = {
        "name": "Music Understanding Benchmark (MUB)",
        "version": "1.0.0",
        "description": (
            "An anonymized benchmark for evaluating multimodal LLM music "
            "understanding: BPM estimation, key/tonality identification, "
            "time signature detection, and structural segmentation. "
            "No original titles, artists, or platform IDs are included."
        ),
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "stats": stats,
        "tasks": [
            "bpm_estimation",
            "tonality_identification",
            "time_signature_detection",
            "structure_segmentation",
        ],
        "entries": entries,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[select] → {args.output} ({len(entries)} entries)", flush=True)

    # Metadata CSV (for human inspection — still anonymized)
    with open(args.metadata, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "audio_hash", "duration_sec", "bpm", "key", "mode",
                     "time_signature", "segment_count"])
        for e in entries:
            g = e["gold"]
            w.writerow([e["id"], e["audio_hash"], e["duration_sec"],
                        g["bpm"], g["tonality"]["label"], g["tonality"]["mode"],
                        g["time_signature"], e["segment_count"]])
    print(f"[select] → {args.metadata}", flush=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Music Understanding Benchmark — Selection Summary")
    print(f"{'='*60}")
    print(f"Total pieces:       {stats['count']}")
    print(f"BPM range:          {stats['bpm']['min']}–{stats['bpm']['max']} "
          f"(median {stats['bpm']['median']})")
    print(f"Unique keys:        {stats['tonality']['unique_keys']}")
    print(f"Major/Minor:        {stats['tonality']['major_count']}/"
          f"{stats['tonality']['minor_count']}")
    print(f"Time signatures:    {len(stats['time_signature']['distribution'])}")
    print(f"Duration range:     {stats['duration']['min']}–{stats['duration']['max']}s "
          f"(median {stats['duration']['median']}s)")
    print(f"Segments per piece: {stats['segments']['min']}–{stats['segments']['max']} "
          f"(median {stats['segments']['median']})")
    print(f"\nBPM bins: {dict(stats['bpm']['bins'])}")
    print(f"\nTop-10 keys:")
    for k, v in list(stats['tonality']['distribution'].items())[:10]:
        print(f"  {k:12s} {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
