#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metric implementations for the Music Understanding Benchmark (MUB).

All metrics are pure functions: (prediction, gold) → score.
This module is the single source of truth for threshold definitions
and comparison logic. Import from evaluate.py — do not duplicate logic.

Tasks and metrics:
  1. BPM estimation          → bpm_hit (relative error ≤ 4%)
  2. Tonality identification → tonality_match (exact or relative major/minor)
  3. Time signature detection → timesig_exact (exact string match)
  4. Structure segmentation  → structure_f1 (greedy IoU≥0.5 matching)
"""
from __future__ import annotations

from typing import Any

# ── Constants ──────────────────────────────────────────────────────────

PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

BPM_TOLERANCE = 0.04          # ±4% relative error → same tempo
IOU_THRESHOLD = 0.5           # structure segment matching threshold


# ── 1. BPM ─────────────────────────────────────────────────────────────

def bpm_hit(pred: float | None, gold: float, tolerance: float = BPM_TOLERANCE) -> bool:
    """True if |pred − gold| ≤ gold × tolerance."""
    if pred is None or gold is None or gold == 0:
        return False
    return abs(pred - gold) <= gold * tolerance


def bpm_relative_error(pred: float | None, gold: float) -> float | None:
    """Relative error |pred−gold|/gold, or None if undefined."""
    if pred is None or gold is None or gold == 0:
        return None
    return abs(pred - gold) / gold


# ── 2. Tonality ────────────────────────────────────────────────────────

def parse_key(label: str | None) -> tuple[int, str] | None:
    """
    Parse 'C Natural major scale' → (0, 'major').
    Returns (pitchClass, mode) or None if unparseable.
    """
    import re
    if not label:
        return None
    m = re.match(r"^([A-G](?:#|b)?)\s*(.*)$", str(label))
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


def tonality_match(
    pred: tuple[int, str] | str | None,
    gold: tuple[int, str] | None,
) -> bool:
    """
    True if pred and gold share the same tonic and mode,
    OR if they are relative major/minor (3-semitone offset, different mode).
    Relative major/minor is perceptually the same key.
    """
    if isinstance(pred, str):
        pred = parse_key(pred)
    if not pred or not gold:
        return False
    if pred[0] == gold[0] and pred[1] == gold[1]:
        return True
    if (pred[0] + 3) % 12 == gold[0] and pred[1] != gold[1]:
        return True
    return False


# ── 3. Time signature ──────────────────────────────────────────────────

def timesig_match(pred: str | None, gold: str | None) -> bool:
    """Exact string match after normalisation."""
    if not pred or not gold:
        return False
    return _normalize_timesig(pred) == _normalize_timesig(gold)


def _normalize_timesig(ts: str) -> str:
    """Normalize '4/4' and '4 / 4' and '4,4' to '4/4'."""
    import re
    nums = re.findall(r"\d+", str(ts))
    if len(nums) >= 2:
        return f"{nums[0]}/{nums[1]}"
    return str(ts).strip()


# ── 4. Structure segmentation ───────────────────────────────────────────

def _iou(a: dict, b: dict) -> float:
    """Intersection-over-Union of two time intervals."""
    inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def structure_f1(
    gold_segs: list[dict],
    pred_segs: list[dict] | None,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, float]:
    """
    Greedy IoU-based segment matching → precision, recall, F1.

    For each gold segment, find the best-IoU unmatched prediction.
    If IoU ≥ threshold, it counts as a match. Greedy (not Hungarian)
    matching is sufficient given the small number of segments (5–22).

    Returns: {precision, recall, f1, matched, gold_count, pred_count}
    """
    if not pred_segs or not gold_segs:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "matched": 0,
            "gold_count": len(gold_segs),
            "pred_count": len(pred_segs or []),
        }

    used: set[int] = set()
    matched = 0
    for g in gold_segs:
        best_idx, best_iou = -1, 0.0
        for i, p in enumerate(pred_segs):
            if i in used:
                continue
            v = _iou(g, p)
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_idx >= 0 and best_iou >= iou_threshold:
            used.add(best_idx)
            matched += 1

    precision = matched / len(pred_segs) if pred_segs else 0.0
    recall = matched / len(gold_segs) if gold_segs else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": matched,
        "gold_count": len(gold_segs),
        "pred_count": len(pred_segs),
    }


# ── 5. Aggregate scoring ───────────────────────────────────────────────

def evaluate_predictions(
    predictions: list[dict], gold_entries: list[dict],
) -> dict[str, Any]:
    """
    Compute aggregate metrics across all benchmark entries.

    Each prediction must have:
      - id: str (matching gold entry id)
      - bpm: float | None
      - key: str | tuple[int,str] | None  (e.g. "C major" or [0, "major"])
      - time_signature: str | None
      - structure: list[{start, end, label}] | None

    Returns a results dict with per-task accuracy and aggregate stats.
    """
    gold_by_id = {e["id"]: e for e in gold_entries}

    bpm_hits, bpm_total = 0, 0
    bpm_errors: list[float] = []
    key_hits, key_total = 0, 0
    ts_hits, ts_total = 0, 0
    struct_f1s: list[float] = []
    struct_precisions: list[float] = []
    struct_recalls: list[float] = []

    per_entry: list[dict] = []

    for pred in predictions:
        gid = pred.get("id")
        if gid not in gold_by_id:
            continue
        gold = gold_by_id[gid]
        gg = gold["gold"]

        result: dict[str, Any] = {"id": gid}

        # BPM
        if gg.get("bpm") is not None:
            bpm_total += 1
            hit = bpm_hit(pred.get("bpm"), gg["bpm"])
            if hit:
                bpm_hits += 1
            err = bpm_relative_error(pred.get("bpm"), gg["bpm"])
            if err is not None:
                bpm_errors.append(err)
            result["bpm_hit"] = hit

        # Tonality
        gold_key = gg.get("tonality")
        if gold_key and "pitch_class" in gold_key:
            gold_tuple = (gold_key["pitch_class"], gold_key["mode"])
            key_total += 1
            hit = tonality_match(pred.get("key"), gold_tuple)
            if hit:
                key_hits += 1
            result["key_hit"] = hit

        # Time signature
        if gg.get("time_signature"):
            ts_total += 1
            hit = timesig_match(pred.get("time_signature"), gg["time_signature"])
            if hit:
                ts_hits += 1
            result["ts_hit"] = hit

        # Structure
        gold_segs = gg.get("structure", [])
        if gold_segs:
            pred_segs = pred.get("structure")
            sf = structure_f1(gold_segs, pred_segs)
            struct_f1s.append(sf["f1"])
            struct_precisions.append(sf["precision"])
            struct_recalls.append(sf["recall"])
            result["structure"] = sf

        per_entry.append(result)

    import statistics

    def safe_mean(lst):
        return round(statistics.mean(lst), 4) if lst else 0.0

    def safe_median(lst):
        return round(statistics.median(lst), 4) if lst else 0.0

    return {
        "count": len(per_entry),
        "bpm": {
            "accuracy": round(bpm_hits / bpm_total, 4) if bpm_total else 0,
            "total": bpm_total,
            "median_relative_error": safe_median(bpm_errors),
            "mean_relative_error": safe_mean(bpm_errors),
        },
        "tonality": {
            "accuracy": round(key_hits / key_total, 4) if key_total else 0,
            "total": key_total,
        },
        "time_signature": {
            "accuracy": round(ts_hits / ts_total, 4) if ts_total else 0,
            "total": ts_total,
        },
        "structure": {
            "mean_f1": safe_mean(struct_f1s),
            "mean_precision": safe_mean(struct_precisions),
            "mean_recall": safe_mean(struct_recalls),
            "median_f1": safe_median(struct_f1s),
            "total": len(struct_f1s),
        },
        "per_entry": per_entry,
    }
