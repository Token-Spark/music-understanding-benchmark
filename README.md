# MUB: Music Understanding Benchmark

An anonymized benchmark for evaluating multimodal large language models (MLLMs) on music structural understanding.

## Overview

MUB evaluates whether multimodal LLMs can truly *listen* to music by testing four tasks that professional music analysts perform:

| Task | Description | Gold Source |
|------|-------------|-------------|
| **BPM Estimation** | Estimate the tempo in beats per minute | Expert annotation modal BPM |
| **Tonality Identification** | Identify the musical key (pitch class + mode) | Expert annotation modal tonality |
| **Time Signature Detection** | Detect the time signature (e.g., 4/4, 3/4) | Expert annotation modal time signature |
| **Structural Segmentation** | Segment the music into structural parts | Expert-annotated segment boundaries + labels |

### Key Design Principle: Full Anonymization

**No original titles, artist names, or platform identifiers appear in the benchmark.** Each piece is assigned an opaque ID (MUB-00001 to MUB-00500). This prevents models from retrieving metadata via title-based lookup, ensuring evaluation measures genuine audio understanding rather than memorized associations.

## Dataset

- **500 music pieces** selected via stratified sampling from 1,111 candidate songs
- **24 key/mode combinations** (all 12 pitch classes × major/minor)
- **BPM range**: 47–180 (median 101)
- **Duration**: 92–535 seconds (median 228)
- **Segments per piece**: 5–22 (median 9)
- **Total annotated segments**: 4,524

### Copyright

Audio files are **not distributed** with this benchmark. The benchmark contains only:
- Anonymized metadata (duration, segment count)
- Audio content hashes (SHA-256 of first 1MB) for verification
- Expert-annotated gold labels (BPM, key, time signature, structure)

Participants must supply their own audio files. The audio hash enables verification without sharing copyrighted content.

## Quick Start

### 1. Prepare prompts

```bash
python src/prepare_prompts.py \
    --benchmark data/benchmark.json \
    --output-dir data/prompts/
```

### 2. Run your model

Feed each prompt + corresponding audio to your MLLM. Save predictions as JSON:

```json
[
  {
    "id": "MUB-00001",
    "bpm": 120,
    "key": "C major",
    "time_signature": "4/4",
    "structure": [
      {"start": 0.0, "end": 15.0, "label": "intro"},
      {"start": 15.0, "end": 45.0, "label": "verse"}
    ]
  }
]
```

### 3. Evaluate

```bash
python src/evaluate.py \
    --benchmark data/benchmark.json \
    --predictions results/your_model.json \
    --output results/report_your_model.json \
    --model-name "your-model-name"
```

### 4. Run baselines

```bash
# Majority-vote baseline (no audio needed)
python baselines/majority_baseline.py \
    --benchmark data/benchmark.json \
    --output results/majority_baseline.json

# DSP baseline (requires audio files + ffmpeg)
python baselines/dsp_baseline.py \
    --benchmark data/benchmark.json \
    --audio-dir /path/to/audio \
    --output results/dsp_baseline.json
```

## Repository Structure

```
music-understanding-benchmark/
├── data/
│   ├── benchmark.json          # 500 anonymized entries with gold labels
│   ├── metadata.csv            # CSV summary for quick inspection
│   └── prompts/                # Generated prompt files (after prepare_prompts.py)
├── src/
│   ├── metrics.py              # Pure-function metric implementations
│   ├── prepare_prompts.py      # Generate evaluation prompts
│   └── evaluate.py             # Score predictions against gold
├── baselines/
│   ├── dsp_baseline.py         # Objective DSP baseline (BPM/key/timesig from waveforms)
│   └── majority_baseline.py    # Majority-vote baseline (text-only lower bound)
├── scripts/
│   └── select_and_anonymize.py # Selection + anonymization script (for reproduction)
├── paper/
│   └── main.tex                # Research paper (LaTeX)
├── results/                    # (created on first evaluation run)
├── requirements.txt
├── LICENSE
└── README.md
```

## Metrics

| Task | Metric | Tolerance |
|------|--------|-----------|
| BPM | Binary hit (relative error) | ±4% of gold BPM |
| Tonality | Exact match or relative major/minor | Same (PC, mode) or 3-semitone offset |
| Time signature | Exact match | Normalized string comparison |
| Structure | Greedy IoU-based F1 | IoU ≥ 0.5 for segment match |

## Citation

```bibtex
@misc{mub2026,
  title={MUB: An Anonymized Benchmark for Evaluating Multimodal Large Language Models on Music Structure Understanding},
  author={Anonymous},
  year={2026}
}
```

## License

- **Code**: MIT License
- **Data (benchmark.json)**: CC BY-SA 4.0
- **Audio**: Not distributed (participants supply their own)

## References

This benchmark is inspired by MuChoMusic (ISMIR 2024) and uses professional music annotation data as gold labels. See `paper/main.tex` for the full methodology and related work.
