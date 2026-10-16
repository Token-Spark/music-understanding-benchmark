# MUB: Music Understanding Benchmark

An identity-masked benchmark for evaluating multimodal language models on tempo, key, metre, and structural segmentation in music.

## Quick Start (Demo)

Run the full pipeline with 5 real entries from the benchmark — no audio files or GPU needed:

```bash
pip install numpy
python demo/run_demo.py
```

This will extract a 5-entry subset from `data/benchmark.json`, generate prompts, run the majority-vote baseline, and evaluate the results.

## Full Benchmark Usage

### 1. Generate prompts

```bash
python src/prepare_prompts.py \
    --benchmark data/benchmark.json \
    --output-dir data/prompts/
```

### 2. Run your model

Send each prompt with its matching audio file to your model. **Only forward the audio and task text** — never send `audio_hash`, gold labels, or source paths to the model.

Save predictions as JSON:

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

## Project Structure

```
music-understanding-benchmark/
├── data/
│   ├── benchmark.json          # 500 identity-masked entries with gold labels
│   ├── metadata.csv             # CSV summary for quick inspection
│   └── prompts/                 # Generated prompt files (5 tasks × 500 entries)
├── demo/
│   ├── generate_demo_subset.py  # Extract 5-entry demo from real data
│   ├── run_demo.py              # One-command pipeline demo
│   └── demo_benchmark.json      # Generated demo data (after running run_demo.py)
├── src/
│   ├── metrics.py               # Metric implementations
│   ├── prepare_prompts.py       # Generate evaluation prompts
│   └── evaluate.py              # Score predictions against gold
├── baselines/
│   ├── dsp_baseline.py          # Waveform BPM/key/metre reference (needs ffmpeg)
│   └── majority_baseline.py     # Test-distribution prior (no audio needed)
├── scripts/
│   └── select_and_anonymize.py  # Selection + identity masking script
├── paper/
│   ├── main.tex                 # Research paper (LaTeX source)
│   ├── main.pdf                 # Compiled paper
│   └── review_comments_2026-10-16.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

| Property | Value |
|----------|-------|
| Entries | 500 |
| BPM range | 47–180 (median 101) |
| Key/mode combinations | 24 (all major/minor) |
| Major / Minor | 268 / 232 |
| Time signatures | 4/4: 492, 3/4: 4, 6/8: 4 |
| Duration | 92–535 s (median 228 s) |
| Structural intervals | 4,676 total (5–22 per entry) |

Labels are derived from an existing annotation workflow. The upstream builder uses the last available submission and modal segment attributes; this does not establish independent annotator agreement or a completed quality-review gate.

## Identity Masking & Inference Protocol

Original titles, artists, platform song/task IDs, account names, and operator names are excluded from model-facing inputs. Each entry has an opaque ID (MUB-00001 to MUB-00500). This reduces title-based lookup but cannot prevent audio recognition, memorization, or training contamination.

Generated JSONL prompt records include evaluator-side routing fields (`audio_hash`, `duration_sec`). **Forward only the audio and task prompt to the model.** Do not forward `audio_hash`, gold labels, revealing filenames, embedded tags, or source paths.

Prompts explicitly prohibit web search, internet access, downloads, code execution, external tools, external DSP, fingerprinting, music recognition, and database lookup. However, prompt text alone cannot enforce compliance — the evaluation harness must disable these capabilities.

## Metrics & Known Limitations

| Task | Current metric | Qualification |
|------|----------------|---------------|
| BPM | Hit within ±4% relative error | Convention; no octave credit |
| Key | Legacy exact-or-directional-relative score | Defective/asymmetric; not standard key accuracy |
| Time signature | Normalized exact match | Dominated by 4/4 frequency |
| Structure | Greedy interval IoU F1 (threshold 0.5) | Labels ignored; no label accuracy implemented |

**Outstanding issues (see `paper/review_comments_2026-10-16.md` for details):**

- The legacy key metric is asymmetric and should be replaced with exact-key accuracy as primary.
- The evaluator excludes missing predictions from denominators; a corrected version must use a fixed 500-entry denominator.
- No recording-level rights clearance was performed; withholding audio is not proof of rights safety.
- No independent annotation agreement study has been conducted.
- No contemporary MLLM comparison has been completed.
- Metre distribution is extremely imbalanced (98.4% in 4/4).

## Paper

See [`paper/main.pdf`](paper/main.pdf) for the full research paper, including related work, methodology, evaluation audit, and limitations.

## Authors

- **Kun Liu** — founder@tokenspark.uno
- **Lyra Tao** — wenyumusic@qq.com

## Citation

```bibtex
@misc{mub2026,
  title={MUB: An Identity-Masked Benchmark for Music Understanding in Multimodal Language Models},
  author={Liu, Kun and Tao, Lyra},
  year={2026}
}
```

## License

- **Code**: MIT License (see [`LICENSE`](LICENSE))
- **Data and annotations**: Rights and redistribution terms are not yet established.
- **Audio**: Not distributed. Participants must supply their own audio through a legitimate access route.

## Related Work

This benchmark is positioned against:

- MuChoMusic (ISMIR 2024) — [arXiv:2408.01337](https://arxiv.org/abs/2408.01337)
- Audio Flamingo 3 (2025) — [arXiv:2507.08128](https://arxiv.org/abs/2507.08128)
- Music Flamingo (2025) — [arXiv:2511.10289](https://arxiv.org/abs/2511.10289)
- Exploring GPT's Ability as a Judge in Music Understanding (2025) — [arXiv:2501.13261](https://arxiv.org/abs/2501.13261)
- HumMusQA (2026) — [arXiv:2603.27877](https://arxiv.org/abs/2603.27877)
- GaMMA (2026) — [arXiv:2605.00371](https://arxiv.org/abs/2605.00371)
- MusTBench (2026) — [arXiv:2605.29300](https://arxiv.org/abs/2605.29300)

Verified citations and task-overlap analysis are in `paper/main.tex`.
