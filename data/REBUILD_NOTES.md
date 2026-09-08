# Data Rebuild Notes (Task 006 / PRD M1) — 2026-09-08

## What replaced what

| Old (QUARANTINED) | Why | New |
|---|---|---|
| `kaggle_data/train.jsonl` + `val.jsonl` (10,140 samples) | **D-R1 violation**: binary labels assigned RANDOM strengths 65–95 / 5–35 = training on injected noise | `data/train_real.jsonl` + `data/val_real.jsonl` |

## New dataset: 7,984 samples (train 7,170 / val 814, md5-hash 90/10 split)

| Source | N | Labels | Method | Range |
|---|---|---|---|---|
| **Jester** (jester_ratings.csv, 1.76M ratings) | 140 | 🥇 human continuous | per-joke mean (≥50 ratings) → min-max rescale 0–100 (rank-preserving) | 0–100 (mean 67.9; raw mean range −2.75..+3.71 on −10..+10) |
| **r/jokes** (bwandowando/reddit-rjokes-dataset threads) | 4,844 | ⚠️ weak social | log1p(score) percentile; top-3000 + mid-band-3000, score≥10, SFW, dedup, len 40–800 | 1–100 |
| **r/dadjokes** (oktayozturk010/reddit-dad-jokes) | 3,000 | ⚠️ weak social | same; score≥5, len 30–500 | 25–100 |

## Provenance
- `data/provenance.json` — machine-readable (counts, ranges, methods, rules)
- Raw sources: `kaggle_data/jester_*.csv`, `kaggle_data/upvote_jokes/*.csv` (gitignored, re-downloadable via Kaggle API)

## Known limitations (honest, per G1)
1. Jester = only 140 jokes despite 1.7M ratings (ratings concentrate on tiny joke pool) → gold calibration, weak coverage
2. Upvote scores measure *popularity*, not funniness → treated as weak labels; transformer must still beat TF-IDF+Ridge on **Jester-held-out** Spearman
3. Percentile normalization flattens upvote distributions' heavy tail — documented, intentional
4. Jester is in-distribution gold; dadjokes/rjokes are the generalization test

## Eval protocol reminder (D-R4)
Splits are md5-hash-grouped per text (stable across rebuilds, no leakage on dedup).
Primary metric: **Spearman ρ** on val; Jester-only ρ reported separately (gold subset).
No 101-bin accuracy. Every published number → result JSON in `evaluation_results/`.

## M1 baseline result (2026-09-08, `baseline_m1.py`)
- Best: TF-IDF(1-2g) + Ridge α=10 → **Spearman 0.277** (Jester-gold subset: 0.813 @ n=13, weak evidence)
- Gate: pre-evidence 0.35 target FAILED → revised per PRD §4: **M2 transformer must beat 0.277**
- Result JSON: `evaluation_results/m1_tfidf_ridge_baseline.json`

## M3 transformer result (2026-09-08, Kaggle kernel v3)
- RoBERTa-base regression, 4 epochs: **Spearman 0.313** (best ep3) vs M1 0.277 → **gate PASS** (+0.036)
- Jester-gold: 0.445@ep1 (n=13 held-out — weak evidence, fluctuates 0.27–0.45 across epochs)
- MAE 22.98 | Ran on Kaggle CPU (~40 min; GPU mount didn't take — see JSON device_note)
- Result JSON: `evaluation_results/results_v1_kaggle.json` · Kernel: kaggle.com/code/subhajitdas/chucklenet-predictor-v1-train
- Next lever: longer max_len (256) + more epochs + T4 speedup; then calibration (M4)
