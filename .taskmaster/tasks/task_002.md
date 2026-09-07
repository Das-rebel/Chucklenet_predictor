# Task ID: 2

**Title:** Train humor prediction model

**Status:** RESCOPED 2026-09-08 — blocked on task_006 data rebuild; use colab_train.py loop on REAL ratings (see docs/PRD.md §4-5)

**Dependencies:** 1

**Priority:** critical

**Description:** Fine-tune the untrained TextHumorClassifier on a humor dataset so predictions are meaningful (not constant ~50/100).

**Details:**
Model is untrained — all predictions converge to ~50/100 regardless of input.

Options:
- A1: Fine-tune on humor dataset (Reddit jokes, Puns dataset, humor detection datasets)
- A2: Use pre-trained humor model weights if available
- A3: Train from scratch — needs large labeled dataset

Recommended: A1 — find labeled humor dataset and fine-tune.

**Test Strategy:**
After training, jokes should score 60-90, non-jokes should score 10-40.
