# Task ID: 2

**Title:** Train humor prediction model

**Status:** DONE 2026-09-08 — M3 PASS: RoBERTa regression rho=0.313 > M1 0.277 (Kaggle kernel, results in evaluation_results/results_v1_kaggle.json; model pt in training_output/)

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
