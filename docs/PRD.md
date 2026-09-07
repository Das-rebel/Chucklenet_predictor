# Chucklenet Predictor PRD v1.0 — Humor Strength Regression

**Date:** 2026-09-08
**Status:** ACTIVE — RETHUNK after alpnewr session findings (v20 gate forensics + Sep-7 audit)
**Repository:** github.com/Das-rebel/Chucklenet_predictor
**Parent project:** ChuckleNet (github.com/Das-rebel/ChuckleNet) — this repo implements **G4: humor STRENGTH prediction (0–100)** from the ChuckleNet decision graph, and serves **Product 3 (Content Scorer)** of ChuckleNet PRD v6.
**Supersedes:** original README claims (multimodal v0, reverse generation, 12+ humor types)

---

## 1. Why the rethink (evidence from alpnewr / Sep-7 audit)

| Finding (ChuckleNet core) | Consequence for this repo |
|---|---|
| 620-video corpus is marker-poor: 243,501 utts, **1.16% positive**; labels genuine but sparse | Word-level laughter prediction is label-starved. Humor-strength regression escapes this by using **abundant continuous human ratings** (Jester: 1.7M+) |
| Honest IoU-F1@0.2 = **0.31**, below 0.51 baseline; every ≥0.9 claim was pseudo-label circularity / random split | This repo must inherit **G1: honest metrics only**. Our first training data **violated it**: binary labels assigned RANDOM strengths (65–95 / 5–35) = training on injected noise |
| Word-level cascade dead at IoU-F1 0.50 ceiling | Do NOT build strength scoring on top of cascade outputs. Standalone text regression first |
| Jester 140-joke subset: narrow 36–69 range | Jester alone ≠ generalization proof. Need joke-level GroupKFold + external binary eval |
| CLIP in a text-only pipeline; sentence-transformer concat unjustified | Architecture was over-built for v0 with zero training evidence. Cut to one backbone + regression head |
| Audio broken (NumPy conflict) + Whisper/AudioCraft absent | Audio is **out of v1 scope**. Prosody delivery-scoring is v2, reusing core-repo prosody assets |

---

## 2. Product

**One sentence:** Given a piece of comedic text, predict a calibrated **funny-strength score 0–100**, with honest held-out-joke evaluation.

**Users:** content creators (headline/caption/script scoring), comedy writers (joke A/B testing), the ChuckleNet portfolio (career-first monetization, DEFINITIVE_PLAN Option C).

**Non-goals (v1):** audio input, reverse generation, per-person laughter/sarcasm, real-time inference SLA.

---

## 3. Data strategy (v1 — replaces the noise-injected dataset)

| Source | Labels | Role |
|---|---|---|
| **Jester 1+2+3** (~300 jokes, ~2M continuous ratings −10..+10) | Real human, continuous | PRIMARY: normalized per-joke mean → 0–100. Gold for training + eval |
| Reddit-style joke corpora with upvote scores (Kaggle) | Continuous-ish social signal | SECONDARY: scale + domain breadth; normalize per-subreddit |
| 200k binary humor dataset | Binary only | AUX binary head only. **Never convert to fake strength again (Rule D-R1)** |
| shortjokes.csv (231K, no ratings) | None | Unlabeled pretraining / augmentation only |

**Kill list:** `kaggle_data/train.jsonl` + `val.jsonl` in current form (random-strength injection) — quarantined, not deleted (NO DELETES policy).

---

## 4. Evaluation protocol (inherits ChuckleNet G1)

1. **Splits:** GroupKFold by joke source set — never random splits on utterance level.
2. **Metrics:** Spearman ρ + Kendall τ (primary, continuous ratings); MAE/RMSE (secondary); PR-AUC for the binary aux head. **No "accuracy over 101 bins"** — meaningless for regression.
3. **Provenance:** every published number cites a result JSON in `evaluation_results/` with split seed, data hash, model hash. **No claim without a source JSON** (rule mirrored from core repo).
4. **Baselines:** M0 = predict dataset mean; M1 = TF-IDF + Ridge; M2 = fine-tuned transformer. M2 must beat M1 to justify its cost.

**Honest targets (v1):** M1 Spearman ≥ 0.35; M2 Spearman ≥ 0.55 on held-out joke groups. If M2 ≤ M1, ship M1 and say so (cf. "When Simple Beats Deep" lesson).

---

## 5. Architecture (v1 — simplified)

```
Input text
  → RoBERTa-base (or DeBERTa-v3-small) encoder        [single backbone]
  → mean-pool + dropout
  → regression head → strength 0–100 (scaled sigmoid or linear+clip)
  → (optional aux) binary funny/not-funny head         [200k dataset, multi-task]
```

**Removed from v0:** CLIP text tower, sentence-transformer 384-d concat, 5× LSTM humor-type extractors (untrained, unvalidated — keep as experimental module behind a flag), Whisper/audio stack.

**Training:** Colab GPU (`colab_train.py`, already verified), MSE on ratings/100, AdamW 3e-5, early stop on val Spearman. ~20–30 min on T4.

---

## 6. Roadmap

| Milestone | Deliverable | Gate |
|---|---|---|
| **M1 Data** | Rebuild dataset from Jester continuous + upvote corpora; `data/REBUILD_NOTES.md` | Range covers ≥ 5–95 with real ratings; provenance table |
| **M2 Baselines** | M0/M1 baselines + eval harness (`evaluate.py`) | M1 Spearman ≥ 0.35, JSON published |
| **M3 Transformer** | Fine-tune RoBERTa regression; Colab notebook v2 | M2 Spearman ≥ 0.55 AND > M1 |
| **M4 Calibration + Demo** | 0–100 calibration curve; Gradio/API demo | Calibration MAE ≤ 10 pts; demo runnable |
| **M5 (v2)** | Multimodal delivery scoring — reuse ChuckleNet prosody features | Only after M4; separate PRD addendum |
| **M6 (v3)** | Reverse generation to target strength | Gated on M4 + text-conditioned generation eval |

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| ~300 rated jokes → weak generalization | Add upvote-scored corpora; report per-source held-out results; do NOT oversell |
| Upvote scores = popularity not funniness | Treat as weak labels; downweight in loss; ablate with/without |
| Jester ratings are per-user-context | Use per-joke mean + std; exclude low-rating-count jokes |
| Regression collapse to mean | Monitor prediction variance; ordinal/binning fallback |
| Scope creep back to audio | v1 hard-scoped text-only |

---

## 8. Success criteria (v1 exit)

- [ ] M2 beats M1 on held-out-joke Spearman with published JSONs
- [ ] Calibration map score→0–100 within ±10 MAE on Jester gold subset
- [ ] README rewritten: claims == result JSONs
- [ ] Demo deployable (HF Spaces or local Gradio)
- [ ] This PRD + VISION + decision log committed to repo
