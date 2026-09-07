# Chucklenet Predictor — Vision

**Version:** 1.0 · **Date:** 2026-09-08 · **Status:** Active (post-alpnewr rethink)

---

## One-line vision

> **ChuckleNet measures laughter; Chucklenet Predictor measures funniness.**
> One repo is the research instrument (where *will* audiences laugh — a hard, honest, partially-unsolved problem); this repo is the product instrument (*how funny is this text, 0–100* — a solvable problem with abundant human ratings).

---

## Where this fits in the ChuckleNet platform

ChuckleNet PRD v6 defined three products. The Sep-2026 alpnewr findings (620-video corpus marker-poor at 1.16% positive; honest IoU-F1@0.2 = 0.31 vs 0.51 baseline; cascade dead at IoU 0.50) showed that **Product 1 (Group Laughter Prediction) is a long-horizon research problem**, gated on better label density.

This repo **activates G4** (humor strength prediction, deferred in the ChuckleNet decision graph) and implements the **product-track scorer** (Product 3 territory: content scoring) *now*, using a completely different data foundation:

| | ChuckleNet (core) | Chucklenet Predictor (this repo) |
|---|---|---|
| Task | Where will the audience laugh (positioning) | How funny is this text (strength 0–100) |
| Labels | Audience-reaction positions in captions — sparse (1.16%), boundary-ambiguous | Continuous human humor ratings (Jester ~2M) + social upvote signals |
| Status | Honest IoU-F1@0.2 = 0.31, below baseline — open research | Solvable now; regression on real ratings |
| Shared assets | Prosody features, audio pipeline, Colab discipline, honest-eval rules (G1) | Reuses eval discipline + later prosody for delivery scoring (v2) |

**The strategic point:** after a month of forensics, the core project learned that leakage-free evaluation is the scarcest resource in humor ML (0.975 → 0.31 once audited). This repo is built *from day one* on that discipline: provenance JSONs, held-out-joke splits, baselines-first. It is the demo that the ChuckleNet way of doing ML — honest, reproducible, baseline-anchored — ships products, not just papers.

---

## Product trajectory

1. **v1 (now): text strength scorer** — calibrated 0–100, Spearman-first eval, demo app. Portfolio-grade.
2. **v2: delivery-aware scoring** — same text scored by *delivery* prosody variants (reuse core prosody assets). Unique vs all text-only scorers; directly serves comedians ("this line lands better with a 0.8s pause").
3. **v3: reverse generation** — rewrite text toward a target strength. Research-grade; gated on v1 quality.
4. **Bridge back:** when core Product 1 solves label density, this scorer becomes its *content-level complement*: score the joke (us) + predict the laugh position (core) = full comedy stack.

---

## Principles (inherited + new)

1. **Honest metrics only** — no claim without a source JSON (ChuckleNet G1).
2. **Real ratings or nothing** — binary labels are never dressed up as continuous scores (lesson D-R1).
3. **Baselines first** — a transformer must beat TF-IDF+Ridge to exist.
4. **Scope hard** — text-only until text works.
5. **NO DELETES** — superseded data/artifacts are quarantined with notes, not removed.

---

## Success looks like

A recruiter or reviewer opens the repo and sees: tight PRD → reproducible Colab → baseline table where the simple model's number sits next to the deep model's → calibration plot → live demo. Whether the transformer wins or loses, the repo demonstrates **judgment** — which is the actual deliverable (DEFINITIVE_PLAN Option C: career-first).
