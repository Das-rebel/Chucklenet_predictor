# Decision Record 002 — Rethink from alpnewr findings

**Date:** 2026-09-08
**Trigger:** Review of alpnewr session (v20 gate forensics, D-GATE-V202) + Sep-7 ChuckleNet audit
**Documents produced:** `docs/PRD.md` (v1.0), `docs/VISION.md` (v1.0)

---

## D-R1: Never fabricate continuous labels from binary ones
**Finding:** Current `train.jsonl`/`val.jsonl` assign RANDOM strengths (65–95 funny / 5–35 not-funny) to the 200k binary dataset — training on injected noise. This is the same sin class as the core repo's pseudo-label circularity (0.975 claims → honest 0.31 after audit).
**Rule:** Binary labels feed only a binary aux head. Continuous training signal comes from real ratings only (Jester −10..+10 → 0–100; upvote-scored corpora as weak labels).
**Action:** Quarantine (not delete) current train/val files; M1 data rebuild per PRD §3.

## D-R2: Chucklenet_predictor = activation of ChuckleNet G4 + Product-3 track
**Finding:** alpnewr confirmed core research (word-level laughter positioning) is long-horizon: 620v corpus marker-poor (1.16% pos), honest IoU-F1@0.2 = 0.31 < 0.51 baseline, cascade dead at IoU 0.50.
**Decision:** This repo proceeds NOW on humor-strength regression with abundant rated data, independent of core's label problem. Bridges back later (v2 delivery scoring, full comedy stack).
**Status:** ACTIVE

## D-R3: v1 scope = text-only, single backbone
**Finding:** v0 architecture carried CLIP (vision model) + sentence-transformer concat + 5 untrained LSTM humor extractors + broken audio stack — zero training evidence for any of it. CPU training measured ~15–20 h (impractical); Colab GPU path verified instead.
**Decision:** v1 = RoBERTa regression head (+ optional binary aux). CLIP, sentence-encoder, humor-type heads moved behind experimental flag. Audio out of scope until M5.
**Action:** Rescope Task 002; defer Task 003 (audio) and Task 005 (reverse generation) to v3.

## D-R4: Evaluation protocol (inherits core G1)
**Rules:** GroupKFold by joke source (never random splits); primary metric Spearman ρ (not bin-accuracy); every published number cites a result JSON; transformer must beat TF-IDF+Ridge (M1) to ship.
**Targets:** M1 ρ ≥ 0.35; M2 ρ ≥ 0.55 (honest, may be revised after M1 evidence).

## D-R5: Jester reality check
**Finding:** Jester has only ~150–300 rated jokes (1.7M ratings concentrated on tiny joke pool; our 140-joke extract spans 36–69). Real ratings but weak coverage.
**Decision:** Jester = gold calibration/eval + primary training; upvote-scored joke corpora added for breadth as weak labels; per-source held-out reporting mandatory.

---

## Task impact

| Task | Change |
|---|---|
| 002 Train model | RESCOPE — blocked on M1 data rebuild (new prerequisite) |
| 003 Audio NumPy fix | DEFER to v2 (out of v1 scope) |
| 004 Examples API fix | Keep (small) |
| 005 Reverse generation | DEFER to v3 |
| 006 (new) Data rebuild | Per PRD §3 — Jester + upvote corpora + provenance table |
