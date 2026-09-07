# Task 006 — Data Rebuild (M1): Real ratings only

**Priority:** CRITICAL (blocks 002) · **Created:** 2026-09-08 · **Status:** pending
**Source:** decisions/002_RETHINK_ALPNEWR_FINDINGS.md (D-R1, D-R5), PRD §3

## Goal
Replace noise-injected train/val with real-rated data.

## Steps
1. Jester 1+2+3: all jokes with ≥50 ratings → per-joke mean/std → normalize to 0–100
2. Add upvote-scored joke corpus (Kaggle Reddit jokes) as weak labels
3. Quarantine old train.jsonl/val.jsonl (rename *.noise_quarantined, keep — NO DELETES)
4. Write data/REBUILD_NOTES.md: provenance table (source, count, label type, hash)
5. GroupKFold split by source; save data/train_real.jsonl, data/val_real.jsonl

## Acceptance
- [ ] Strength range from real ratings (expect wider than 36–69)
- [ ] Provenance table complete
- [ ] Old data quarantined with README note
