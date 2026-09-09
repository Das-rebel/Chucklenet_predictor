# Chucklenet Predictor v2 — Audio Delivery Complement Plan

**Date:** 2026-09-08
**Purpose:** Map which ChuckleNet assets genuinely complement this project, and how.

---

## What ChuckleNet Has (real, validated)

| Asset | Status | Use for v2 |
|---|---|---|
| `fusion_mlp_v2.pt` (791-dim: WavLM+23-prosody → 1) | ✅ Real, validated F1=0.975 on 21K utterances | Delivery-feature extractor + quality scorer |
| 340 `.wav` comedy audio files (audio_final/) | ✅ Real | Extract delivery features from comedy audio |
| `laughter_model_v2.pkl` (38-dim energy LR) | ⚠️ Circular (energy features → energy pseudo-labels) | Structural reference only |
| Honest eval discipline (GroupKFold, Spearman, provenance JSONs) | ✅ Inherited | Use in v2 protocol |
| 23-dim prosody features (RMS, MFCC1-13, F0, ZCR, spectral) | ✅ Real (when extracted properly) | Input to delivery MLP |

---

## Complementary Architecture (v2)

```
TEXT PATH                    AUDIO PATH
────────                    ─────────
RoBERTa(text)               WavLM(audio) → 768-dim
  → text_strength 0-100       + 23-prosody → 791-dim
                               ↓
                            fusion_mlp_v2.pt (frozen) → delivery_score 0-1
                               ↓
                    [text_strength, delivery_score]
                               ↓
                    Late-fusion → final_strength 0-100
```

**Key insight:** `fusion_mlp_v2.pt` was trained to detect LAUGHTER (not rate FUNNYNESS). It outputs whether a segment has audience laughter, not how funny the content is. This is a DIFFERENT task — but its internal WavLM+prosody features still capture "engaging delivery." We use it as a delivery-quality proxy, NOT a humor scorer directly.

---

## Data Reality Check

| Source | Audio available? | Jester text matched? |
|---|---|---|
| 340 audio_final .wav | Yes — but which jokes do they correspond to? | Unknown (video IDs ≠ joke IDs) |
| Jester 140 rated jokes | No audio recorded | N/A |
| r/jokes / dadjokes | No audio | N/A |

**Honest problem:** Jester has no audio. r/jokes and dadjokes have no audio. The audio and text datasets have ZERO overlap.

**Solution options:**

### Option A: Pretrain delivery scorer on ChuckleNet audio → apply to any comedy audio
- Extract WavLM+prosody from 340 audio files
- Cluster into "engaging delivery" vs "flat delivery" profiles
- Use as delivery-quality score for ANY joke that has associated audio
- **Problem:** No labeled comedy audio with humor-strength scores — unsupervised only

### Option B: Synthetic prosody prediction from text features
- Train a text→prosody mapper: given text features (incongruity, etc.) predict expected delivery features
- Use mapped prosody features with fusion_mlp_v2.pt as proxy
- **Problem:** Circular — text features aren't trained for prosody prediction

### Option C (Recommended): Delivery diversity scorer
- Use fusion_mlp_v2.pt to score AUDIO VARIETY/ENERGY across segments
- High variety = engaging delivery = boost to text_score
- Works on ANY audio (standup clips, not Jester)
- No labeled (audio, strength) pairs needed — it's a prior boost, not direct scoring

---

## Honest v2 Roadmap

| Step | Action | Gate |
|---|---|---|
| v2.1 | Extract WavLM+prosody from 340 audio files → `audio_prosody_cache.npz` | Chunks exist |
| v2.2 | Compute per-video delivery score from fusion_mlp_v2.pt (frozen) | Scores computed |
| v2.3 | Categorize videos into high/medium/low delivery energy | Clusters valid |
| v2.4 | In text model: add text_excitement features (variance of surprise scores) as proxy | Ablation |
| v2.5 | Late fusion: combine text_strength × delivery_boost | Evaluate |

**v2 honest constraint:** No labeled (joke_text, audio, strength) triplets exist. Delivery scoring is inference-only prior, not supervised. Final combined score must beat text-only ρ on held-out Jester jokes to justify the complexity.

---

## Dependency on ChuckleNet core research

| ChuckleNet finding | Relevance to v2 |
|---|---|
| 620-video corpus marker-poor (1.16% positive) | Explains why laughter detection is hard; delivery features are sparse per video |
| WavLM+prosody F1=0.975 on laughter detection | Validates WavLM+prosody feature quality for ANY audio scoring |
| IoU-F1 ceiling 0.50 (cascade dead) | Word-level boundary precision isn't the goal — delivery PROFILE is |
| Honest eval (G1): provenance JSONs | Inherited fully for v2 protocol |

---

## What NOT to share / dead ends

- ❌ Reuse 620-video VTT labels for training (sparse, wrong task)
- ❌ Word-level cascade outputs (IoU ceiling, wrong architecture)
- ❌ Energy pseudo-label model (circular, meaningless)
- ❌ Random-split eval (violates G1)
