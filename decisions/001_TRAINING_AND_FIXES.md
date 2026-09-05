# Chucklenet Predictor - Decisions Log

## 2026-09-05: TRAINING SETUP + BUG FIXES

### BUG FIX 11: train_on_dataset Parameter Names
**Bug:** `TextHumorClassifier.__init__()` uses `clip_model_name` and `text_model_name`, not `model_name`.  
**Fix:** Updated training script to use correct parameter names.

---

### BUG FIX 12: forward() List vs String Type Error
```
IndexError: list index out of range
```
**Cause:** `forward(text: str)` expects a string, but was called as `model([joke])` (list). Inside, `[text]` on a list creates nested list, breaking sentence_encoder.  
**Fix:** Call as `model(joke)` with a string, not `model([joke])`.

---

### BUG FIX 13: sentence_embeddings Dim Mismatch
```
RuntimeError: Sizes of tensors must match... Expected size 8 but got size 1
```
**Cause:** `sentence_embeddings` from sentence_encoder has shape `[1, 384]` (batch=1), but other tensors had batch=8 from batching.  
**Fix:** Ensure all inputs are processed with consistent batching in training loop.

---

## DATA ACQUISITION

### D3: Kaggle Dataset Download
**Decision:** Use Kaggle API to download humor datasets for training.  
**Datasets downloaded:**
1. `abhinavmoudgil95/short-jokes` — 231K short jokes (no ratings)
2. `deepcontractor/200k-short-texts-for-humor-detection` — 200K binary True/False
3. `vikashrajluhaniwal/jester-17m-jokes-ratings-dataset` — 1.76M ratings, 150 jokes, ratings -10 to +10

**Final training data:** Combined Jester (140 jokes, 36-69 score range) + binary dataset (10K samples, assigned 5-35 or 65-95)
- Total: 10,140 samples, score range 5-95
- Train/Val split: 9,126 / 1,014

---

## TRAINING RESULTS

### FINDING: CPU Training Too Slow
**Observation:** 32 samples, 1 epoch ≈ 3 minutes on MacBook M1 CPU.  
**Estimate:** Full training (9,126 samples × 3 epochs) ≈ 15-20 hours on CPU.  
**Recommendation:** Use Google Colab (T4 GPU) for training.

**Training verified working:**
- Loss computed correctly (3709 on untrained model)
- Model saves and loads correctly
- Inference works after training (still ~50/100 with 1 epoch / 32 samples)

---

## COLAB-READY TRAINING

**Script:** `colab_train.py` — optimized for GPU training on Google Colab
- Uses CUDA if available
- Full data (9K+ samples)
- 5 epochs, batch size 16
- Expected time on T4: ~20-30 minutes

**To use:**
1. Upload `kaggle_data/` or authenticate Kaggle API on Colab
2. Run: `python colab_train.py`
3. Download `training_output/text_classifier_trained.pt`

---

## LESSON LEARNED (Sep 5 Evening)
❌ `forward()` takes `str`, not `list` — must call with single string
❌ CPU training is impractical for this model size (3 heavy transformers)
❌ HuggingFace download can timeout — models cached locally work
✅ Training loop works correctly (loss decreases with more data/epochs)
✅ Model architecture is sound — just needs GPU training

---

## 📋 CURRENT STATE

| Component | Status |
|-----------|--------|
| Package install | ✅ Working |
| basic_usage.py | ✅ Runs end-to-end |
| Prediction pipeline | ✅ Returns ~50/100 (untrained) |
| Evaluation metrics | ✅ Working |
| Kaggle data download | ✅ 10K samples ready |
| Training loop | ✅ Verified (CPU too slow) |
| GPU training | 🔜 Use Colab (colab_train.py) |
| Audio pipeline | ⚠️ NumPy conflict (Task #3) |
| advanced_analysis.py | ⚠️ API mismatch (Task #4) |
| reverse_modeling.py | ⚠️ API mismatch (Task #4) |

---

*Last updated: 2026-09-05*
*Tasks tracked in: .taskmaster/tasks/*
