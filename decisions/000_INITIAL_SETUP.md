# Chucklenet Predictor - Decisions Log

## 2026-09-05: INITIAL SETUP + BUG FIXES

### RULE D1: ARCHITECTURE — MULTI-MODAL REVERSE MODELING

**Decision:** Invert traditional laughter detection to predict humor generation potential (0-100 scale) using text + audio fusion.

**Alternatives rejected:**
- Single-modal (text-only) — less expressive
- Rule-based only — lacks generalizability
- End-to-end generation — poor interpretability

**Outcome:** TextHumorClassifier + AudioLaughDetector + FusionEngine + ReverseModel

---

### RULE D2: PACKAGE STRUCTURE

**Decision:** Split into `models/`, `utils/`, `examples/` under `src/chucklenet_predictor/`.

**Alternatives rejected:**
- Flat structure — poor scalability
- Separate repos per component — coordination overhead

**Outcome:** Hierarchical package with `__init__.py` exposing unified API

---

### BUG FIX 1: setup.py SyntaxError
```
TypeError: __init__() got multiple values for keyword argument 'keywords'
```
**Cause:** Duplicate `keywords=` in setup() call.
**Fix:** Removed duplicate `keywords=` line.

---

### BUG FIX 2: requirements.txt Invalid Package
```
ERROR: Could not find a version that matches conda>=23.0.0
```
**Cause:** `conda` is not a PyPI package.
**Fix:** Removed `conda>=23.0.0` from requirements.txt.

---

### BUG FIX 3: __init__.py Param Name Mismatch
```
TypeError: __init__() got an unexpected keyword argument 'model_name'
```
**Cause:** `ChucklenetPredictor.__init__()` uses `text_model_name=` and `whisper_model_name=`, not `model_name=`.
**Fix:** Changed calls to match actual signature.

---

### BUG FIX 4: __init__.py Missing Defaults
```
SyntaxError: non-default argument follows default argument
```
**Cause:** `strength_targets` and `test_targets` params had no defaults but came after params with defaults.
**Fix:** Added appropriate default values.

---

### BUG FIX 5: LSTM Hidden Size Mismatch
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x64 and 128x101)
```
**Cause:** wordplay/timing LSTM extractors output 64-dim, but incongruity/surprise/relatability Linear extractors output 128-dim. All 5 must match for concatenation.
**Fix:** Changed LSTM hidden size from 64→128 for both wordplay and timing.

---

### BUG FIX 6: CLIP Dimension Wrong
**Cause:** CLIP text embedding assumed to be 384-dim, actual is 512-dim.
**Fix:** Updated `text_fusion` input from `768+384=1152` to `768+512=1280`.

---

### BUG FIX 7: humor_classifier Input Dimension Wrong
**Cause:** `humor_classifier` expected 256-dim input but `combined_fused` was 768-dim (fused_text=256 + humor_fused=128 + sentence_emb=384).
**Fix:** Changed `humor_classifier = nn.Linear(256, humor_classes)` → `nn.Linear(768, humor_classes)`.

---

### BUG FIX 8: text_processor Type Error
```
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```
**Cause:** `calculate_humor_score()` tried to sum category feature values, some of which were strings (`'neutral'`, `'label_0'`).
**Fix:** Filter to numeric values only: `numeric_values = [v for v in category_features.values() if isinstance(v, (int, float))]`.

---

### BUG FIX 9: evaluation_metrics Label Count Mismatch
```
ValueError: Number of classes, 3, does not match size of target_names, 5
```
**Cause:** `classification_report` and `confusion_matrix` used all 5 defined bin names but only 3 classes present in small test set.
**Fix:** Extract `present_labels = np.unique(np.concatenate([y_true_binned, y_pred_binned]))` before calling sklearn metrics.

---

### BUG FIX 10: evaluation_metrics Variable Scope
```
NameError: cannot access local variable 'present_labels' where it is not associated with a value
```
**Cause:** `present_labels` defined inside `if len(unique_classes) > 1:` block but used outside after the if/else.
**Fix:** Moved `present_labels` definition outside the if/else block.

---

### LESSON LEARNED (Sep 5)
❌ Don't trust assumed API signatures — always verify against actual implementation.
❌ LSTM output dim != hidden dim. Must be explicit about what each layer outputs.
❌ CLIP text embedding is 512-dim, not 384-dim.
❌ When concatenating tensors, sum all input dims correctly before the Linear layer.

✅ Systematic debugging: one bug at a time, verify fix before moving on.
✅ Basic usage example is the best test — runs full pipeline, catches dim mismatches.

---

## 📋 CURRENT STATE

| Component | Status |
|-----------|--------|
| Package install | ✅ Working |
| basic_usage.py | ✅ Runs end-to-end |
| Prediction pipeline | ✅ Returns ~50/100 (untrained) |
| Evaluation metrics | ✅ MAE 16, RMSE 18.4, Accuracy 40% |
| Model training | ⚠️ Pending (Task #2) |
| Audio pipeline | ⚠️ NumPy conflict (Task #3) |
| advanced_analysis.py | ⚠️ API mismatch (Task #4) |
| reverse_modeling.py | ⚠️ API mismatch (Task #4) |
| GitHub push | ⚠️ Pending |

---

## 🔜 NEXT DECISIONS NEEDED

### PENDING: How to train the model?
- Find/label humor dataset
- Fine-tune strategy (learning rate, epochs, etc.)

### PENDING: How to fix audio pipeline?
- Downgrade NumPy or refactor to librosa-only?

### PENDING: Deploy strategy?
- FastAPI, Streamlit, or CLI?

---

*Last updated: 2026-09-05*
*Tasks tracked in: .taskmaster/tasks/*
