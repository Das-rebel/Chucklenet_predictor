# Task ID: 1

**Title:** Fix install bugs and get basic pipeline running

**Status:** done

**Dependencies:** None

**Priority:** critical

**Description:** Fix all installation errors and get basic_usage.py running end-to-end with prediction + evaluation.

**Details:**
Fixed 4 install bugs:
1. setup.py: Removed duplicate `keywords=` keyword arg (SyntaxError)
2. requirements.txt: Removed `conda>=23.0.0` (not a PyPI package)
3. __init__.py: Changed `model_name=` → `text_model_name=` / `whisper_model_name=` to match actual signatures
4. __init__.py: Fixed `strength_targets` and `test_targets` missing defaults (param-order SyntaxError)

Fixed model dimension mismatches:
1. LSTM hidden: 64→128 (wordplay/timing extractors)
2. CLIP dim: 384→512 (actual CLIP text embedding size)
3. text_fusion input: 768+384=1152 → 768+512=1280
4. humor_classifier input: 256 → 256+128+384=768 (fused_text + humor_fused + sentence_emb)

Fixed text processor:
- calculate_humor_score: Filter non-numeric values before summing

Fixed evaluation metrics:
- classification_report/confusion_matrix: Dynamic label handling (present_labels only)

**Test Strategy:**
`python examples/basic_usage.py` runs without error, outputs predictions + eval metrics.
