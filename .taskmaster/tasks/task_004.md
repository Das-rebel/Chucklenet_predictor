# Task ID: 4

**Title:** Fix advanced_analysis.py and reverse_modeling.py for current API

**Status:** pending

**Dependencies:** 1

**Priority:** medium

**Description:** Both example scripts use old API (process_text, analyze_humor_components) that no longer exists.

**Details:**
advanced_analysis.py issues:
- `text_processor.process_text()` doesn't exist → use `predictor.predict_humor_strength(text=...)`
- `analyze_humor_components()` doesn't exist
- Result dict keys wrong (humor_strength → fusion.strength)

reverse_modeling.py issues:
- Imports `ReverseModelingEngine` which doesn't exist
- Should use `ReverseHumorModel.generate_funny_content()`

**Test Strategy:**
Both scripts run without import errors or API mismatches.
