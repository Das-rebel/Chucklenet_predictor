# Task ID: 3

**Title:** Fix audio pipeline (NumPy/Whisper conflict)

**Status:** pending

**Dependencies:** None

**Priority:** medium

**Description:** Whisper fails with "Numba needs NumPy 2.2 or less. Got NumPy 2.5". Fix audio feature extraction.

**Details:**
Error: `Failed to load Whisper model: Numba needs NumPy 2.2 or less. Got NumPy 2.5`

Options:
- A1: Downgrade NumPy to 2.0 or 1.x
- A2: Use CPU-only audio features (librosa) instead of Whisper
- A3: Refactor AudioLaughDetector to use acoustic features only (MFCCs, F0, RMS)

**Test Strategy:**
Audio pipeline loads without error, processes audio file successfully.
