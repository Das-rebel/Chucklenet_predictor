# Task ID: 5

**Title:** Implement context-aware reverse humor generation

**Status:** DEFERRED 2026-09-08 to v3 (decision D-R3; gated on v1 M4 calibration)

**Dependencies:** 2

**Priority:** low

**Description:** Implement reverse-model generation — given a target humor score, generate text that achieves it.

**Details:**
Options:
- A1: Gradient-based input optimization (perturb text to hit target score)
- A2: Template-based generation with scored components
- A3: LLM prompting with humor strength as reward signal

Depends on Task 2 (trained model) first.

**Test Strategy:**
`generate_funny_content(target_strength=75)` produces text scoring ~75±10.
