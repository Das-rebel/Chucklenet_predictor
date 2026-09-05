"""
Model components for Chucklenet Predictor

This package contains the core model components for humor prediction and generation.

Available Models:
- text_humor_classifier.py: Text-based humor classification
- audio_laugh_detector.py: Audio-based laughter detection
- reverse_model.py: Reverse modeling for content generation
- fusion_engine.py: Multi-modal fusion and combination
"""

from .text_humor_classifier import TextHumorClassifier
from .audio_laugh_detector import AudioLaughDetector
from .reverse_model import ReverseHumorModel as ReverseModel
from .fusion_engine import FusionEngine

__all__ = [
    'TextHumorClassifier',
    'AudioLaughDetector',
    'ReverseModel',
    'FusionEngine',
]