"""
Examples for Chucklenet Predictor

This package contains comprehensive examples demonstrating the capabilities of the Chucklenet Predictor.

Available Examples:
- basic_usage.py: Basic functionality demonstration
- advanced_analysis.py: Advanced analysis techniques
- reverse_modeling.py: Reverse modeling capabilities  
- training_pipeline.py: Model training and fine-tuning

Author: Subho Das
"""

__version__ = "1.0.0"

from .basic_usage import main as basic_usage_main
from .advanced_analysis import main as advanced_analysis_main
from .reverse_modeling import main as reverse_modeling_main
from .training_pipeline import main as training_pipeline_main

__all__ = [
    'basic_usage_main',
    'advanced_analysis_main', 
    'reverse_modeling_main',
    'training_pipeline_main'
]