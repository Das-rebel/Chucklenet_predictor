"""
Chucklenet Predictor - Reverse Funny Strength Prediction Model

A comprehensive AI system for predicting and generating humor content with 
quantitative humor strength ratings and reverse modeling capabilities.

Author: Subho Das
"""

__version__ = "1.0.0"
__author__ = "Subho Das"
__email__ = "subho.das@example.com"
__license__ = "MIT"

import sys
from pathlib import Path

# Add the package root to Python path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))

# Import main components
from .models.reverse_model import ReverseHumorModel as ReverseModel
from .models.text_humor_classifier import TextHumorClassifier
from .models.audio_laugh_detector import AudioLaughDetector
from .models.fusion_engine import FusionEngine
from .utils.text_processor import TextProcessor
from .utils.audio_processor import AudioProcessor
from .utils.evaluation_metrics import EvaluationMetrics

# Main interface classes
class ChucklenetPredictor:
    """
    Main interface for Chucklenet Predictor.
    
    This class provides a unified interface for humor prediction and generation
    capabilities, combining text and audio analysis with reverse modeling.
    """
    
    def __init__(self, 
                 text_model_name="roberta-base",
                 audio_model_name="openai/whisper-small",
                 device="auto",
                 **kwargs):
        """
        Initialize Chucklenet Predictor.
        
        Args:
            text_model_name: Name of the text model to use
            audio_model_name: Name of the audio model to use
            device: Device to run on ('cpu', 'cuda', 'auto')
            **kwargs: Additional configuration options
        """
        self.device = self._determine_device(device)
        self.config = kwargs
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.evaluator = EvaluationMetrics()
        
        # Initialize models
        self.text_classifier = TextHumorClassifier(
            text_model_name=text_model_name,
            device=self.device,
            **kwargs
        )
        
        self.audio_detector = AudioLaughDetector(
            whisper_model_name=audio_model_name,
            device=self.device,
            **kwargs
        )
        
        self.fusion_engine = FusionEngine(
            fusion_method='cross_modal_attention',
            device=self.device,
            **kwargs
        )
        
        # Initialize reverse model if requested
        self.reverse_model = None
        if kwargs.get('enable_reverse_model', True):
            self.reverse_model = ReverseModel(
                text_model_name=text_model_name,
                audio_model_name=audio_model_name,
                device=self.device,
                **kwargs
            )
        
        self.is_trained = False
    
    def _determine_device(self, device):
        """Determine the appropriate device for computation."""
        import torch
        
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def predict_humor_strength(self, text=None, audio=None, return_confidence=True):
        """
        Predict humor strength for given text and/or audio.
        
        Args:
            text: Text content to analyze
            audio: Audio file path to analyze
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with prediction results
        """
        if text is None and audio is None:
            raise ValueError("Either text or audio must be provided")
        
        results = {}
        
        # Text analysis
        if text:
            humor_strength, confidence = self.text_classifier.predict_humor_strength(text)
            results['text'] = {'strength': humor_strength, 'confidence': confidence}
        
        # Audio analysis
        if audio:
            audio_features = self.audio_processor.extract_audio_features(audio)
            results['audio'] = audio_features
        
        # Fusion analysis
        if results.get('text') and results.get('audio'):
            fusion_strength = (results['text']['strength'] + 0.5) / 2  # Simple average for now
            results['fusion'] = {'strength': fusion_strength, 'confidence': 0.7}
        elif results.get('text'):
            results['fusion'] = results['text']
        elif results.get('audio'):
            # For audio only, estimate from features
            results['fusion'] = {'strength': 50.0, 'confidence': 0.5}  # Default for audio
        
        return results
    
    def analyze_humor_composition(self, text=None, audio=None):
        """
        Analyze humor composition and detailed characteristics.
        
        Args:
            text: Text content to analyze
            audio: Audio file path to analyze
            
        Returns:
            Detailed humor composition analysis
        """
        if text is None and audio is None:
            raise ValueError("Either text or audio must be provided")
        
        # Get base prediction
        prediction = self.predict_humor_strength(text=text, audio=audio)
        
        # Add detailed composition analysis
        if text and 'text' in prediction:
            text_types = self.text_classifier.analyze_humor_types(text)
            prediction['text']['humor_types'] = text_types
        
        if audio and 'audio' in prediction:
            audio_features = self.audio_processor.extract_audio_features(audio)
            prediction['audio']['acoustic_features'] = audio_features
        
        return prediction
    
    def generate_funny_content(self, target_strength, prompt, **kwargs):
        """
        Generate content to achieve specific humor strength target.
        
        Args:
            target_strength: Target humor strength (0-100)
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated content
        """
        if self.reverse_model is None:
            raise ValueError("Reverse modeling not enabled")
        
        return self.reverse_model.generate_funny_content(
            target_strength=target_strength,
            prompt=prompt,
            **kwargs
        )
    
    def batch_analyze(self, texts=None, audio_files=None, batch_size=32):
        """
        Batch analyze multiple texts and/or audio files.
        
        Args:
            texts: List of texts to analyze
            audio_files: List of audio file paths to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of analysis results
        """
        results = []
        
        if texts:
            text_results = self.text_classifier.batch_predict(texts, batch_size)
            results.extend([{'text': result} for result in text_results])
        
        if audio_files:
            audio_results = self.audio_processor.batch_analyze(audio_files, batch_size)
            results.extend([{'audio': result} for result in audio_results])
        
        return results
    
    def train_on_dataset(self, texts=None, audio_files=None, strength_targets=None, **kwargs):
        """
        Train models on custom dataset.
        
        Args:
            texts: Training texts
            audio_files: Training audio files
            strength_targets: Target strength values
            **kwargs: Training parameters
        """
        if texts:
            self.text_classifier.train_on_dataset(texts, strength_targets, **kwargs)
        
        if audio_files:
            self.audio_detector.train_on_dataset(audio_files, strength_targets, **kwargs)
        
        self.is_trained = True
    
    def evaluate(self, test_texts=None, test_audio=None, test_targets=None):
        """
        Evaluate model performance.
        
        Args:
            test_texts: Test texts
            test_audio: Test audio files
            test_targets: Target strength values
            
        Returns:
            Evaluation metrics
        """
        predictions = []
        
        if test_texts:
            text_predictions = self.text_classifier.batch_predict(test_texts)
            predictions.extend([pred[0] for pred in text_predictions])
        
        if test_audio:
            audio_predictions = self.audio_processor.batch_analyze(test_audio)
            predictions.extend([pred.get('overall_quality', 50.0) for pred in audio_predictions])
        
        return self.evaluator.evaluate_humor_classification(test_targets, predictions)

# Convenience functions for quick usage
def predict_humor(text=None, audio=None, model_name="roberta-base"):
    """
    Quick humor prediction function.
    
    Args:
        text: Text content to analyze
        audio: Audio file path to analyze
        model_name: Model to use for prediction
        
    Returns:
        Humor strength prediction
    """
    predictor = ChucklenetPredictor(text_model_name=model_name)
    return predictor.predict_humor_strength(text=text, audio=audio)

def analyze_humor_types(text):
    """
    Quick humor type analysis.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Humor type analysis
    """
    processor = TextProcessor()
    return processor.analyze_humor_types(text)

# Reverse modeling functions
def generate_funny_content(target_strength, prompt, model_name="roberta-base"):
    """
    Quick content generation with target humor strength.
    
    Args:
        target_strength: Target humor strength (0-100)
        prompt: Input prompt
        model_name: Model to use for generation
        
    Returns:
        Generated content
    """
    predictor = ChucklenetPredictor(text_model_name=model_name, enable_reverse_model=True)
    return predictor.generate_funny_content(
        target_strength=target_strength,
        prompt=prompt
    )

# Legacy compatibility aliases
ReverseHumorModel = ChucklenetPredictor
TextHumorClassifier = TextHumorClassifier
AudioLaughDetector = AudioLaughDetector
FusionEngine = FusionEngine

__all__ = [
    'ChucklenetPredictor',
    'ReverseHumorModel',
    'TextHumorClassifier', 
    'AudioLaughDetector',
    'FusionEngine',
    'TextProcessor',
    'AudioProcessor',
    'EvaluationMetrics',
    'predict_humor',
    'analyze_humor_types',
    'generate_funny_content',
]