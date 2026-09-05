"""
Reverse Humor Model - Main System Architecture

This module implements the core reverse funny strength prediction system that
inverts laughter detection models to predict humor potential (0-100% funny strength).

Author: Subho Das
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from .text_humor_classifier import TextHumorClassifier
from .audio_laugh_detector import AudioLaughDetector
from .fusion_engine import FusionEngine


class ReverseHumorModel(nn.Module):
    """
    Main reverse funny strength prediction system.
    
    This model implements the core innovation of inverting traditional laughter detection
    to predict humor generation potential using a multi-modal approach.
    
    Architecture:
    1. Text Analysis: CLIP + Transformers for semantic humor understanding
    2. Audio Analysis: Whisper + Custom features for laughter pattern recognition
    3. Fusion Engine: Multi-modal fusion for comprehensive humor assessment
    4. Reverse Modeling: Generative optimization for humor content creation
    """
    
    def __init__(
        self,
        text_model_name: str = "roberta-base",
        audio_model_name: str = "openai/whisper-small",
        device: str = "auto",
        fusion_method: str = "cross_modal_attention"
    ):
        super().__init__()
        
        self.device = self._get_device(device)
        self.fusion_method = fusion_method
        
        # Initialize text humor classifier
        self.text_classifier = TextHumorClassifier(
            clip_model_name="openai/clip-vit-base-patch32",
            text_model_name=text_model_name,
            device=device
        )
        
        # Initialize audio laugh detector
        self.audio_detector = AudioLaughDetector(
            whisper_model_name=audio_model_name,
            device=device
        )
        
        # Initialize fusion engine
        self.fusion_engine = FusionEngine(
            fusion_method=fusion_method,
            device=device
        )
        
        # Reverse modeling components
        self.reverse_modeling = ReverseModelingEngine(device=device)
        
        # Final prediction head
        self.final_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def forward(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        target_strength: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for reverse humor prediction.
        
        Args:
            text: Input text string (optional)
            audio_path: Path to audio file (optional)
            target_strength: Target humor strength for reverse modeling (0-100)
            
        Returns:
            Dictionary containing:
            - humor_strength: Predicted humor strength (0-100%)
            - confidence: Confidence in prediction
            - reverse_features: Reverse modeling features
            - multi_modal_features: Fused multi-modal features
        """
        # Extract text features
        text_features = None
        if text is not None:
            text_results = self.text_classifier(text)
            text_features = text_results['humor_scores']
            
            # Add text embedding features
            text_features = torch.cat([
                text_results['clip_features'],
                text_results['text_features'],
                text_results['sentence_embeddings'],
            ], dim=1)
        
        # Extract audio features
        audio_features = None
        if audio_path is not None:
            audio_results = self.audio_detector(audio_path)
            audio_features = audio_results['reverse_score']
            
            # Add audio feature details
            audio_features = torch.cat([
                audio_results['laugh_prob'],
                audio_results['reverse_score'],
                audio_results['final_score'],
            ], dim=1)
        
        # Multi-modal fusion
        fused_features = self.fusion_engine(text_features, audio_features)
        
        # Reverse modeling for humor generation potential
        if target_strength is not None and text is not None:
            reverse_features = self.reverse_modeling(
                fused_features, 
                target_strength=target_strength,
                text=text
            )
        else:
            reverse_features = fused_features
        
        # Final humor strength prediction
        humor_strength = self.final_predictor(reverse_features) * 100  # Convert to percentage
        confidence = self.confidence_estimator(reverse_features)
        
        return {
            'humor_strength': humor_strength,
            'confidence': confidence,
            'reverse_features': reverse_features,
            'multi_modal_features': fused_features,
            'text_features': text_features,
            'audio_features': audio_features,
        }
    
    def predict_humor_strength(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Predict humor strength as a percentage (0-100%) and confidence.
        
        Args:
            text: Input text string (optional)
            audio_path: Path to audio file (optional)
            
        Returns:
            Tuple of (humor_strength, confidence)
        """
        self.eval()
        with torch.no_grad():
            results = self(text=text, audio_path=audio_path)
            humor_strength = results['humor_strength'].cpu().numpy()[0][0]
            confidence = results['confidence'].cpu().numpy()[0][0]
            
        return float(humor_strength), float(confidence)
    
    def generate_funny_content(
        self,
        target_strength: float,
        prompt: str,
        max_iterations: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate humor-optimized content using reverse modeling.
        
        Args:
            target_strength: Target humor strength (0-100%)
            prompt: Text prompt for content generation
            max_iterations: Maximum optimization iterations
            temperature: Generation temperature
            
        Returns:
            Generated humor-optimized content
        """
        self.eval()
        with torch.no_grad():
            # Use reverse modeling to generate content
            optimized_content = self.reverse_modeling.generate_content(
                target_strength=target_strength,
                prompt=prompt,
                max_iterations=max_iterations,
                temperature=temperature
            )
            
        return optimized_content
    
    def analyze_humor_composition(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Analyze the composition of humor in the input.
        
        Args:
            text: Input text string (optional)
            audio_path: Path to audio file (optional)
            
        Returns:
            Dictionary with detailed humor composition analysis
        """
        self.eval()
        
        analysis = {}
        
        if text is not None:
            # Text humor type analysis
            text_analysis = self.text_classifier.analyze_humor_types(text)
            analysis['text_humor_types'] = text_analysis
            
            # Text feature analysis
            text_results = self.text_classifier(text)
            analysis['text_features'] = {
                'clip_embedding_norm': torch.norm(text_results['clip_features']).item(),
                'sentence_embedding_norm': torch.norm(text_results['sentence_embeddings']).item(),
                'confidence': text_results['confidence'].item(),
            }
        
        if audio_path is not None:
            # Audio analysis
            audio_analysis = self.audio_detector.analyze_audio_features(audio_path)
            analysis['audio_analysis'] = audio_analysis
        
        # Multi-modal analysis
        results = self(text=text, audio_path=audio_path)
        analysis['fusion'] = {
            'humor_strength': results['humor_strength'].item(),
            'confidence': results['confidence'].item(),
            'feature_norm': torch.norm(results['multi_modal_features']).item(),
        }
        
        return analysis
    
    def batch_analyze(
        self,
        texts: List[str],
        audio_paths: Optional[List[str]] = None
    ) -> List[Dict[str, Union[float, Dict[str, float]]]]:
        """
        Batch analysis for multiple texts and audio files.
        
        Args:
            texts: List of input text strings
            audio_paths: Optional list of audio file paths
            
        Returns:
            List of analysis results for each input
        """
        results = []
        
        for i, text in enumerate(texts):
            audio_path = audio_paths[i] if audio_paths and i < len(audio_paths) else None
            analysis = self.analyze_humor_composition(text=text, audio_path=audio_path)
            results.append(analysis)
            
        return results
    
    def get_humor_recommendations(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        target_strength: float = 75.0
    ) -> Dict[str, List[str]]:
        """
        Get recommendations for improving humor based on current analysis.
        
        Args:
            text: Input text string (optional)
            audio_path: Path to audio file (optional)
            target_strength: Target humor strength for recommendations
            
        Returns:
            Dictionary with humor improvement recommendations
        """
        analysis = self.analyze_humor_composition(text=text, audio_path=audio_path)
        
        recommendations = {
            'text_improvements': [],
            'audio_improvements': [],
            'general_tips': [],
        }
        
        # Analyze current humor strength
        current_strength = analysis.get('fusion', {}).get('humor_strength', 0)
        
        if text is not None:
            # Text-based recommendations
            text_types = analysis.get('text_humor_types', {})
            
            if current_strength < target_strength:
                recommendations['text_improvements'].extend([
                    "Add more surprising elements to increase incongruity",
                    "Use relatable scenarios to improve audience connection",
                    "Incorporate wordplay or puns for added humor",
                    "Consider timing adjustments for better comedic effect"
                ])
            else:
                recommendations['text_improvements'].extend([
                    "Great humor score! Maintain current approach",
                    "Consider variations to keep content fresh",
                    "Experiment with different humor styles"
                ])
        
        if audio_path is not None:
            # Audio-based recommendations
            audio_analysis = analysis.get('audio_analysis', {})
            
            if current_strength < target_strength:
                recommendations['audio_improvements'].extend([
                    "Increase vocal energy and expressiveness",
                    "Add laughter cues at appropriate moments",
                    "Improve timing and pacing for comedic effect",
                    "Use vocal modulation to convey humor"
                ])
        
        # General humor recommendations
        recommendations['general_tips'].extend([
            "Know your audience and tailor humor appropriately",
            "Balance surprise with relatability",
            "Use the rule of three for comedic structure",
            "Practice timing and delivery",
            "Read the room and adapt accordingly"
        ])
        
        return recommendations
    
    def save_pretrained(self, save_directory: str):
        """Save the complete model to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model components
        torch.save(self.state_dict(), os.path.join(save_directory, 'reverse_humor_model.pth'))
        
        # Save individual components
        self.text_classifier.save_pretrained(os.path.join(save_directory, 'text_classifier'))
        self.audio_detector.save_pretrained(os.path.join(save_directory, 'audio_detector'))
        self.fusion_engine.save_pretrained(os.path.join(save_directory, 'fusion_engine'))
        self.reverse_modeling.save_pretrained(os.path.join(save_directory, 'reverse_modeling'))
        
        # Save configuration
        config = {
            'text_model_name': 'roberta-base',
            'audio_model_name': 'openai/whisper-small',
            'device': str(self.device),
            'fusion_method': self.fusion_method,
        }
        
        import json
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load a pretrained model from the specified directory."""
        import json
        
        # Load configuration
        config_path = os.path.join(load_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model with loaded configuration
        model = cls(
            text_model_name=config.get('text_model_name', 'roberta-base'),
            audio_model_name=config.get('audio_model_name', 'openai/whisper-small'),
            device=config.get('device', 'auto'),
            fusion_method=config.get('fusion_method', 'cross_modal_attention'),
            **kwargs
        )
        
        # Load pretrained weights
        model.load_state_dict(torch.load(os.path.join(load_directory, 'reverse_humor_model.pth')))
        
        return model


class ReverseModelingEngine(nn.Module):
    """
    Reverse modeling engine for humor content generation optimization.
    
    Uses generative approaches to create content that maximizes humor potential
    based on target humor strength specifications.
    """
    
    def __init__(self, device: str = "auto"):
        super().__init__()
        
        self.device = self._get_device(device)
        
        # Humor optimization layers
        self.humor_optimizer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Content generation head
        self.content_generator = nn.Sequential(
            nn.Linear(64 + 1, 32),  # Features + target strength
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Reinforcement learning components
        self.reward_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def generate_content(
        self,
        target_strength: float,
        prompt: str,
        max_iterations: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate humor-optimized content using reverse modeling.
        
        Args:
            target_strength: Target humor strength (0-100%)
            prompt: Text prompt for content generation
            max_iterations: Maximum optimization iterations
            temperature: Generation temperature
            
        Returns:
            Generated humor-optimized content
        """
        # This is a placeholder for the actual content generation
        # In a real implementation, this would use diffusion models or 
        # other generative techniques from the AI-research-SKILLS repository
        
        # For now, return a simple optimization-based response
        optimized_content = self._optimize_content(prompt, target_strength, max_iterations)
        
        return optimized_content
    
    def _optimize_content(
        self, 
        prompt: str, 
        target_strength: float, 
        max_iterations: int
    ) -> str:
        """
        Optimize content to achieve target humor strength.
        
        This is a simplified version - real implementation would use
        more sophisticated optimization techniques.
        """
        # Start with original prompt
        content = prompt
        
        # Simple optimization loop (placeholder)
        for i in range(max_iterations):
            # Calculate current humor score
            # (This would use the actual model in practice)
            current_strength = np.random.uniform(0, 100)  # Placeholder
            
            # Check if target achieved
            if abs(current_strength - target_strength) < 5:
                break
            
            # Apply optimization (placeholder)
            if current_strength < target_strength:
                # Add humor elements
                content = f"{content} (with added humor)"
            else:
                # Reduce humor elements
                content = f"{content} (more serious)"
        
        return content
    
    def forward(self, features: torch.Tensor, target_strength: Optional[float] = None) -> torch.Tensor:
        """Forward pass for reverse modeling."""
        humor_score = self.humor_optimizer(features)
        
        if target_strength is not None:
            # Apply reinforcement learning for target optimization
            target_tensor = torch.tensor([[target_strength / 100.0]], device=self.device)
            combined = torch.cat([features, target_tensor], dim=1)
            optimized = self.content_generator(combined)
            return optimized
        else:
            return humor_score