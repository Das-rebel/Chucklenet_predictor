"""
Fusion Engine - Multi-Modal Humor Analysis

This module implements the fusion engine that combines text and audio features
for comprehensive humor analysis.

Author: Subho Das
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing text and audio features.
    
    Uses self-attention and cross-attention to model relationships between
    different modalities and their contributions to humor detection.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Projection layers
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Self-attention layers
        self.text_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention layers
        self.text_to_audio_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.text_norm1 = nn.LayerNorm(hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim)
        self.audio_norm1 = nn.LayerNorm(hidden_dim)
        self.audio_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward networks
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.audio_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            text_features: Text feature tensor [batch_size, text_seq_len, text_dim]
            audio_features: Audio feature tensor [batch_size, audio_seq_len, audio_dim]
            
        Returns:
            Dictionary with fused features and attention weights
        """
        results = {}
        
        # Project features to hidden dimension
        if text_features is not None:
            text_proj = self.text_proj(text_features)  # [batch, seq_len, hidden_dim]
        if audio_features is not None:
            audio_proj = self.audio_proj(audio_features)  # [batch, seq_len, hidden_dim]
        
        # Self-attention processing
        if text_features is not None:
            text_self_attn_out, text_self_weights = self.text_self_attn(
                text_proj, text_proj, text_proj
            )
            text_self_attn_out = self.text_norm1(text_proj + text_self_attn_out)
            text_ffn_out = self.text_ffn(text_self_attn_out)
            text_final = self.text_norm2(text_self_attn_out + text_ffn_out)
            results['text_features'] = text_final
        
        if audio_features is not None:
            audio_self_attn_out, audio_self_weights = self.audio_self_attn(
                audio_proj, audio_proj, audio_proj
            )
            audio_self_attn_out = self.audio_norm1(audio_proj + audio_self_attn_out)
            audio_ffn_out = self.audio_ffn(audio_self_attn_out)
            audio_final = self.audio_norm2(audio_self_attn_out + audio_ffn_out)
            results['audio_features'] = audio_final
        
        # Cross-attention fusion
        if text_features is not None and audio_features is not None:
            # Text to audio attention
            text_to_audio, text_to_audio_attn = self.text_to_audio_attn(
                audio_proj, text_proj, audio_proj
            )
            
            # Audio to text attention
            audio_to_text, audio_to_text_attn = self.audio_to_text_attn(
                text_proj, audio_proj, text_proj
            )
            
            # Fuse modalities
            fused_features = (text_final + audio_final + text_to_audio + audio_to_text) / 4
            
            results['fused_features'] = fused_features
            results['attention_weights'] = {
                'text_self': text_self_weights,
                'audio_self': audio_self_weights,
                'text_to_audio': text_to_audio_attn,
                'audio_to_text': audio_to_text_attn,
            }
        
        return results


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion engine that combines text and audio features
    using various fusion strategies.
    """
    
    def __init__(
        self,
        fusion_method: str = "cross_modal_attention",
        text_dim: int = 768,
        audio_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        
        # Choose fusion method
        if fusion_method == "cross_modal_attention":
            self.fusion_layer = CrossModalAttention(
                text_dim=text_dim,
                audio_dim=audio_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif fusion_method == "concatenation":
            self.fusion_dim = text_dim + audio_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        elif fusion_method == "attention_weighted":
            self.attention_weights = nn.Sequential(
                nn.Linear(text_dim + audio_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),  # weights for text and audio
                nn.Softmax(dim=-1)
            )
            self.fusion_dim = text_dim + audio_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Humor-specific processing layers
        self.humor_analysis = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Final fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal fusion.
        
        Args:
            text_features: Text feature tensor
            audio_features: Audio feature tensor
            
        Returns:
            Dictionary with fused features and analysis results
        """
        results = {}
        
        # Handle missing modalities
        if text_features is None and audio_features is None:
            raise ValueError("At least one modality (text or audio) must be provided")
        
        if text_features is None:
            # Use only audio features
            if hasattr(self, 'fusion_layer') and isinstance(self.fusion_layer, CrossModalAttention):
                audio_results = self.fusion_layer(None, audio_features)
                fused_features = audio_results['audio_features']
            else:
                fused_features = audio_features
        
        elif audio_features is None:
            # Use only text features
            if hasattr(self, 'fusion_layer') and isinstance(self.fusion_layer, CrossModalAttention):
                text_results = self.fusion_layer(text_features, None)
                fused_features = text_results['text_features']
            else:
                fused_features = text_features
        
        else:
            # Both modalities available - apply fusion method
            if self.fusion_method == "cross_modal_attention":
                fusion_results = self.fusion_layer(text_features, audio_features)
                fused_features = fusion_results['fused_features']
                results['attention_weights'] = fusion_results['attention_weights']
            
            elif self.fusion_method == "concatenation":
                # Concatenate features
                combined = torch.cat([text_features, audio_features], dim=-1)
                fused_features = self.fusion_layer(combined)
            
            elif self.fusion_method == "attention_weighted":
                # Compute attention weights
                combined_features = torch.cat([text_features, audio_features], dim=-1)
                weights = self.attention_weights(combined_features)
                
                # Apply weighted fusion
                text_weight = weights[..., 0:1]
                audio_weight = weights[..., 1:2]
                
                weighted_text = text_features * text_weight
                weighted_audio = audio_features * audio_weight
                
                fused_features = self.fusion_layer(torch.cat([weighted_text, weighted_audio], dim=-1))
                
                results['attention_weights'] = {
                    'text_weight': text_weight,
                    'audio_weight': audio_weight
                }
        
        # Humor analysis
        humor_features = self.humor_analysis(fused_features)
        humor_score = self.fusion_head(humor_features)
        
        results.update({
            'fused_features': fused_features,
            'humor_features': humor_features,
            'humor_score': humor_score * 100,  # Convert to percentage
            'fusion_method': self.fusion_method,
        })
        
        return results


class FusionEngine(nn.Module):
    """
    Complete fusion engine for multi-modal humor analysis.
    
    Combines text and audio features using various fusion strategies
    and provides comprehensive humor analysis capabilities.
    """
    
    def __init__(
        self,
        fusion_method: str = "cross_modal_attention",
        text_dim: int = 768,
        audio_dim: int = 256,
        device: str = "auto"
    ):
        super().__init__()
        
        self.device = self._get_device(device)
        self.fusion_method = fusion_method
        
        # Initialize multi-modal fusion
        self.multimodal_fusion = MultiModalFusion(
            fusion_method=fusion_method,
            text_dim=text_dim,
            audio_dim=audio_dim
        )
        
        # Feature normalization layers
        self.text_norm = nn.LayerNorm(text_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)
        
        # Late fusion components
        self.late_fusion = nn.Sequential(
            nn.Linear(text_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Early fusion components
        self.early_fusion = nn.Sequential(
            nn.Linear(text_dim + audio_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Hybrid fusion (combines early and late fusion)
        self.hybrid_fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),  # Combined from early and late fusion
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def normalize_features(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        """Normalize input features."""
        normalized = {}
        
        if text_features is not None:
            normalized['text'] = self.text_norm(text_features)
        else:
            normalized['text'] = None
            
        if audio_features is not None:
            normalized['audio'] = self.audio_norm(audio_features)
        else:
            normalized['audio'] = None
            
        return normalized
    
    def early_fusion_method(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Early fusion method - combine features before processing."""
        if text_features is None and audio_features is None:
            raise ValueError("At least one modality must be provided")
        
        if text_features is None:
            return audio_features
        elif audio_features is None:
            return text_features
        else:
            combined = torch.cat([text_features, audio_features], dim=-1)
            return self.early_fusion(combined)
    
    def late_fusion_method(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Late fusion method - process features separately then combine."""
        if text_features is None and audio_features is None:
            raise ValueError("At least one modality must be provided")
        
        text_score = text_features.mean(dim=-1, keepdim=True) if text_features is not None else torch.zeros((1, 1), device=self.device)
        audio_score = audio_features.mean(dim=-1, keepdim=True) if audio_features is not None else torch.zeros((1, 1), device=self.device)
        
        combined = torch.cat([text_score, audio_score], dim=-1)
        return self.late_fusion(combined)
    
    def hybrid_fusion_method(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Hybrid fusion method - combine early and late fusion results."""
        # Early fusion
        early_result = self.early_fusion_method(text_features, audio_features)
        
        # Late fusion
        late_result = self.late_fusion_method(text_features, audio_features)
        
        # Multi-modal fusion
        normalized = self.normalize_features(text_features, audio_features)
        multimodal_result = self.multimodal_fusion(normalized['text'], normalized['audio'])
        
        # Combine all fusion methods
        combined_features = torch.cat([
            early_result,
            late_result,
            multimodal_result['humor_score'] / 100.0  # Normalize to 0-1
        ], dim=-1)
        
        hybrid_result = self.hybrid_fusion(combined_features)
        
        return {
            'early_fusion': early_result,
            'late_fusion': late_result,
            'multimodal_fusion': multimodal_result,
            'hybrid_fusion': hybrid_result,
        }
    
    def forward(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for fusion engine.
        
        Args:
            text_features: Text feature tensor
            audio_features: Audio feature tensor
            
        Returns:
            Dictionary with all fusion results and analysis
        """
        results = {}
        
        # Normalize features
        normalized = self.normalize_features(text_features, audio_features)
        
        # Apply fusion methods
        fusion_results = self.hybrid_fusion_method(normalized['text'], normalized['audio'])
        results.update(fusion_results)
        
        # Combine all fusion scores
        all_scores = torch.stack([
            fusion_results['early_fusion'],
            fusion_results['late_fusion'],
            fusion_results['multimodal_fusion']['humor_score'] / 100.0,
            fusion_results['hybrid_fusion']
        ], dim=-1)
        
        # Final weighted combination
        final_score = all_scores.mean(dim=-1) * 100  # Convert to percentage
        
        # Calculate confidence based on agreement between methods
        confidence = 1.0 - torch.std(all_scores, dim=-1).mean()
        
        results.update({
            'final_humor_score': final_score,
            'confidence': confidence,
            'fusion_results': fusion_results,
            'normalized_features': normalized,
        })
        
        return results
    
    def get_fusion_weights(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        Get the relative importance of each fusion method.
        
        Args:
            text_features: Text feature tensor
            audio_features: Audio feature tensor
            
        Returns:
            Dictionary with fusion method weights
        """
        fusion_results = self.hybrid_fusion_method(text_features, audio_features)
        
        weights = {
            'early_fusion': fusion_results['early_fusion'].mean().item(),
            'late_fusion': fusion_results['late_fusion'].mean().item(),
            'multimodal_fusion': fusion_results['multimodal_fusion']['humor_score'].mean().item() / 100.0,
            'hybrid_fusion': fusion_results['hybrid_fusion'].mean().item(),
        }
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def analyze_modalities(self, text_features: Optional[torch.Tensor], audio_features: Optional[torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Analyze the contribution of each modality to humor detection.
        
        Args:
            text_features: Text feature tensor
            audio_features: Audio feature tensor
            
        Returns:
            Dictionary with modality analysis
        """
        results = {}
        
        if text_features is not None:
            text_analysis = {
                'feature_dimension': text_features.shape[-1],
                'feature_norm': torch.norm(text_features).item(),
                'mean_activation': text_features.mean().item(),
                'std_activation': text_features.std().item(),
            }
            results['text_analysis'] = text_analysis
        
        if audio_features is not None:
            audio_analysis = {
                'feature_dimension': audio_features.shape[-1],
                'feature_norm': torch.norm(audio_features).item(),
                'mean_activation': audio_features.mean().item(),
                'std_activation': audio_features.std().item(),
            }
            results['audio_analysis'] = audio_analysis
        
        # Cross-modality analysis
        if text_features is not None and audio_features is not None:
            # Calculate correlation between modalities
            text_flat = text_features.flatten()
            audio_flat = audio_features.flatten()
            
            if len(text_flat) == len(audio_flat):
                correlation = torch.corrcoef(torch.stack([text_flat, audio_flat]))[0, 1].item()
            else:
                correlation = 0.0
            
            results['cross_modality_analysis'] = {
                'correlation': correlation,
                'feature_ratio': text_features.shape[-1] / audio_features.shape[-1],
            }
        
        return results
    
    def save_pretrained(self, save_directory: str):
        """Save the fusion engine to the specified directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        torch.save(self.state_dict(), os.path.join(save_directory, 'fusion_engine.pth'))
        
        # Save configuration
        config = {
            'fusion_method': self.fusion_method,
            'text_dim': self.text_dim,
            'audio_dim': self.audio_dim,
            'device': str(self.device),
        }
        
        import json
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load a pretrained fusion engine from the specified directory."""
        import json
        import torch
        
        # Load configuration
        config_path = os.path.join(load_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model with loaded configuration
        model = cls(
            fusion_method=config.get('fusion_method', 'cross_modal_attention'),
            text_dim=config.get('text_dim', 768),
            audio_dim=config.get('audio_dim', 256),
            device=config.get('device', 'auto'),
            **kwargs
        )
        
        # Load pretrained weights
        model.load_state_dict(torch.load(os.path.join(load_directory, 'fusion_engine.pth')))
        
        return model