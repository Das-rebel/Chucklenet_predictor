"""
Audio Laughter Detector

This module implements audio-based laughter detection and reverse modeling
using Whisper and AudioCraft from the AI-research-SKILLS repository.

Author: Subho Das
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union
import os

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not available. Install with: pip install openai-whisper")

try:
    import audiocraft
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_read
    AUDIOCRAFT_AVAILABLE = True
except ImportError:
    AUDIOCRAFT_AVAILABLE = False
    print("Warning: AudioCraft not available. Install with: pip install audiocraft")


class AudioFeaturesExtractor(nn.Module):
    """
    Extract audio features from raw audio data for laughter detection.
    
    Features include:
    - Spectral features (MFCC, spectral centroid, bandwidth, rolloff)
    - Temporal features (zero crossing rate, energy)
    - Pitch and prosodic features
    - Laughter-specific acoustic features
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Spectral feature extractors
        self.mfcc_layer = nn.Sequential(
            nn.Linear(n_mfcc, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.spectral_layer = nn.Sequential(
            nn.Linear(4, 64),  # spectral_centroid, bandwidth, rolloff, flux
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Temporal feature extractor
        self.temporal_layer = nn.Sequential(
            nn.Linear(2, 32),  # zero_crossing_rate, energy
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Pitch and prosodic features
        self.pitch_layer = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Laughter-specific features
        self.laughter_layer = nn.Sequential(
            nn.Linear(10, 64),  # laughter-specific features
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def extract_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features."""
        if not librosa:
            return torch.zeros(1, self.n_mfcc)
        
        mfccs = librosa.feature.mfcc(
            y=audio.numpy(),
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Normalize MFCCs
        mfccs = (mfccs - mfccs.mean()) / (mfccs.std() + 1e-8)
        mfccs = torch.from_numpy(mfccs).float()
        
        # Time-domain average
        mfccs_mean = mfccs.mean(dim=1, keepdim=True)
        return self.mfcc_layer(mfccs_mean)
    
    def extract_spectral_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract spectral features."""
        if not librosa:
            return torch.zeros(1, 4)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio.numpy(), sr=self.sample_rate, hop_length=self.hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio.numpy(), sr=self.sample_rate, hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio.numpy(), sr=self.sample_rate, hop_length=self.hop_length
        )
        spectral_flux = librosa.feature.spectral_flux(
            y=audio.numpy(), sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Normalize and concatenate
        spectral_features = np.stack([
            (spectral_centroid - spectral_centroid.mean()) / (spectral_centroid.std() + 1e-8),
            (spectral_bandwidth - spectral_bandwidth.mean()) / (spectral_bandwidth.std() + 1e-8),
            (spectral_rolloff - spectral_rolloff.mean()) / (spectral_rolloff.std() + 1e-8),
            (spectral_flux - spectral_flux.mean()) / (spectral_flux.std() + 1e-8),
        ], axis=0)
        
        spectral_features = torch.from_numpy(spectral_features).float()
        spectral_features_mean = spectral_features.mean(dim=1, keepdim=True)
        
        return self.spectral_layer(spectral_features_mean)
    
    def extract_temporal_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract temporal features."""
        if not librosa:
            return torch.zeros(1, 2)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio.numpy(), frame_length=self.n_fft, hop_length=self.hop_length
        )
        
        # Energy
        energy = librosa.feature.rms(
            y=audio.numpy(), frame_length=self.n_fft, hop_length=self.hop_length
        )
        
        # Normalize
        zcr_norm = (zcr - zcr.mean()) / (zcr.std() + 1e-8)
        energy_norm = (energy - energy.mean()) / (energy.std() + 1e-8)
        
        temporal_features = torch.from_numpy(np.stack([zcr_norm, energy_norm], axis=0)).float()
        temporal_features_mean = temporal_features.mean(dim=1, keepdim=True)
        
        return self.temporal_layer(temporal_features_mean)
    
    def extract_pitch_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pitch features."""
        if not librosa:
            return torch.zeros(1, 1)
        
        # Fundamental frequency (pitch)
        pitches, magnitudes = librosa.piptrack(
            y=audio.numpy(), 
            sr=self.sample_rate, 
            hop_length=self.hop_length
        )
        
        # Get the most prominent pitch at each frame
        pitch_frames = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch_frames.append(pitches[index, i])
        
        pitch_frames = np.array(pitch_frames)
        pitch_frames = np.where(pitch_frames > 0, pitch_frames, 0)  # Replace zeros with silence
        
        # Normalize
        if len(pitch_frames) > 0 and pitch_frames.std() > 0:
            pitch_norm = (pitch_frames - pitch_frames.mean()) / (pitch_frames.std() + 1e-8)
        else:
            pitch_norm = np.zeros_like(pitch_frames)
        
        pitch_tensor = torch.from_numpy(pitch_norm).float().mean(dim=0, keepdim=True)
        
        return self.pitch_layer(pitch_tensor)
    
    def extract_laughter_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract laughter-specific acoustic features."""
        # High-frequency energy (laughter often has high freq content)
        high_freq = torch.fft.fft(audio)
        high_energy = torch.abs(high_freq[len(high_freq)//2:]).mean()
        
        # Low-frequency energy ratio
        low_energy = torch.abs(high_freq[:len(high_freq)//2]).mean()
        
        # Voice activity detection
        vad = torch.abs(audio) > torch.abs(audio).mean() * 1.5
        
        # Rapid amplitude changes (characteristic of laughter)
        amplitude_diff = torch.abs(torch.diff(audio))
        rapid_changes = (amplitude_diff > amplitude_diff.mean() * 2).float()
        
        # Formant analysis (simplified)
        formants = self._estimate_formants(audio)
        
        # Laughter feature vector
        laughter_features = torch.tensor([
            high_energy.item(),
            low_energy.item(),
            vad.float().mean().item(),
            rapid_changes.float().mean().item(),
            formants['f1'].item() if formants['f1'] is not None else 0,
            formants['f2'].item() if formants['f2'] is not None else 0,
            formants['f3'].item() if formants['f3'] is not None else 0,
            1.0,  # Placeholder for laughter detection score
            1.0,  # Placeholder for laugh intensity
            1.0,  # Placeholder for laugh duration
        ])
        
        return self.laughter_layer(laughter_features.float())
    
    def _estimate_formants(self, audio: torch.Tensor) -> Dict[str, Optional[float]]:
        """Estimate formant frequencies (simplified)."""
        try:
            if not librosa:
                return {'f1': None, 'f2': None, 'f3': None}
            
            # Simple formant estimation using LPC
            lpc_order = 8
            a = librosa.lpc(audio.numpy(), order=lpc_order)
            
            # Find roots of LPC polynomial
            roots = np.roots(a)
            roots = [r for r in roots if np.iscomplex(r)]
            
            if len(roots) >= 3:
                # Convert to angles and get formants
                angles = np.angle(roots)
                fs = angles * self.sample_rate / (2 * np.pi)
                
                # Sort by magnitude and take first 3 formants
                sorted_fs = sorted([f.real for f in fs if f > 0 and f < 5000])
                
                return {
                    'f1': sorted_fs[0] if len(sorted_fs) > 0 else None,
                    'f2': sorted_fs[1] if len(sorted_fs) > 1 else None,
                    'f3': sorted_fs[2] if len(sorted_fs) > 2 else None,
                }
        except:
            pass
        
        return {'f1': None, 'f2': None, 'f3': None}
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract comprehensive audio features."""
        features = []
        
        # Extract different types of features
        features.append(self.extract_mfcc(audio))
        features.append(self.extract_spectral_features(audio))
        features.append(self.extract_temporal_features(audio))
        features.append(self.extract_pitch_features(audio))
        features.append(self.extract_laughter_features(audio))
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=1)
        
        return combined_features


class AudioLaughDetector(nn.Module):
    """
    Complete audio-based laughter detection system with reverse modeling capabilities.
    
    Combines:
    - Whisper for speech recognition and context
    - Custom audio features extraction
    - Deep learning models for laughter detection
    - Reverse modeling for humor generation
    """
    
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-small",
        device: str = "auto"
    ):
        super().__init__()
        
        self.device = self._get_device(device)
        
        # Initialize Whisper if available
        if WHISPER_AVAILABLE:
            self.whisper_model = whisper.load_model(whisper_model_name)
            self.whisper_model.to(self.device)
        else:
            self.whisper_model = None
        
        # Audio feature extractor
        self.feature_extractor = AudioFeaturesExtractor()
        self.feature_extractor.to(self.device)
        
        # Laughter detection layers
        self.laughter_classifier = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 32 + 64, 256),  # All feature types
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Reverse modeling layers (for humor generation potential)
        self.reverse_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Time analysis layers
        self.temporal_classifier = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.final_fusion = nn.Sequential(
            nn.Linear(128 + 1 + 1, 64),  # LSTM output + laugh prob + reverse score
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and preprocess."""
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio = torch.from_numpy(audio).float()
            
            # Normalize audio
            if torch.abs(audio).max() > 0:
                audio = audio / torch.abs(audio).max()
            
            # Ensure correct length (adjust padding if needed)
            target_length = 16000 * 10  # 10 seconds default
            if len(audio) < target_length:
                padding = target_length - len(audio)
                audio = F.pad(audio, (0, padding))
            elif len(audio) > target_length:
                audio = audio[:target_length]
            
            return audio.to(self.device)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return torch.zeros(160000, device=self.device)  # 10 seconds of silence
    
    def extract_whisper_features(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract features using Whisper for speech context."""
        if self.whisper_model is None:
            return None
        
        try:
            # Whisper inference
            result = self.whisper_model.transcribe(audio.cpu().numpy())
            text = result['text']
            
            # Simple text-based humor features from Whisper transcription
            # This could be enhanced with more sophisticated text analysis
            if text:
                # Basic text features that might indicate humor
                text_features = torch.tensor([
                    len(text) / 100.0,  # Length normalized
                    text.count('?') / len(text) if len(text) > 0 else 0,  # Question marks
                    text.count('!') / len(text) if len(text) > 0 else 0,  # Exclamation marks
                    text.lower().count('lol') + text.lower().count('haha'),  # Laugh indicators
                    1.0 if any(word in text.lower() for word in ['funny', 'joke', 'humor', 'laugh']) else 0.0,
                ])
                return text_features.to(self.device)
        
        except Exception as e:
            print(f"Whisper inference error: {e}")
        
        return None
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for audio laughter detection and reverse modeling.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Dictionary containing:
            - laugh_prob: Probability of laughter detection
            - reverse_score: Reverse humor potential (0-100%)
            - features: Extracted features
            - temporal_patterns: Temporal analysis results
        """
        # Extract audio features
        audio_features = self.feature_extractor(audio)
        
        # Initial laughter detection
        laugh_prob = self.laughter_classifier(audio_features)
        
        # Reverse modeling for humor potential
        reverse_features = self.reverse_model(audio_features)
        reverse_score = reverse_features * 100  # Convert to percentage
        
        # Temporal analysis
        temporal_input = audio_features.unsqueeze(0)  # Add batch dimension
        temporal_output, _ = self.temporal_classifier(temporal_input)
        temporal_features = temporal_output.mean(dim=1)  # Mean pooling
        
        # Final fusion
        final_features = torch.cat([temporal_features, laugh_prob, reverse_features], dim=1)
        final_score = self.final_fusion(final_features)
        
        return {
            'laugh_prob': laugh_prob,
            'reverse_score': reverse_score,
            'features': audio_features,
            'temporal_patterns': temporal_features,
            'final_score': final_score,
        }
    
    def detect_laughter(self, audio_path: str) -> Tuple[float, float]:
        """
        Detect laughter and predict reverse humor potential from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (laughter_probability, reverse_humor_score)
        """
        self.eval()
        with torch.no_grad():
            # Load audio
            audio = self.load_audio(audio_path)
            
            # Forward pass
            results = self(audio)
            
            laugh_prob = results['laugh_prob'].cpu().numpy()[0][0]
            reverse_score = results['reverse_score'].cpu().numpy()[0][0]
            
            return float(laugh_prob), float(reverse_score)
    
    def analyze_audio_features(self, audio_path: str) -> Dict[str, float]:
        """
        Analyze detailed audio features for comprehensive analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detailed audio analysis
        """
        self.eval()
        with torch.no_grad():
            audio = self.load_audio(audio_path)
            results = self(audio)
            
            # Extract detailed feature information
            features = results['features'].cpu().numpy()[0]
            
            # Time-domain analysis
            temporal = results['temporal_patterns'].cpu().numpy()[0]
            
            analysis = {
                'laughter_probability': float(results['laugh_prob'].cpu().numpy()[0][0]),
                'reverse_humor_score': float(results['reverse_score'].cpu().numpy()[0][0]),
                'final_confidence': float(results['final_score'].cpu().numpy()[0][0]),
                'feature_complexity': float(np.std(features)),
                'temporal_variance': float(np.std(temporal)),
                'audio_energy': float(torch.mean(audio**2).cpu().numpy()),
                'audio_duration': float(len(audio) / 16000),  # seconds
            }
            
            # Add Whisper-based features if available
            whisper_features = self.extract_whisper_features(audio)
            if whisper_features is not None:
                analysis['whisper_text_features'] = whisper_features.cpu().numpy().tolist()
            
            return analysis
    
    def save_pretrained(self, save_directory: str):
        """Save the model to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, 'audio_laugh_detector.pth'))
        
        # Save configuration
        config = {
            'whisper_model_name': 'openai/whisper-small' if self.whisper_model else None,
            'device': str(self.device),
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
            whisper_model_name=config.get('whisper_model_name', 'openai/whisper-small'),
            device=config.get('device', 'auto'),
            **kwargs
        )
        
        # Load pretrained weights
        model.load_state_dict(torch.load(os.path.join(load_directory, 'audio_laugh_detector.pth')))
        
        return model