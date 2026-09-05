"""
Audio Processor for Chucklenet Predictor

This module provides comprehensive audio processing capabilities including:
- Audio feature extraction
- Laughter detection
- Acoustic analysis
- Prosodic features
- Emotional analysis from audio

Author: Subho Das
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings
import logging
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Comprehensive audio processing for humor analysis.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 duration: float = 10.0,
                 use_gpu: bool = True):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            duration: Default duration for audio segments
            use_gpu: Whether to use GPU for processing
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize audio models if available
        self.whisper_model = None
        self.laughter_detector = None
        self.emotion_analyzer = None
        
        try:
            # Initialize Whisper for speech recognition
            import whisper
            self.whisper_model = whisper.load_model("tiny", device="cuda" if self.use_gpu else "cpu")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Whisper model: {e}")
        
        # Acoustic parameters
        self.frame_length = 2048
        self.hop_length = 512
        
        # Feature extraction parameters
        self.n_mfcc = 13
        self.n_mels = 128
        self.n_fft = 2048
        
    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Process audio file and extract features.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Validate audio
            if len(audio) == 0:
                return {'error': 'Empty audio file'}
            
            # Extract features
            features = self.extract_all_features(audio, sr)
            
            # Add metadata
            features['metadata'] = {
                'file_path': audio_path,
                'sample_rate': sr,
                'duration': len(audio) / sr,
                'channels': 1 if len(audio.shape) == 1 else audio.shape[0],
                'file_size': Path(audio_path).stat().st_size
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return {'error': str(e)}
    
    def extract_all_features(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, Dict]:
        """
        Extract all audio features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'spectral': self._extract_spectral_features(audio, sr),
            'temporal': self._extract_temporal_features(audio, sr),
            'mfcc': self._extract_mfcc_features(audio, sr),
            'pitch': self._extract_pitch_features(audio, sr),
            'prosodic': self._extract_prosodic_features(audio, sr),
            'laughter': self._detect_laughter(audio, sr),
            'emotional': self._analyze_emotional_content(audio, sr),
            'rhythmic': self._extract_rhythmic_features(audio, sr),
            'quality': self._analyze_audio_quality(audio, sr)
        }
        
        return features
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract spectral features."""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                S=magnitude, sr=sr, hop_length=self.hop_length
            )[0]
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                S=magnitude, sr=sr, hop_length=self.hop_length
            )[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                S=magnitude, sr=sr, hop_length=self.hop_length
            )[0]
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                S=magnitude, sr=sr, hop_length=self.hop_length
            )
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                S=magnitude, sr=sr, hop_length=self.hop_length
            )
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(
                chroma=chroma
            )
            
            return {
                'centroid': spectral_centroid,
                'bandwidth': spectral_bandwidth,
                'rolloff': spectral_rolloff,
                'contrast': spectral_contrast,
                'chroma': chroma,
                'tonnetz': tonnetz,
                'magnitude': magnitude,
                'phase': phase
            }
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            return {}
    
    def _extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract temporal features."""
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
            
            # Root mean square energy
            rms = librosa.feature.rms(
                y=audio, frame_length=self.frame_length, hop_length=self.hop_length
            )[0]
            
            # Amplitude envelope
            amplitude_envelope = np.abs(librosa.effects.preemphasis(audio))
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sr, hop_length=self.hop_length
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(
                y=audio, sr=sr, hop_length=self.hop_length
            )
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            return {
                'zero_crossing_rate': zcr,
                'rms_energy': rms,
                'amplitude_envelope': amplitude_envelope,
                'onset_times': onset_times,
                'tempo': tempo,
                'beat_times': beat_times,
                'beat_times': beat_times
            }
        except Exception as e:
            logger.warning(f"Error extracting temporal features: {e}")
            return {}
    
    def _extract_mfcc_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract MFCC features."""
        try:
            # Standard MFCC
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
            )
            
            # Delta and delta-delta MFCC
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
            
            # MFCC Cepstral Coefficients
            mfcc_coefficients = mfcc.T
            mfcc_stats = {
                'mean': np.mean(mfcc_coefficients, axis=0),
                'std': np.std(mfcc_coefficients, axis=0),
                'min': np.min(mfcc_coefficients, axis=0),
                'max': np.max(mfcc_coefficients, axis=0)
            }
            
            return {
                'mfcc': mfcc,
                'delta': mfcc_delta,
                'delta_delta': mfcc_delta_delta,
                'coefficients': mfcc_coefficients,
                'statistics': mfcc_stats
            }
        except Exception as e:
            logger.warning(f"Error extracting MFCC features: {e}")
            return {}
    
    def _extract_pitch_features(self, audio: np.ndarray, sr: int) -> Dict[str, Union[np.ndarray, float]]:
        """Extract pitch features."""
        try:
            # Fundamental frequency (f0) estimation
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'), sr=sr
            )
            
            # Pitch statistics
            voiced_f0 = f0[voiced_flag]
            pitch_stats = {
                'mean': np.mean(voiced_f0) if len(voiced_f0) > 0 else 0,
                'std': np.std(voiced_f0) if len(voiced_f0) > 0 else 0,
                'min': np.min(voiced_f0) if len(voiced_f0) > 0 else 0,
                'max': np.max(voiced_f0) if len(voiced_f0) > 0 else 0,
                'range': np.max(voiced_f0) - np.min(voiced_f0) if len(voiced_f0) > 0 else 0,
                'median': np.median(voiced_f0) if len(voiced_f0) > 0 else 0,
                'skewness': skew(voiced_f0) if len(voiced_f0) > 1 else 0,
                'kurtosis': kurtosis(voiced_f0) if len(voiced_f0) > 1 else 0
            }
            
            # Pitch contour analysis
            pitch_contour = f0[voiced_flag]
            pitch_velocity = np.diff(pitch_contour) if len(pitch_contour) > 1 else np.array([])
            
            # Formant frequencies (simplified)
            formant_f1 = np.mean(librosa.yin(audio, fmin=85, fmax=255, sr=sr)[voiced_flag]) if len(voiced_f0) > 0 else 0
            formant_f2 = np.mean(librosa.yin(audio, fmin=400, fmax=2000, sr=sr)[voiced_flag]) if len(voiced_f0) > 0 else 0
            
            return {
                'f0': f0,
                'voiced_flag': voiced_flag,
                'voiced_probs': voiced_probs,
                'voiced_f0': voiced_f0,
                'pitch_stats': pitch_stats,
                'pitch_contour': pitch_contour,
                'pitch_velocity': pitch_velocity,
                'formant_f1': formant_f1,
                'formant_f2': formant_f2,
                'voiced_ratio': np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0
            }
        except Exception as e:
            logger.warning(f"Error extracting pitch features: {e}")
            return {}
    
    def _extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Union[float, np.ndarray]]:
        """Extract prosodic features."""
        try:
            # Fundamental frequency for prosodic analysis
            f0, voiced_flag, _ = librosa.pyin(
                audio, fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'), sr=sr
            )
            
            # Prosodic features
            prosodic_features = {}
            
            # Fundamental frequency statistics
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0:
                prosodic_features['f0_mean'] = np.mean(voiced_f0)
                prosodic_features['f0_std'] = np.std(voiced_f0)
                prosodic_features['f0_range'] = np.max(voiced_f0) - np.min(voiced_f0)
                prosodic_features['f0_median'] = np.median(voiced_f0)
            else:
                prosodic_features['f0_mean'] = 0
                prosodic_features['f0_std'] = 0
                prosodic_features['f0_range'] = 0
                prosodic_features['f0_median'] = 0
            
            # Speaking rate
            duration = len(audio) / sr
            speaking_rate = len(np.where(voiced_flag)[0]) / duration if duration > 0 else 0
            prosodic_features['speaking_rate'] = speaking_rate
            
            # Pause ratio
            pause_ratio = 1 - (np.sum(voiced_flag) / len(voiced_flag)) if len(voiced_flag) > 0 else 0
            prosodic_features['pause_ratio'] = pause_ratio
            
            # Pause count
            pause_segments = np.diff(np.where(np.diff(np.concatenate(([True], ~voiced_flag, [True]))))[0])
            pause_count = np.sum(pause_segments > 1)  # Pauses longer than 1 frame
            prosodic_features['pause_count'] = pause_count
            
            # Rhythm regularity
            if len(voiced_f0) > 10:
                # Calculate inter-pitch intervals
                pitch_intervals = np.diff(voiced_f0)
                rhythm_regularity = 1 / (1 + np.std(pitch_intervals))
            else:
                rhythm_regularity = 0
            prosodic_features['rhythm_regularity'] = rhythm_regularity
            
            # Energy variation
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            energy_variation = np.std(rms) / (np.mean(rms) + 1e-8)
            prosodic_features['energy_variation'] = energy_variation
            
            # Amplitude modulation
            am_modulation = np.abs(np.fft.rfft(amplitude_envelope))
            am_features = {
                'dominant_freq': np.argmax(am_modulation),
                'amplitude': np.max(am_modulation),
                'bandwidth': np.std(am_modulation)
            }
            prosodic_features['am_modulation'] = am_features
            
            return prosodic_features
        except Exception as e:
            logger.warning(f"Error extracting prosodic features: {e}")
            return {}
    
    def _detect_laughter(self, audio: np.ndarray, sr: int) -> Dict[str, Union[bool, float, List]]:
        """Detect laughter in audio."""
        try:
            # Simple laughter detection based on energy and frequency
            # In a real implementation, you would use a proper laughter detector model
            
            # Energy threshold for laughter
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            energy_threshold = np.mean(rms) + 2 * np.std(rms)
            
            # Frequency characteristics of laughter
            f0, voiced_flag, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            
            # Laughter detection heuristics
            laughter_frames = []
            laughter_segments = []
            
            for i, energy in enumerate(rms):
                if energy > energy_threshold:
                    start_frame = i * self.hop_length
                    end_frame = min((i + 1) * self.hop_length, len(audio))
                    
                    # Check if high energy coincides with voiced speech
                    if i < len(voiced_flag) and voiced_flag[i]:
                        segment = audio[start_frame:end_frame]
                        # Laughter typically has higher frequency content
                        if len(segment) > 0:
                            segment_freq = np.abs(np.fft.rfft(segment))
                            high_freq_energy = np.sum(segment_freq[len(segment_freq)//2:]) / np.sum(segment_freq)
                            
                            if high_freq_energy > 0.3:  # High frequency energy ratio
                                laughter_frames.append(i)
                                laughter_segments.append((start_frame, end_frame))
            
            # Calculate laughter statistics
            laughter_ratio = len(laughter_frames) / len(rms) if len(rms) > 0 else 0
            laughter_duration = len(laughter_frames) * self.hop_length / sr
            laughter_count = len(laughter_segments)
            
            return {
                'detected': laughter_count > 0,
                'laughter_frames': laughter_frames,
                'laughter_segments': laughter_segments,
                'laughter_ratio': laughter_ratio,
                'laughter_duration': laughter_duration,
                'laughter_count': laughter_count,
                'confidence': min(1.0, laughter_ratio * 10)  # Simple confidence metric
            }
        except Exception as e:
            logger.warning(f"Error detecting laughter: {e}")
            return {'detected': False, 'laughter_frames': [], 'laughter_segments': [], 'laughter_ratio': 0, 'laughter_duration': 0, 'laughter_count': 0, 'confidence': 0}
    
    def _analyze_emotional_content(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze emotional content from audio."""
        try:
            # In a real implementation, you would use an emotion recognition model
            # Here we use simple heuristic-based emotion detection
            
            # Basic emotion indicators
            features = {}
            
            # Energy (loudness)
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            avg_energy = np.mean(rms)
            features['energy'] = avg_energy
            
            # Pitch variability
            f0, voiced_flag, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0:
                pitch_variability = np.std(voiced_f0)
            else:
                pitch_variability = 0
            features['pitch_variability'] = pitch_variability
            
            # Speech rate
            duration = len(audio) / sr
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) > 0 else 0
            speech_rate = len(np.where(voiced_flag)[0]) / duration if duration > 0 else 0
            features['speech_rate'] = speech_rate
            
            # Pause patterns
            pause_ratio = 1 - voiced_ratio
            features['pause_ratio'] = pause_ratio
            
            # Simple emotion classification based on features
            # These are very simplified rules for demonstration
            
            # Excitement (high energy, high pitch variability)
            excitement = min(1.0, (avg_energy / 0.1) + (pitch_variability / 100))
            features['excitement'] = excitement
            
            # Calmness (low energy, low pitch variability)
            calmness = min(1.0, (1 / (avg_energy + 0.1)) * (1 / (pitch_variability + 1)))
            features['calmness'] = calmness
            
            # Happiness (laughter detection)
            laughter = self._detect_laughter(audio, sr)
            features['happiness'] = laughter['laughter_ratio']
            
            # Anger (high energy, low pitch variability, fast speech)
            anger = min(1.0, (avg_energy / 0.1) * (speech_rate / 10) * (1 / (pitch_variability + 1)))
            features['anger'] = anger
            
            # Sadness (low energy, slow speech, low pitch)
            if len(voiced_f0) > 0:
                avg_pitch = np.mean(voiced_f0)
                sadness = min(1.0, (1 / (avg_energy + 0.1)) * (1 / speech_rate) * (1 / (avg_pitch + 1)))
            else:
                sadness = 0
            features['sadness'] = sadness
            
            # Fear (high pitch variability, moderate energy)
            fear = min(1.0, (pitch_variability / 100) * (avg_energy / 0.1))
            features['fear'] = fear
            
            # Surprise (sudden changes in energy or pitch)
            energy_changes = np.abs(np.diff(rms))
            sudden_changes = np.sum(energy_changes > np.mean(energy_changes) + 2 * np.std(energy_changes))
            surprise = min(1.0, sudden_changes / len(energy_changes))
            features['surprise'] = surprise
            
            # Normalize emotion scores
            total_emotion = sum(features.values())
            if total_emotion > 0:
                for emotion in features:
                    features[emotion] /= total_emotion
            
            # Add dominant emotion
            if features:
                dominant_emotion = max(features, key=features.get)
                features['dominant_emotion'] = dominant_emotion
                features['dominant_score'] = features[dominant_emotion]
            
            return features
            
        except Exception as e:
            logger.warning(f"Error analyzing emotional content: {e}")
            return {}
    
    def _extract_rhythmic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Union[float, np.ndarray]]:
        """Extract rhythmic features."""
        try:
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Beat histogram
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Beat strength
            beat_strength = np.abs(librosa.onset.onset_strength(y=audio, sr=sr, hop_length=self.hop_length))
            
            # Rhythmic patterns
            if len(beats) > 1:
                beat_intervals = np.diff(beat_times)
                rhythmic_regularity = 1 / (1 + np.std(beat_intervals))
            else:
                rhythmic_regularity = 0
            
            # Beat classification
            if tempo > 160:
                beat_type = 'fast'
            elif tempo > 100:
                beat_type = 'medium'
            else:
                beat_type = 'slow'
            
            # Syncopation (off-beat emphasis)
            syncopation = self._analyze_syncopation(audio, sr)
            
            return {
                'tempo': tempo,
                'beats': beats,
                'beat_times': beat_times,
                'beat_strength': beat_strength,
                'rhythmic_regularity': rhythmic_regularity,
                'beat_type': beat_type,
                'syncopation': syncopation
            }
        except Exception as e:
            logger.warning(f"Error extracting rhythmic features: {e}")
            return {}
    
    def _analyze_syncopation(self, audio: np.ndarray, sr: int) -> float:
        """Analyze syncopation in audio."""
        try:
            # Simple syncopation detection
            # This is a very basic heuristic
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=self.hop_length)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            
            # Beat grid
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
            
            # Calculate syncopation as onset-beat offset
            syncopation_score = 0
            total_onsets = 0
            
            for onset in onset_times:
                # Find nearest beat
                if len(beat_times) > 0:
                    nearest_beat_idx = np.argmin(np.abs(beat_times - onset))
                    beat_time = beat_times[nearest_beat_idx]
                    offset = abs(onset - beat_time)
                    
                    # Syncopation if offset is significant
                    if offset > 0.1:  # More than 0.1s off beat
                        syncopation_score += offset
                        total_onsets += 1
            
            syncopation = syncopation_score / max(total_onsets, 1)
            
            return min(1.0, syncopation / 0.5)  # Normalize to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error analyzing syncopation: {e}")
            return 0.0
    
    def _analyze_audio_quality(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze audio quality."""
        try:
            features = {}
            
            # Signal-to-noise ratio (SNR)
            signal_energy = np.sum(audio ** 2)
            noise_energy = np.sum((audio - np.mean(audio)) ** 2)
            snr = 10 * np.log10(signal_energy / (noise_energy + 1e-8))
            features['snr'] = snr
            
            # Clipping detection
            max_amplitude = np.max(np.abs(audio))
            features['clipping'] = 1.0 if max_amplitude >= 1.0 else 0.0
            
            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            features['dynamic_range'] = dynamic_range
            
            # Frequency content analysis
            stft = librosa.stft(audio, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            freq_content = np.sum(magnitude, axis=1)
            features['frequency_content_balance'] = np.std(freq_content) / (np.mean(freq_content) + 1e-8)
            
            # Background noise estimation
            # Use silent portions to estimate noise
            silent_threshold = 0.01
            silent_frames = np.abs(audio) < silent_threshold
            if np.any(silent_frames):
                noise_level = np.mean(np.abs(audio[silent_frames]))
            else:
                noise_level = np.mean(np.abs(audio))
            features['noise_level'] = noise_level
            
            # Quality assessment
            quality_score = 0
            # Good SNR (>20 dB)
            if snr > 20:
                quality_score += 0.3
            elif snr > 10:
                quality_score += 0.1
            
            # No clipping
            if features['clipping'] == 0:
                quality_score += 0.2
            
            # Good dynamic range (>0.8)
            if dynamic_range > 0.8:
                quality_score += 0.2
            
            # Low noise
            if noise_level < 0.01:
                quality_score += 0.2
            
            # Balanced frequency content
            if 0.1 < features['frequency_content_balance'] < 2.0:
                quality_score += 0.1
            
            features['overall_quality'] = min(1.0, quality_score)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error analyzing audio quality: {e}")
            return {'overall_quality': 0.0}
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        Extract audio features from file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Extracted audio features
        """
        return self.process_audio_file(audio_path)
    
    def analyze_laughter_patterns(self, audio: np.ndarray, sr: int) -> Dict[str, Union[float, List]]:
        """Analyze laughter patterns in detail."""
        try:
            laughter_features = self._detect_laughter(audio, sr)
            
            if not laughter_features['detected']:
                return {'detected': False, 'message': 'No laughter detected'}
            
            # Analyze laughter segments
            laughter_segments = laughter_features['laughter_segments']
            laughter_analysis = []
            
            for start, end in laughter_segments:
                segment = audio[start:end]
                
                # Segment analysis
                segment_features = {
                    'start_time': start / sr,
                    'end_time': end / sr,
                    'duration': (end - start) / sr,
                    'energy': np.mean(segment ** 2),
                    'pitch': np.mean(librosa.yin(segment, fmin=85, fmax=255, sr=sr)[np.isfinite(librosa.yin(segment, fmin=85, fmax=255, sr=sr))]) if len(segment) > 0 else 0,
                    'frequency_content': np.sum(np.abs(np.fft.rfft(segment))[len(np.abs(np.fft.rfft(segment)))//2:]) / np.sum(np.abs(np.fft.rfft(segment))),
                    'laughter_type': self._classify_laughter_type(segment, sr)
                }
                
                laughter_analysis.append(segment_features)
            
            # Overall laughter analysis
            total_laughter_duration = sum(seg['duration'] for seg in laughter_analysis)
            avg_laughter_duration = total_laughter_duration / len(laughter_analysis) if laughter_analysis else 0
            
            # Laughter intensity
            laughter_intensities = [seg['energy'] for seg in laughter_analysis]
            avg_intensity = np.mean(laughter_intensities) if laughter_intensities else 0
            
            # Laughter type distribution
            laughter_types = [seg['laughter_type'] for seg in laughter_analysis]
            type_distribution = {}
            for laughter_type in laughter_types:
                type_distribution[laughter_type] = type_distribution.get(laughter_type, 0) + 1
            
            # Most common laughter type
            most_common_type = max(type_distribution, key=type_distribution.get) if type_distribution else None
            
            return {
                'detected': True,
                'laughter_segments': laughter_analysis,
                'total_laughter_duration': total_laughter_duration,
                'average_laughter_duration': avg_laughter_duration,
                'average_intensity': avg_intensity,
                'laughter_type_distribution': type_distribution,
                'most_common_laughter_type': most_common_type,
                'laughter_frequency': len(laughter_segments),
                'laughter_ratio': laughter_features['laughter_ratio']
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing laughter patterns: {e}")
            return {'detected': False, 'message': f'Error: {str(e)}'}
    
    def _classify_laughter_type(self, audio_segment: np.ndarray, sr: int) -> str:
        """Classify laughter type."""
        try:
            # Simple heuristic-based laughter classification
            # This is a very basic implementation
            
            if len(audio_segment) < 100:
                return 'very_short'
            
            # Energy analysis
            energy = np.mean(audio_segment ** 2)
            
            # Frequency analysis
            freq_spectrum = np.abs(np.fft.rfft(audio_segment))
            high_freq_content = np.sum(freq_spectrum[len(freq_spectrum)//2:]) / np.sum(freq_spectrum)
            
            # Duration analysis
            duration = len(audio_segment) / sr
            
            # Classify based on characteristics
            if duration < 0.2:
                return 'short_burst'
            elif duration > 2.0:
                return 'long_laugh'
            elif energy > 0.1:
                return 'intense_laugh'
            elif high_freq_content > 0.6:
                return 'high_pitch_laugh'
            else:
                return 'normal_laugh'
                
        except Exception as e:
            logger.warning(f"Error classifying laughter type: {e}")
            return 'unknown'
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict[str, Union[float, np.ndarray]]:
        """Extract prosodic features for speech analysis."""
        return self._extract_prosodic_features(audio, sr)
    
    def batch_analyze(self, audio_files: List[str], batch_size: int = 4) -> List[Dict]:
        """
        Batch analyze multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            batch_size: Batch size for processing
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            batch_results = []
            
            for audio_file in batch_files:
                try:
                    result = self.process_audio_file(audio_file)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"Error processing {audio_file}: {e}")
                    batch_results.append({'error': str(e)})
            
            results.extend(batch_results)
        
        return results
    
    def preprocess_audio(self, audio: np.ndarray, 
                        normalize: bool = True,
                        trim_silence: bool = True,
                        resample: bool = True) -> np.ndarray:
        """
        Preprocess audio data.
        
        Args:
            audio: Input audio signal
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence
            resample: Whether to resample to target rate
            
        Returns:
            Preprocessed audio signal
        """
        processed_audio = audio.copy()
        
        # Resample if needed
        if resample and len(audio) > 0:
            processed_audio = librosa.resample(processed_audio, orig_sr=len(audio)/10, target_sr=self.sample_rate)
        
        # Normalize
        if normalize:
            max_val = np.max(np.abs(processed_audio))
            if max_val > 0:
                processed_audio = processed_audio / max_val
        
        # Trim silence
        if trim_silence and len(processed_audio) > 0:
            processed_audio, _ = librosa.effects.trim(processed_audio, top_db=20)
        
        return processed_audio
    
    def get_audio_statistics(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Get basic audio statistics."""
        try:
            return {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'channels': 1 if len(audio.shape) == 1 else audio.shape[0],
                'max_amplitude': np.max(np.abs(audio)),
                'mean_amplitude': np.mean(np.abs(audio)),
                'rms_energy': np.sqrt(np.mean(audio ** 2)),
                'zero_crossings': np.sum(np.diff(np.sign(audio)) != 0),
                'zero_crossing_rate': np.mean(np.abs(np.diff(np.sign(audio))) / 2)
            }
        except Exception as e:
            logger.warning(f"Error getting audio statistics: {e}")
            return {}
    
    def visualize_audio_features(self, audio: np.ndarray, sr: int, features: Dict = None):
        """
        Visualize audio features (placeholder for actual visualization).
        
        Args:
            audio: Audio signal
            sr: Sample rate
            features: Extracted features
        """
        # This would be implemented with matplotlib or other visualization libraries
        logger.info("Audio visualization not implemented in this version")
    
    def save_audio_features(self, audio_path: str, output_path: str):
        """
        Save extracted audio features to file.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save features
        """
        try:
            features = self.process_audio_file(audio_path)
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            serializable_features = convert_numpy(features)
            
            # Save as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(serializable_features, f, indent=2)
            
            logger.info(f"Audio features saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving audio features: {e}")
    
    def create_feature_vector(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Create a flattened feature vector from audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Flattened feature vector
        """
        try:
            # Extract all features
            features = self.extract_all_features(audio, sr)
            
            # Flatten features into a single vector
            feature_vector = []
            
            for category, category_features in features.items():
                if isinstance(category_features, dict):
                    for feature_name, feature_value in category_features.items():
                        if isinstance(feature_value, np.ndarray):
                            # Take mean of array features
                            feature_vector.append(np.mean(feature_value))
                        elif isinstance(feature_value, (int, float)):
                            feature_vector.append(feature_value)
            
            return np.array(feature_vector)
            
        except Exception as e:
            logger.warning(f"Error creating feature vector: {e}")
            return np.array([])