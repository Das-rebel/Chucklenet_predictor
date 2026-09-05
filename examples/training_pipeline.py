"""
Training Pipeline Example for Chucklenet Predictor

This example demonstrates how to train and fine-tune the Chucklenet Predictor models,
including:
- Data preparation and preprocessing
- Model training with multiple backends
- Hyperparameter optimization
- Evaluation and validation

Author: Subho Das
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

# Import model components
from chucklenet_predictor.models.text_humor_classifier import TextHumorClassifier
from chucklenet_predictor.models.audio_laugh_detector import AudioLaughDetector
from chucklenet_predictor.models.reverse_model import ReverseModel
from chucklenet_predictor.utils import TextProcessor, AudioProcessor, EvaluationMetrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Training pipeline for Chucklenet Predictor components.
    """
    
    def __init__(self, output_dir: str = "training_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.evaluator = EvaluationMetrics()
        
        # Model components
        self.text_classifier = None
        self.audio_detector = None
        self.reverse_model = None
        
        # Training history
        self.training_history = {
            'text_classifier': {},
            'audio_detector': {},
            'reverse_model': {},
            'overall': {}
        }
        
        logger.info(f"Training pipeline initialized. Output directory: {self.output_dir}")
    
    def create_sample_dataset(self, size: int = 1000) -> pd.DataFrame:
        """
        Create a sample dataset for training.
        """
        logger.info(f"Creating sample dataset with {size} examples...")
        
        # Sample humor texts with varying strengths
        humor_texts = [
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "The mitochondria is the powerhouse of the cell. That's what they said in biology class.",
            "Parallel lines have so much in common. It's a shame they'll never meet.",
            "I'm reading a book about anti-gravity. It's impossible to put down!",
            "Why don't scientists trust atoms? Because they make up everything!",
            "I need to go to the dentist. I have a toothache.",
            "Can a kangaroo jump higher than a house? Of course, a house doesn't jump."
        ]
        
        data = []
        
        for i in range(size):
            # Select random text
            text = np.random.choice(humor_texts)
            
            # Generate random humor strength (0-100)
            humor_strength = np.random.randint(0, 101)
            
            # Generate confidence score
            confidence = np.random.uniform(0.5, 1.0)
            
            # Generate humor type scores
            humor_types = {
                'wordplay': np.random.uniform(0, 1),
                'slapstick': np.random.uniform(0, 1),
                'absurd': np.random.uniform(0, 1),
                'irony': np.random.uniform(0, 1),
                'sarcasm': np.random.uniform(0, 1),
                'wit': np.random.uniform(0, 1),
                'pun': np.random.uniform(0, 1),
                'joke': np.random.uniform(0, 1),
                'comedy': np.random.uniform(0, 1),
                'humor': np.random.uniform(0, 1),
                'laughter': np.random.uniform(0, 1),
                'funny': np.random.uniform(0, 1)
            }
            
            # Normalize humor types
            total = sum(humor_types.values())
            if total > 0:
                for key in humor_types:
                    humor_types[key] /= total
            
            data.append({
                'id': i,
                'text': text,
                'humor_strength': humor_strength,
                'confidence': confidence,
                'humor_types': humor_types,
                'length': len(text),
                'word_count': len(text.split())
            })
        
        df = pd.DataFrame(data)
        
        # Save dataset
        dataset_path = self.output_dir / "sample_dataset.csv"
        df.to_csv(dataset_path, index=False)
        logger.info(f"Sample dataset saved to {dataset_path}")
        
        return df
    
    def preprocess_text_data(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Preprocess text data for training.
        """
        logger.info("Preprocessing text data...")
        
        # Extract features
        features_list = []
        
        for text in texts:
            try:
                features = self.text_processor.extract_all_features(text)
                
                # Flatten features
                flattened_features = []
                for category, category_features in features.items():
                    if isinstance(category_features, dict):
                        for feature_name, value in category_features.items():
                            if isinstance(value, np.ndarray) and value.size > 0:
                                # Take mean for array features
                                flattened_features.append(np.mean(value))
                            elif isinstance(value, (int, float)):
                                flattened_features.append(value)
                
                features_list.append(flattened_features)
                
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                # Use zero features as fallback
                features_list.append([0.0] * 100)  # Assume 100 features
        
        return {
            'features': np.array(features_list),
            'feature_names': self._get_feature_names()
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for text processing."""
        # This would return actual feature names in a real implementation
        return [
            'sentence_count', 'word_count', 'character_count', 'avg_sentence_length',
            'avg_word_length', 'word_density', 'exclamation_count', 'question_count',
            'punctuation_ratio', 'lexical_diversity', 'unique_word_ratio',
            'avg_syllables_per_word', 'complex_word_ratio', 'wordplay_count',
            'sarcastic_count', 'absurd_count', 'irony_count', 'surprise_count',
            'relatable_count', 'exaggeration_count', 'understatement_count',
            'irony_count_ratio', 'hyperbole_count', 'deadpan_count',
            'double_entendre_score', 'setup_punchline_ratio', 'abrupt_endings',
            'pauses', 'positive_emotion_ratio', 'negative_emotion_ratio'
        ]
    
    def preprocess_audio_data(self, audio_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Preprocess audio data for training.
        """
        logger.info("Preprocessing audio data...")
        
        features_list = []
        
        for audio_path in audio_paths:
            try:
                # Process audio file
                result = self.audio_processor.process_audio_file(audio_path)
                
                if 'error' not in result:
                    # Extract feature vectors
                    feature_vector = []
                    
                    # MFCC features
                    mfccs = result['features'].get('mfcc', np.zeros((13, 1)))
                    feature_vector.extend(np.mean(mfccs, axis=1))
                    
                    # Spectral features
                    spectral = result['features'].get('spectral', {})
                    for feature_name, values in spectral.items():
                        if len(values) > 0:
                            feature_vector.extend([np.mean(values), np.std(values)])
                    
                    # Temporal features
                    temporal = result['features'].get('temporal', {})
                    for feature_name, values in temporal.items():
                        if len(values) > 0:
                            feature_vector.extend([np.mean(values), np.std(values)])
                    
                    # Pitch features
                    pitch = result['features'].get('pitch', {})
                    for feature_name, value in pitch.items():
                        feature_vector.append(value)
                    
                    # Laughter features
                    laughter = result['features'].get('laughter', {})
                    for feature_name, value in laughter.items():
                        feature_vector.append(value)
                    
                    # Prosodic features
                    prosodic = result['features'].get('prosodic', {})
                    for feature_name, value in prosodic.items():
                        feature_vector.append(value)
                    
                    features_list.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Error processing audio {audio_path}: {e}")
                # Use zero features as fallback
                features_list.append([0.0] * 50)  # Assume 50 features
        
        return {
            'features': np.array(features_list),
            'feature_names': self._get_audio_feature_names()
        }
    
    def _get_audio_feature_names(self) -> List[str]:
        """Get feature names for audio processing."""
        # This would return actual feature names in a real implementation
        return [
            'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7',
            'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13',
            'centroid_mean', 'centroid_std', 'bandwidth_mean', 'bandwidth_std',
            'rolloff_mean', 'rolloff_std', 'flux_mean', 'flux_std', 'zcr_mean', 'zcr_std',
            'energy_mean', 'energy_std', 'amplitude_envelope_mean', 'amplitude_envelope_std',
            'temporal_centroid', 'pitch_mean', 'pitch_std', 'pitch_range', 'pitch_centroid',
            'voiced_ratio', 'high_freq_energy', 'rapid_changes', 'formant_f1', 'formant_f2',
            'formant_f3', 'voice_activity_ratio', 'burst_count', 'burst_ratio',
            'speech_rate', 'pause_count', 'pause_ratio', 'energy_variation', 'rhythm_regularity'
        ]
    
    def train_text_classifier(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Train the text humor classifier.
        """
        logger.info("Training text humor classifier...")
        
        try:
            # Initialize model
            self.text_classifier = TextHumorClassifier(
                model_name="roberta-base",
                num_classes=101,  # 0-100% humor strength
                dropout_rate=0.1
            )
            
            # Preprocess data
            train_texts = train_data['text'].tolist()
            val_texts = val_data['text'].tolist()
            
            train_features = self.preprocess_text_data(train_texts)
            val_features = self.preprocess_text_data(val_texts)
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(train_features['features'])
            y_train = torch.FloatTensor(train_data['humor_strength'].values)
            
            X_val = torch.FloatTensor(val_features['features'])
            y_val = torch.FloatTensor(val_data['humor_strength'].values)
            
            # Training parameters
            epochs = 10
            batch_size = 16
            learning_rate = 1e-4
            
            # Training loop
            self.text_classifier.train()
            optimizer = torch.optim.Adam(self.text_classifier.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                self.text_classifier.train()
                total_train_loss = 0
                
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.text_classifier(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                
                # Validation
                self.text_classifier.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    val_outputs = self.text_classifier(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val)
                    total_val_loss = val_loss.item()
                
                avg_train_loss = total_train_loss / (len(X_train) // batch_size)
                avg_val_loss = total_val_loss
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                if (epoch + 1) % 2 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Evaluation
            self.text_classifier.eval()
            with torch.no_grad():
                val_predictions = self.text_classifier(X_val).squeeze().numpy()
                val_targets = y_val.numpy()
                
                # Calculate metrics
                mae = np.mean(np.abs(val_predictions - val_targets))
                mse = np.mean((val_predictions - val_targets) ** 2)
                
                # Convert to bins for accuracy calculation
                val_pred_bins = self._humor_strength_to_bins(val_predictions)
                val_true_bins = self._humor_strength_to_bins(val_targets)
                
                accuracy = accuracy_score(val_true_bins, val_pred_bins)
            
            results = {
                'training_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses
                },
                'final_metrics': {
                    'mae': mae,
                    'mse': mse,
                    'accuracy': accuracy,
                    'final_val_loss': avg_val_loss
                }
            }
            
            # Save model
            model_path = self.output_dir / "text_classifier.pth"
            torch.save(self.text_classifier.state_dict(), model_path)
            logger.info(f"Text classifier saved to {model_path}")
            
            self.training_history['text_classifier'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training text classifier: {e}")
            return {'error': str(e)}
    
    def train_audio_detector(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Train the audio laughter detector.
        """
        logger.info("Training audio laughter detector...")
        
        try:
            # Initialize model
            self.audio_detector = AudioLaughDetector(
                model_name="openai/whisper-small",
                num_classes=2  # laugh/no laugh
            )
            
            # For demo purposes, we'll use mock audio features
            # In a real implementation, you would process actual audio files
            
            # Create mock features
            train_features = np.random.random((len(train_data), 50))  # 50 audio features
            val_features = np.random.random((len(val_data), 50))
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(train_features)
            y_train = torch.FloatTensor(train_data['humor_strength'].values > 70).float()  # Binary classification
            
            X_val = torch.FloatTensor(val_features)
            y_val = torch.FloatTensor(val_data['humor_strength'].values > 70).float()
            
            # Training parameters
            epochs = 10
            batch_size = 16
            learning_rate = 1e-4
            
            # Training loop
            self.audio_detector.train()
            optimizer = torch.optim.Adam(self.audio_detector.parameters(), lr=learning_rate)
            criterion = nn.BCELoss()
            
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            for epoch in range(epochs):
                # Training
                self.audio_detector.train()
                total_train_loss = 0
                
                # Mini-batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.audio_detector(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                
                # Validation
                self.audio_detector.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    val_outputs = self.audio_detector(X_val)
                    val_loss = criterion(val_outputs.squeeze(), y_val)
                    total_val_loss = val_loss.item()
                    
                    # Calculate accuracy
                    val_predictions = (val_outputs.squeeze() > 0.5).float()
                    val_accuracy = (val_predictions == y_val).float().mean()
                    val_accuracies.append(val_accuracy.item())
                
                avg_train_loss = total_train_loss / (len(X_train) // batch_size)
                avg_val_loss = total_val_loss
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                if (epoch + 1) % 2 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Final evaluation
            self.audio_detector.eval()
            with torch.no_grad():
                val_outputs = self.audio_detector(X_val)
                val_predictions = (val_outputs.squeeze() > 0.5).float()
                val_targets = y_val
                
                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(val_targets, val_predictions).ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                accuracy = (tn + tp) / (tn + fp + fn + tp)
            
            results = {
                'training_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies
                },
                'final_metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_positives': tp
                }
            }
            
            # Save model
            model_path = self.output_dir / "audio_detector.pth"
            torch.save(self.audio_detector.state_dict(), model_path)
            logger.info(f"Audio detector saved to {model_path}")
            
            self.training_history['audio_detector'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training audio detector: {e}")
            return {'error': str(e)}
    
    def train_reverse_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict:
        """
        Train the reverse model.
        """
        logger.info("Training reverse model...")
        
        try:
            # Initialize model
            self.reverse_model = ReverseModel(
                text_model_name="roberta-base",
                audio_model_name="openai/whisper-small",
                device="auto"
            )
            
            # Prepare training data
            train_texts = train_data['text'].tolist()
            val_texts = val_data['text'].tolist()
            
            # Training parameters
            epochs = 5
            batch_size = 8
            learning_rate = 1e-4
            
            # Training loop
            self.reverse_model.train()
            optimizer = torch.optim.Adam(self.reverse_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                self.reverse_model.train()
                total_train_loss = 0
                
                # Mini-batch training
                for i in range(0, len(train_texts), batch_size):
                    batch_texts = train_texts[i:i+batch_size]
                    batch_targets = train_data['humor_strength'].values[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Generate content
                    generated_contents = []
                    for text in batch_texts:
                        content = self.reverse_model.generate_funny_content(
                            target_strength=np.mean(batch_targets),
                            prompt=text,
                            max_iterations=10,
                            temperature=0.7
                        )
                        generated_contents.append(content)
                    
                    # Analyze generated content
                    strengths = []
                    for content in generated_contents:
                        analysis = self.reverse_model.analyze_humor_composition(text=content)
                        strengths.append(analysis.get('fusion', {}).get('humor_strength', 0))
                    
                    # Calculate loss
                    strengths_tensor = torch.FloatTensor(strengths)
                    targets_tensor = torch.FloatTensor(batch_targets)
                    loss = criterion(strengths_tensor, targets_tensor)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                
                # Validation
                self.reverse_model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    # Validate on a subset
                    val_subset = val_data.head(20)
                    val_texts_subset = val_subset['text'].tolist()
                    val_targets = val_subset['humor_strength'].values
                    
                    generated_contents = []
                    for text in val_texts_subset:
                        content = self.reverse_model.generate_funny_content(
                            target_strength=np.mean(val_targets),
                            prompt=text,
                            max_iterations=10,
                            temperature=0.7
                        )
                        generated_contents.append(content)
                    
                    # Analyze generated content
                    strengths = []
                    for content in generated_contents:
                        analysis = self.reverse_model.analyze_humor_composition(text=content)
                        strengths.append(analysis.get('fusion', {}).get('humor_strength', 0))
                    
                    # Calculate loss
                    strengths_tensor = torch.FloatTensor(strengths)
                    targets_tensor = torch.FloatTensor(val_targets)
                    val_loss = criterion(strengths_tensor, targets_tensor)
                    total_val_loss = val_loss.item()
                
                avg_train_loss = total_train_loss / (len(train_texts) // batch_size)
                avg_val_loss = total_val_loss
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                if (epoch + 1) % 1 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Final evaluation
            self.reverse_model.eval()
            with torch.no_grad():
                # Evaluate on validation set
                val_subset = val_data.head(50)
                val_texts_subset = val_subset['text'].tolist()
                val_targets = val_subset['humor_strength'].values
                
                generated_contents = []
                strengths = []
                for text in val_texts_subset:
                    content = self.reverse_model.generate_funny_content(
                        target_strength=np.mean(val_targets),
                        prompt=text,
                        max_iterations=10,
                        temperature=0.7
                    )
                    generated_contents.append(content)
                    
                    analysis = self.reverse_model.analyze_humor_composition(text=content)
                    strengths.append(analysis.get('fusion', {}).get('humor_strength', 0))
                
                # Calculate metrics
                mae = np.mean(np.abs(np.array(strengths) - val_targets))
                mse = np.mean((np.array(strengths) - val_targets) ** 2)
                
                # Convert to bins for accuracy
                pred_bins = self._humor_strength_to_bins(np.array(strengths))
                true_bins = self._humor_strength_to_bins(val_targets)
                
                accuracy = accuracy_score(true_bins, pred_bins)
            
            results = {
                'training_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses
                },
                'final_metrics': {
                    'mae': mae,
                    'mse': mse,
                    'accuracy': accuracy,
                    'final_val_loss': avg_val_loss
                }
            }
            
            # Save model
            model_path = self.output_dir / "reverse_model.pth"
            torch.save(self.reverse_model.state_dict(), model_path)
            logger.info(f"Reverse model saved to {model_path}")
            
            self.training_history['reverse_model'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error training reverse model: {e}")
            return {'error': str(e)}
    
    def _humor_strength_to_bins(self, strengths: np.ndarray, bin_size: int = 10) -> np.ndarray:
        """Convert humor strength values to bins."""
        return (strengths // bin_size).astype(int)
    
    def evaluate_complete_model(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate the complete model on test data.
        """
        logger.info("Evaluating complete model...")
        
        results = {}
        
        try:
            # Initialize complete model if not already done
            if self.text_classifier is None or self.audio_detector is None or self.reverse_model is None:
                logger.warning("Some models not trained, loading from saved files...")
                # In a real implementation, you would load saved models here
            
            # Test data
            test_texts = test_data['text'].tolist()
            test_targets = test_data['humor_strength'].values
            
            # Analyze all test texts
            predictions = []
            confidences = []
            
            for text in test_texts:
                try:
                    # Get text analysis
                    if self.text_classifier is not None:
                        # Text classifier analysis
                        pass  # In real implementation, use trained classifier
                    
                    # Reverse model analysis
                    analysis = self.reverse_model.analyze_humor_composition(text=text)
                    
                    strength = analysis.get('fusion', {}).get('humor_strength', 0)
                    confidence = analysis.get('fusion', {}).get('confidence', 0)
                    
                    predictions.append(strength)
                    confidences.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing text: {e}")
                    predictions.append(0)  # Fallback
                    confidences.append(0)
            
            # Calculate metrics
            predictions = np.array(predictions)
            confidences = np.array(confidences)
            
            mae = np.mean(np.abs(predictions - test_targets))
            mse = np.mean((predictions - test_targets) ** 2)
            
            # Convert to bins for accuracy
            pred_bins = self._humor_strength_to_bins(predictions)
            true_bins = self._humor_strength_to_bins(test_targets)
            
            accuracy = accuracy_score(true_bins, true_bins)
            f1 = f1_score(true_bins, pred_bins, average='weighted')
            
            # Evaluation using metrics class
            evaluator_results = self.evaluator.evaluate_humor_classification(
                test_targets, predictions
            )
            
            results = {
                'test_metrics': {
                    'mae': mae,
                    'mse': mse,
                    'accuracy': accuracy,
                    'f1': f1,
                    'avg_confidence': np.mean(confidences)
                },
                'detailed_evaluation': evaluator_results
            }
            
            logger.info(f"Evaluation Results:")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            
            self.training_history['overall'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating complete model: {e}")
            return {'error': str(e)}
    
    def save_training_results(self):
        """Save all training results and models."""
        logger.info("Saving training results...")
        
        try:
            # Save training history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2, default=str)
            
            # Save final model summary
            summary = {
                'text_classifier_trained': self.text_classifier is not None,
                'audio_detector_trained': self.audio_detector is not None,
                'reverse_model_trained': self.reverse_model is not None,
                'output_directory': str(self.output_dir),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            summary_path = self.output_dir / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Training results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
    
    def run_training_pipeline(self, dataset_size: int = 1000, train_ratio: float = 0.7):
        """
        Run the complete training pipeline.
        """
        logger.info("Starting complete training pipeline...")
        
        # Create dataset
        dataset = self.create_sample_dataset(dataset_size)
        
        # Split data
        train_data, val_data = train_test_split(dataset, train_ratio=train_ratio, random_state=42)
        test_data = val_data  # For demo, use validation as test
        
        logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Train individual models
        text_results = self.train_text_classifier(train_data, val_data)
        audio_results = self.train_audio_detector(train_data, val_data)
        reverse_results = self.train_reverse_model(train_data, val_data)
        
        # Evaluate complete model
        complete_results = self.evaluate_complete_model(test_data)
        
        # Save results
        self.save_training_results()
        
        # Return comprehensive results
        return {
            'text_classifier': text_results,
            'audio_detector': audio_results,
            'reverse_model': reverse_results,
            'complete_model': complete_results,
            'dataset_info': {
                'size': dataset_size,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data)
            }
        }


def main():
    """
    Main function to run the training pipeline.
    """
    print("🎭 Chucklenet Predictor - Training Pipeline")
    print("=" * 60)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(output_dir="training_output")
    
    # Run training pipeline
    results = pipeline.run_training_pipeline(dataset_size=500, train_ratio=0.7)
    
    # Print results summary
    print("\n" + "=" * 60)
    print("🎉 Training Pipeline Complete!")
    print("=" * 60)
    
    print("\n📊 Training Results Summary:")
    print("-" * 40)
    
    if 'text_classifier' in results and 'final_metrics' in results['text_classifier']:
        text_metrics = results['text_classifier']['final_metrics']
        print(f"Text Classifier - Accuracy: {text_metrics['accuracy']:.3f}, MAE: {text_metrics['mae']:.2f}")
    
    if 'audio_detector' in results and 'final_metrics' in results['audio_detector']:
        audio_metrics = results['audio_detector']['final_metrics']
        print(f"Audio Detector - F1: {audio_metrics['f1']:.3f}, Accuracy: {audio_metrics['accuracy']:.3f}")
    
    if 'reverse_model' in results and 'final_metrics' in results['reverse_model']:
        reverse_metrics = results['reverse_model']['final_metrics']
        print(f"Reverse Model - Accuracy: {reverse_metrics['accuracy']:.3f}, MAE: {reverse_metrics['mae']:.2f}")
    
    if 'complete_model' in results and 'test_metrics' in results['complete_model']:
        complete_metrics = results['complete_model']['test_metrics']
        print(f"Complete Model - Accuracy: {complete_metrics['accuracy']:.3f}, MAE: {complete_metrics['mae']:.2f}")
    
    print(f"\n📁 Results saved to training_output/")
    print("📊 Check the JSON files for detailed results!")


if __name__ == "__main__":
    main()