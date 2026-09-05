"""
Text Humor Classifier

This module implements text-based humor detection using foundation models
from the AI-research-SKILLS repository, particularly CLIP and Transformers.

Author: Subho Das
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Optional, Tuple


class TextHumorClassifier(nn.Module):
    """
    Text-based humor classifier that predicts humor strength from text content.
    
    Uses a combination of:
    - CLIP for multimodal text encoding
    - BERT/RoBERTa for semantic humor understanding
    - Custom humor-specific classification head
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "roberta-base",
        humor_classes: int = 101,  # 0-100% funny strength
        device: str = "auto"
    ):
        super().__init__()
        
        self.device = self._get_device(device)
        self.humor_classes = humor_classes
        
        # Initialize CLIP for multimodal understanding
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.to(self.device)
        
        # Initialize text model for semantic understanding
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            text_model_name,
            num_labels=humor_classes,
            ignore_mismatched_sizes=True
        )
        self.text_model.to(self.device)
        
        # Sentence transformer for additional text embeddings
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_encoder.to(self.device)
        
        # Custom fusion and classification layers
        self.text_fusion = nn.Sequential(
            nn.Linear(768 + 512, 512),  # BERT + CLIP dims
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.humor_classifier = nn.Linear(256 + 128 + 384, humor_classes)
        
        # Humor-specific features
        self.humor_feature_extractors = nn.ModuleDict({
            'incongruity': nn.Linear(768, 128),
            'surprise': nn.Linear(768, 128),
            'relatability': nn.Linear(768, 128),
            'wordplay': nn.LSTM(768, 128, batch_first=True),
            'timing': nn.LSTM(768, 128, batch_first=True),
        })

        self.humor_fusion = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def extract_clip_features(self, text: str) -> torch.Tensor:
        """Extract CLIP text embeddings."""
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
        return text_features
    
    def extract_text_features(self, text: str) -> torch.Tensor:
        """Extract BERT/RoBERTa text embeddings."""
        tokenizer = AutoTokenizer.from_pretrained(self.text_model.config.name_or_path)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs, output_hidden_states=True)
            text_features = outputs.hidden_states[-1]  # Last hidden state
            text_features = text_features.mean(dim=1)  # Mean pooling
            
        return text_features
    
    def extract_humor_features(self, text: str) -> torch.Tensor:
        """Extract humor-specific features."""
        tokenizer = AutoTokenizer.from_pretrained(self.text_model.config.name_or_path)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
        humor_features = []
        
        # Extract different humor types
        for feature_type, extractor in self.humor_feature_extractors.items():
            if isinstance(extractor, nn.LSTM):
                # LSTM-based features for wordplay and timing
                word_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                word_tokens = {k: v.to(self.device) for k, v in word_tokens.items()}
                # Get BERT embeddings, not raw token IDs
                with torch.no_grad():
                    bert_outputs = self.text_model(**word_tokens, output_hidden_states=True)
                    bert_embeddings = bert_outputs.hidden_states[-1]  # [batch, seq, 768]
                word_features, _ = extractor(bert_embeddings.float())
                humor_features.append(word_features.mean(dim=1))
            else:
                # Linear layer features
                feature = extractor(hidden_states.mean(dim=1))
                humor_features.append(feature)
        
        return torch.cat(humor_features, dim=1)
    
    def forward(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Forward pass for text humor classification.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary containing:
            - humor_scores: Predicted humor strength (0-100%)
            - confidence: Confidence in prediction
            - features: Extracted features for further analysis
        """
        # Extract multiple text representations
        clip_features = self.extract_clip_features(text)
        text_features = self.extract_text_features(text)
        humor_features = self.extract_humor_features(text)
        
        # Sentence embeddings for additional context
        sentence_embeddings = self.sentence_encoder.encode([text], convert_to_tensor=True)
        sentence_embeddings = sentence_embeddings.to(self.device)
        
        # Fusion of different text representations
        combined_features = torch.cat([text_features, clip_features], dim=1)
        fused_text = self.text_fusion(combined_features)
        
        # Humor-specific feature processing
        humor_fused = self.humor_fusion(humor_features)
        
        # Final humor classification
        combined_fused = torch.cat([fused_text, humor_fused, sentence_embeddings], dim=1)
        humor_logits = self.humor_classifier(combined_fused)
        
        # Apply softmax to get probabilities
        humor_probs = F.softmax(humor_logits, dim=-1)
        
        # Convert to percentage (0-100%)
        humor_scores = (humor_probs * torch.arange(self.humor_classes, device=self.device).float()).sum(dim=1)
        
        # Calculate confidence (max probability)
        confidence = humor_probs.max(dim=-1).values
        
        return {
            'humor_scores': humor_scores,
            'confidence': confidence,
            'clip_features': clip_features,
            'text_features': text_features,
            'humor_features': humor_features,
            'sentence_embeddings': sentence_embeddings,
        }
    
    def predict_humor_strength(self, text: str) -> Tuple[float, float]:
        """
        Predict humor strength as a percentage (0-100%) and confidence.
        
        Args:
            text: Input text string
            
        Returns:
            Tuple of (humor_strength, confidence)
        """
        self.eval()
        with torch.no_grad():
            results = self(text)
            humor_strength = results['humor_scores'].cpu().numpy()[0]
            confidence = results['confidence'].cpu().numpy()[0]
            
        return float(humor_strength), float(confidence)
    
    def batch_predict(self, texts: List[str]) -> List[Tuple[float, float]]:
        """
        Batch prediction for multiple texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of tuples (humor_strength, confidence) for each text
        """
        self.eval()
        results = []
        
        for text in texts:
            strength, confidence = self.predict_humor_strength(text)
            results.append((strength, confidence))
            
        return results
    
    def analyze_humor_types(self, text: str) -> Dict[str, float]:
        """
        Analyze specific humor types in the text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with humor type scores
        """
        self.eval()
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained(self.text_model.config.name_or_path)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.text_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            humor_scores = {}
            
            for feature_type, extractor in self.humor_feature_extractors.items():
                if 'lstm' in str(extractor):
                    # LSTM features
                    word_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    word_tokens = {k: v.to(self.device) for k, v in word_tokens.items()}
                    features = extractor(word_tokens['input_ids'])
                    score = features.mean().item()
                else:
                    # Linear layer features
                    features = extractor(hidden_states.mean(dim=1))
                    score = features.mean().item()
                
                humor_scores[feature_type] = score
            
            return humor_scores
    
    def train_on_dataset(self, texts, strength_targets, epochs=3, lr=1e-5, batch_size=4):
        """
        Train the model on text-humor strength pairs.

        Args:
            texts: List of text strings
            strength_targets: List of humor strength values (0-100)
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size

        Returns:
            Dict with training results
        """
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader

        class HumorDataset(Dataset):
            def __init__(self, texts, targets):
                self.texts = texts
                self.targets = targets

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                # Handle None values gracefully
                if self.texts[idx] is None or self.targets[idx] is None:
                    # Return a dummy sample
                    return "", torch.tensor(0.0, dtype=torch.float32)
                return self.texts[idx], torch.tensor(self.targets[idx], dtype=torch.float32)

        # Filter out None values first
        valid_pairs = [(t, s) for t, s in zip(texts, strength_targets) 
                       if t is not None and s is not None]
        if not valid_pairs:
            return {"error": "No valid training samples"}
        
        texts_clean, targets_clean = zip(*valid_pairs)
        texts_clean = list(texts_clean)
        targets_clean = list(targets_clean)

        # Use a smaller subset for quick training due to heavy model
        dataset_size = len(texts_clean)
        sample_size = min(dataset_size, 256)
        indices = torch.randperm(dataset_size)[:sample_size].tolist()
        
        texts_sample = [texts_clean[i] for i in indices]
        targets_sample = [targets_clean[i] for i in indices]

        dataset = HumorDataset(texts_sample, targets_sample)
        loader = DataLoader(dataset, batch_size=max(1, batch_size), shuffle=True)

        optimizer = optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch_texts, batch_targets in loader:
                optimizer.zero_grad()
                batch_loss = 0.0
                valid_samples = 0
                for i, text in enumerate(batch_texts):
                    if text == "":  # Skip dummy samples
                        continue
                    try:
                        result = self(text)
                        pred = result['humor_scores']
                        target = batch_targets[i].to(self.device)
                        loss = criterion(pred.unsqueeze(0), target.unsqueeze(0))
                        batch_loss += loss
                        valid_samples += 1
                    except Exception as e:
                        continue
                if batch_loss > 0 and valid_samples > 0:
                    batch_loss.backward()
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                    batch_count += 1
            avg_loss = epoch_loss / max(1, batch_count)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        return {"epochs": epochs, "final_loss": losses[-1], "losses": losses}

    def save_pretrained(self, save_directory: str):
        """Save the model to the specified directory."""
        os.makedirs(save_directory, exist_ok=True)
        self.save_pretrained(save_directory)
        self.clip_processor.save_pretrained(save_directory)
        
        # Save additional configuration
        config = {
            'clip_model_name': self.clip_model.config.name_or_path,
            'text_model_name': self.text_model.config.name_or_path,
            'humor_classes': self.humor_classes,
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
            clip_model_name=config['clip_model_name'],
            text_model_name=config['text_model_name'],
            humor_classes=config['humor_classes'],
            device=config.get('device', 'auto'),
            **kwargs
        )
        
        # Load pretrained weights
        model.load_state_dict(torch.load(os.path.join(load_directory, 'pytorch_model.bin')))
        
        return model