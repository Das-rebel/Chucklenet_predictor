#!/usr/bin/env python3
"""
Training Script for Chucklenet Predictor

Trains the humor strength prediction model on the converted word-level laughter data.
"""

import sys
import os
import json
import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chucklenet_predictor import ChucklenetPredictor

def load_training_data(data_file: str):
    """Load training data from JSONL file."""
    texts = []
    targets = []
    
    with open(data_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
            targets.append(data['humor_strength'])
    
    return texts, targets

def main():
    """Main training pipeline."""
    data_file = "/Users/Subho/funny-strength-predictor/training_data.jsonl"
    model_save_dir = "/Users/Subho/funny-strength-predictor/models/"
    Path(model_save_dir).mkdir(exist_ok=True)
    
    print("Loading training data...")
    texts, targets = load_training_data(data_file)
    
    print(f"Loaded {len(texts)} training samples")
    
    # Initialize model
    print("Initializing Chucklenet Predictor...")
    predictor = ChucklenetPredictor()
    
    # Train the text model
    if texts:
        print("Starting training...")
        results = predictor.train_on_dataset(
            texts=texts,
            strength_targets=targets
        )
    
    print(f"Training results: {results}")
    
    # Save the model
    print("Saving model...")
    model_path = Path(model_save_dir) / "chucklenet_humor_predictor.pt"
    torch.save(predictor.text_classifier.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return True

if __name__ == "__main__":
    main()