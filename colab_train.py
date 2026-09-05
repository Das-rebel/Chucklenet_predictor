#!/usr/bin/env python3
"""
Colab Training Script for Chucklenet Predictor
Run this on Google Colab with GPU (T4 free tier sufficient)

Upload kaggle_data/ directory to Colab or download via Kaggle API.
"""

import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys

sys.path.insert(0, 'src')
from chucklenet_predictor.models.text_humor_classifier import TextHumorClassifier

# ===== CONFIG =====
DATA_PATH = 'kaggle_data/train.jsonl'
MODEL_SAVE = 'training_output/text_classifier_trained.pt'
EPOCHS = 5
BATCH_SIZE = 16
LR = 3e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==================

def load_jsonl(path):
    texts, scores = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            texts.append(d['text'])
            scores.append(d['humor_strength'])
    return texts, scores

class HumorDataset(Dataset):
    def __init__(self, texts, scores):
        self.texts = texts
        self.scores = scores
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.scores[idx]

print(f"Loading data from {DATA_PATH}...")
texts, scores = load_jsonl(DATA_PATH)
print(f"Total samples: {len(texts)}, Score range: {min(scores):.0f}-{max(scores):.0f}")

print(f"\nDevice: {DEVICE}")
print("Loading model...")
model = TextHumorClassifier(
    clip_model_name="openai/clip-vit-base-patch32",
    text_model_name="roberta-base",
    humor_classes=101,
    device=DEVICE
)

dataset = HumorDataset(texts, scores)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.MSELoss()

Path("training_output").mkdir(exist_ok=True)

print(f"\nTraining {EPOCHS} epochs on {len(texts)} samples...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batches = 0
    
    for batch_texts, batch_scores in loader:
        targets = torch.FloatTensor([s/100 for s in batch_scores]).to(DEVICE)
        
        outputs = model(batch_texts)
        logits = outputs['logits']
        
        loss = criterion(logits.squeeze(-1), targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    avg_loss = total_loss / batches
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.2f}")

print(f"\nSaving to {MODEL_SAVE}...")
torch.save(model.state_dict(), MODEL_SAVE)

# Quick test
test_jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "The mitochondria is the powerhouse of the cell.",
]
print("\nSample predictions:")
model.eval()
for joke in test_jokes:
    out = model(joke)
    print(f"  [{out['humor_scores'].item():.0f}/100] {joke[:60]}...")

print("\n✅ Training complete!")
