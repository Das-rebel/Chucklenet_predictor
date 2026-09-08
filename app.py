#!/usr/bin/env python3
"""Chucklenet Predictor demo — text -> humor strength 0-100 (calibrated). Run: python app.py"""
import pickle, torch, numpy as np, gradio as gr
from transformers import AutoTokenizer, AutoModel
from torch import nn

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained("roberta-base")
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(768, 1)
    def forward(self, ids, am):
        h = self.enc(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float()
        pooled = (h*m).sum(1)/m.sum(1).clamp(min=1e-9)
        return torch.sigmoid(self.drop(pooled) @ self.head.weight.t() + self.head.bias).squeeze(-1)

tok = AutoTokenizer.from_pretrained("roberta-base")
model = Regressor(); model.load_state_dict(torch.load("training_output/roberta_regression_best.pt", map_location="cpu")); model.eval()
cal = pickle.load(open("training_output/isotonic_calibrator.pkl","rb"))

def score(text):
    if not text.strip(): return "Enter a joke."
    e = tok(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        raw = float(model(e["input_ids"], e["attention_mask"])[0]) * 100
    c = cal["iso" if cal["chosen"]=="isotonic" else "lin"]
    calibrated = float(c.predict(np.array([raw]))[0]) if cal["chosen"]=="isotonic" else float(c.predict(np.array([[raw]]))[0])
    return f"Raw: {raw:.1f}/100 -> Calibrated: {calibrated:.1f}/100 (honest eval: rho=0.31, MAE ~23 — see results JSONs)"

gr.Interface(fn=score, inputs=gr.Textbox(lines=4, label="Joke text"), outputs=gr.Text(label="Humor strength"),
             title="Chucklenet Predictor v1", description="Humor strength regression (PRD v1.0 M4 demo). Eval: Spearman 0.313 vs TF-IDF+Ridge 0.277.").launch()
