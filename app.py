#!/usr/bin/env python3
"""Chucklenet Predictor demo — text -> humor strength 0-100. Run: python app.py"""
import pickle, torch, numpy as np, gradio as gr
from transformers import AutoTokenizer, AutoModel
from torch import nn

MODEL_PATH = "/Users/Subho/models/chuckle_predictor/roberta_v7.pt"
# Download from Kaggle if missing:
#   kaggle kernels output subhajitdas/chucklenet-predictor-v7-train -p /tmp/v7_out
#   cp /tmp/v7_out/roberta_v7.pt "$MODEL_PATH"
import os
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. "
        "Run: kaggle kernels output subhajitdas/chucklenet-predictor-v7-train -p /tmp/v7_out")

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
model = Regressor()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def score(text):
    if not text.strip(): return "Enter a joke."
    e = tok(text, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        pred = float(model(e["input_ids"], e["attention_mask"])[0]) * 100
    return (
        f"Humor strength: **{pred:.0f}/100**\n\n"
        f"Eval context: Spearman rho=0.3226 (v7, len256), Jester gold rho=0.5659. "
        f"TF-IDF baseline: 0.277. MAE ~23 on validation."
    )

gr.Interface(
    fn=score,
    inputs=gr.Textbox(lines=4, label="Joke or humorous text"),
    outputs=gr.Text(label="Humor strength"),
    title="Chucklenet Predictor — Humor Strength Regression",
    description=(
        "Predicts how funny a joke or humorous text is on a 0-100 scale. "
        "Trained on 7,984 real-labeled jokes (Jester gold + upvote-weak). "
        "v7: RoBERTa-base regression, len256, Spearman rho=0.3226 overall, 0.5659 on Jester gold."
    ),
    examples=[
        ["Why don't scientists trust atoms? Because they make up everything!"],
        ["I'm reading a book about anti-gravity. It's impossible to put down!"],
        ["I told my wife she was drawing her eyebrows too high. She looked surprised."],
        ["Why did the scarecrow win an award? Because he was outstanding in his field."],
        ["I used to hate facial hair, but then it grew on me."],
    ]
).launch()
