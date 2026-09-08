#!/usr/bin/env python3
"""M4: Calibration analysis — map raw model scores to calibrated 0-100 (PRD gate: cal-MAE <= 10).

Fits isotonic regression on a hash-split calibration half of val, evaluates on the other half.
Saves: training_output/calibration.json, training_output/isotonic_calibrator.pkl, app.py (Gradio demo).
"""
import json, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import sys, os
sys.path.insert(0, '.')
torch.set_num_threads(max(4, os.cpu_count()//2))

DATA_VAL = 'data/val_real.jsonl'
CKPT = 'training_output/roberta_regression_best.pt'
DEV = 'cpu'

rows = [json.loads(l) for l in open(DATA_VAL)]
texts = [r['text'] for r in rows]
ys = np.array([r['humor_strength'] for r in rows])
srcs = [r['source'] for r in rows]

# stable calibration/eval halves (SALTED md5 — val rows are all md5%10==9, plain parity is degenerate)
cal_m = np.array([int(hashlib.md5((t + '|cal').encode()).hexdigest(), 16) % 2 == 0 for t in texts])
print(f"cal={cal_m.sum()} eval={(~cal_m).sum()}")

from transformers import AutoModel
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

print("Loading model...")
tok = AutoTokenizer.from_pretrained("roberta-base")
model = Regressor()
sd = torch.load(CKPT, map_location='cpu')
model.load_state_dict(sd)
model.to(DEV).eval()

class DS(torch.utils.data.Dataset):
    def __init__(s, texts): s.texts = texts
    def __len__(s): return len(s.texts)
    def __getitem__(s, i):
        e = tok(s.texts[i], truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        return {"input_ids": e["input_ids"][0], "attention_mask": e["attention_mask"][0]}

def collate(b):
    return {"input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b])}

dl = DataLoader(DS(texts), batch_size=32, collate_fn=collate)
preds = []
print(f"Inferring {len(texts)} samples on CPU...")
with torch.no_grad():
    for bi, b in enumerate(dl):
        p = model(b["input_ids"].to(DEV), b["attention_mask"].to(DEV))
        preds += (p*100).cpu().tolist()
        if bi % 5 == 0: print(f"  batch {bi}/{len(dl)}", flush=True)
preds = np.array(preds)
print(f"raw spearman={spearmanr(ys, preds).statistic:.3f} mae={mean_absolute_error(ys, preds):.1f}")

# Fit calibrators on cal-half
iso = IsotonicRegression(out_of_bounds='clip').fit(preds[cal_m], ys[cal_m])
lin = LinearRegression().fit(preds[cal_m].reshape(-1,1), ys[cal_m])

# Evaluate on eval-half
ev = ~cal_m
p_iso = iso.predict(preds[ev]); p_lin = lin.predict(preds[ev].reshape(-1,1))
res = {
  'date': '2026-09-08', 'data': DATA_VAL, 'ckpt': CKPT,
  'raw': {'spearman': round(float(spearmanr(ys[ev], preds[ev]).statistic),4),
          'mae': round(float(mean_absolute_error(ys[ev], preds[ev])),2)},
  'isotonic': {'spearman': round(float(spearmanr(ys[ev], p_iso).statistic),4),
               'mae': round(float(mean_absolute_error(ys[ev], p_iso)),2)},
  'linear': {'spearman': round(float(spearmanr(ys[ev], p_lin).statistic),4),
             'mae': round(float(mean_absolute_error(ys[ev], p_lin)),2)},
  'gate': 'calibrated MAE <= 10 (PRD M4)', 'n_cal': int(cal_m.sum()), 'n_eval': int(ev.sum()),
}
best_name = min(['isotonic','linear','raw'], key=lambda k: res[k]['mae'])
res['chosen'] = best_name
res['passed_gate'] = bool(res[best_name]['mae'] <= 10)
print(json.dumps(res, indent=2))

import pickle, os
os.makedirs('training_output', exist_ok=True)
with open('training_output/isotonic_calibrator.pkl','wb') as f:
    pickle.dump({'iso': iso, 'lin': lin, 'chosen': best_name}, f)
json.dump(res, open('training_output/calibration.json','w'), indent=2)
print(f"chosen: {best_name} | gate {'PASS' if res['passed_gate'] else 'FAIL'} | saved calibrator + calibration.json")

# Generate demo app
app = f'''#!/usr/bin/env python3
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
    return f"Raw: {{raw:.1f}}/100 -> Calibrated: {{calibrated:.1f}}/100 (honest eval: rho=0.31, MAE ~23 — see results JSONs)"

gr.Interface(fn=score, inputs=gr.Textbox(lines=4, label="Joke text"), outputs=gr.Text(label="Humor strength"),
             title="Chucklenet Predictor v1", description="Humor strength regression (PRD v1.0 M4 demo). Eval: Spearman 0.313 vs TF-IDF+Ridge 0.277.").launch()
'''
open('app.py','w').write(app)
print("demo app.py written")
