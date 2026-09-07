#!/usr/bin/env python3
"""M1 baseline: TF-IDF + Ridge on real-rated data (PRD §4 — transformer must beat this)."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r['text'] for r in rows], np.array([r['humor_strength'] for r in rows]), [r.get('source','?') for r in rows]

train_texts, train_y, train_src = load('data/train_real.jsonl')
val_texts, val_y, val_src = load('data/val_real.jsonl')
print(f"train={len(train_y)} val={len(val_y)}")

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=60000, sublinear_tf=True)
Xtr = vec.fit_transform(train_texts)
Xva = vec.transform(val_texts)

model = Ridge(alpha=1.0)
model.fit(Xtr, train_y)
pred = model.predict(Xva)

rho_all = spearmanr(val_y, pred).statistic
mask = np.array(val_src) == 'jester'
rho_jester = spearmanr(val_y[mask], pred[mask]).statistic if mask.sum() >= 5 else float('nan')
mae = mean_absolute_error(val_y, pred)

print(f"\n=== M1 Baseline: TF-IDF(1-2g) + Ridge ===")
print(f"Spearman (all val):   {rho_all:.3f}   [target >= 0.35]")
print(f"Spearman (Jester):    {rho_jester:.3f}  (n={mask.sum()})")
print(f"MAE:                  {mae:.1f} pts")
print(f"Pred range:           {pred.min():.0f}–{pred.max():.0f} (std {pred.std():.1f})")

results = {
    'model': 'tfidf_ridge_m1', 'date': '2026-09-08',
    'data': 'data/train_real.jsonl + data/val_real.jsonl (v1 rebuild, see data/REBUILD_NOTES.md)',
    'spearman_all': round(float(rho_all), 4),
    'spearman_jester_gold': round(float(rho_jester), 4) if not np.isnan(rho_jester) else None,
    'mae': round(float(mae), 2),
    'n_train': len(train_y), 'n_val': len(val_y),
    'gate': 'PRD M1: spearman_all >= 0.35',
    'passed_gate': bool(rho_all >= 0.35),
}
import os; os.makedirs('evaluation_results', exist_ok=True)
json.dump(results, open('evaluation_results/m1_tfidf_ridge_baseline.json', 'w'), indent=2)
print(f"\nSaved -> evaluation_results/m1_tfidf_ridge_baseline.json (gate {'PASS ✅' if results['passed_gate'] else 'FAIL ❌'})")
