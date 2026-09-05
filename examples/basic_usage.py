#!/usr/bin/env python3
"""Basic Usage Example for Chucklenet Predictor"""

import sys, os, logging, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chucklenet_predictor import ChucklenetPredictor
from chucklenet_predictor.utils.evaluation_metrics import EvaluationMetrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    print("=" * 50)
    print("Chucklenet Predictor - Basic Usage")
    print("=" * 50)

    predictor = ChucklenetPredictor()
    evaluator = EvaluationMetrics(save_results=True)

    texts = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "I'm reading a book about anti-gravity. It's impossible to put down!"
    ]

    print(f"\n📝 Processing {len(texts)} samples...\n")

    results = []
    for i, text in enumerate(texts, 1):
        print(f"--- Sample {i} ---")
        print(f"Text: {text}")
        pred = predictor.predict_humor_strength(text=text)
        score = pred['fusion']['strength']
        confidence = pred['fusion']['confidence']
        print(f"  Score: {score:.1f}/100  |  Confidence: {confidence:.2f}")
        results.append({'text': text, 'score': score, 'confidence': confidence})
        print()

    ground_truth = np.array([65, 72, 58, 80, 45])
    y_pred = np.array([r['score'] for r in results])

    eval_results = evaluator.evaluate_humor_classification(ground_truth, y_pred)

    print("📊 Evaluation Results:")
    print(f"  MAE: {eval_results['regression_metrics']['mae']:.2f}")
    print(f"  RMSE: {eval_results['regression_metrics']['rmse']:.2f}")
    print(f"  R²: {eval_results['regression_metrics']['r2']:.3f}")
    print(f"  Accuracy: {eval_results['classification_metrics']['accuracy']:.1%}")

    Path("evaluation_results").mkdir(exist_ok=True)
    import json
    with open("evaluation_results/basic_results.json", 'w') as f:
        json.dump({'predictions': results, 'eval': eval_results}, f, default=str, indent=2)

    print("\n✅ Done! Results saved to evaluation_results/basic_results.json")
    print("=" * 50)

if __name__ == "__main__":
    main()
