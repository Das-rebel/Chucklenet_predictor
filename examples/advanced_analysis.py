#!/usr/bin/env python3
"""
Advanced Analysis Example for Chucklenet Predictor

This script demonstrates advanced analysis capabilities including
component analysis, comparative evaluation, and detailed profiling.

Author: Subho Das
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chucklenet_predictor import ChucklenetPredictor
from chucklenet_predictor.utils.text_processor import TextProcessor
from chucklenet_predictor.utils.evaluation_metrics import EvaluationMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def advanced_text_analysis():
    """Perform advanced text analysis with detailed profiling."""
    print("\n🔬 Advanced Text Analysis")
    print("=" * 50)
    
    # Initialize components
    text_processor = TextProcessor()
    predictor = ChucklenetPredictor()
    
    # Complex humor examples
    complex_examples = [
        {
            'text': "I asked my dog what's two minus two. He said nothing.",
            'type': 'pun',
            'expected_category': 'Witty'
        },
        {
            'text': "I haven't slept for ten days, because that would be too long.",
            'type': 'absurd',
            'expected_category': 'Surprising'
        },
        {
            'text': "Why don't skeletons fight each other? They don't have the guts.",
            'type': 'wordplay',
            'expected_category': 'Clever'
        },
        {
            'text': "I used to hate facial hair, but then it grew on me.",
            'type': 'pun',
            'expected_category': 'Witty'
        },
        {
            'text': "I'm writing a book on reverse psychology. Please don't read it.",
            'type': 'meta',
            'expected_category': 'Ironic'
        }
    ]
    
    print(f"Analyzing {len(complex_examples)} complex humor examples...")
    
    detailed_results = []
    
    for i, example in enumerate(complex_examples, 1):
        print(f"\n--- Example {i}: {example['type']} ---")
        print(f"Text: {example['text']}")
        print(f"Expected category: {example['expected_category']}")
        
        # Process text
        text_features = text_processor.preprocess_text(example['text'])
        
        # Advanced analysis
        humor_prediction = predictor.predict_humor_strength(example['text'], text_features)
        component_analysis = predictor.analyze_humor_components(example['text'], text_features)
        reverse_analysis = predictor.reverse_model_desired_output(example['text'], humor_prediction['score'])
        comparative_analysis = predictor.compare_humor_styles(example['text'])
        
        # Compile detailed result
        detailed_result = {
            'example_num': i,
            'text': example['text'],
            'type': example['type'],
            'expected_category': example['expected_category'],
            'predicted_score': humor_prediction['score'],
            'predicted_category': humor_prediction['category'],
            'confidence': humor_prediction['confidence'],
            'component_analysis': component_analysis,
            'reverse_analysis': reverse_analysis,
            'comparative_analysis': comparative_analysis,
            'text_features': text_features['features']
        }
        
        detailed_results.append(detailed_result)
        
        # Print detailed analysis
        print(f"Predicted Score: {humor_prediction['score']:.1f}/100")
        print(f"Predicted Category: {humor_prediction['category']}")
        print(f"Confidence: {humor_prediction['confidence']:.2f}")
        print(f"Component Analysis:")
        print(f"  - Wit: {component_analysis['wit_level']:.1f}/10")
        print(f"  - Sarcasm: {component_analysis['sarcasm_level']:.1f}/10")
        print(f"  - Irony: {component_analysis['irony_level']:.1f}/10")
        print(f"  - Surprise: {component_analysis['surprise_level']:.1f}/10")
        print(f"  - Absurdity: {component_analysis['absurdity_level']:.1f}/10")
        
        print(f"Reverse Analysis:")
        print(f"  - Desired characteristic: {reverse_analysis['desired_characteristic']}")
        print(f"  - Intensity: {reverse_analysis['intensity']:.1f}/10")
        print(f"  - Key phrases: {', '.join(reverse_analysis['key_phrases'])}")
        
        print(f"Style Comparison:")
        print(f"  - vs Wit: {comparative_analysis['wit_comparison']['score']:.1f}")
        print(f"  - vs Sarcastic: {comparative_analysis['sarcastic_comparison']['score']:.1f}")
        print(f"  - vs Absurd: {comparative_analysis['absurd_comparison']['score']:.1f}")
        
        # Category match check
        category_match = (humor_prediction['category'] == example['expected_category'])
        print(f"Category Match: {'✅' if category_match else '❌'}")
        
    return detailed_results

def comparative_model_analysis():
    """Compare different humor prediction approaches."""
    print("\n🎭 Comparative Model Analysis")
    print("=" * 50)
    
    # Initialize components
    text_processor = TextProcessor()
    predictor = ChucklenetPredictor()
    
    # Test samples
    test_samples = [
        "Why don't eggs tell jokes? They'd crack each other up!",
        "I'm reading a book about anti-gravity. It's impossible to put down!",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
        "I used to be a baker, but I couldn't make enough dough.",
        "Why don't skeletons fight each other? They don't have the guts."
    ]
    
    print(f"Comparing humor prediction models on {len(test_samples)} samples...")
    
    comparison_results = []
    
    for i, text in enumerate(test_samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Text: {text}")
        
        # Process text
        text_features = text_processor.preprocess_text(text)
        
        # Different prediction approaches
        base_prediction = predictor.predict_humor_strength(text, text_features)
        
        # Component-based prediction
        component_analysis = predictor.analyze_humor_components(text, text_features)
        
        # Reverse modeling
        reverse_prediction = predictor.reverse_model_desired_output(text, 70)  # Target 70% humor
        
        # Style comparison
        style_comparison = predictor.compare_humor_styles(text)
        
        # Compile comparison
        comparison_result = {
            'sample_num': i,
            'text': text,
            'base_prediction': {
                'score': base_prediction['score'],
                'category': base_prediction['category'],
                'confidence': base_prediction['confidence']
            },
            'component_analysis': component_analysis,
            'reverse_prediction': reverse_prediction,
            'style_comparison': style_comparison
        }
        
        comparison_results.append(comparison_result)
        
        # Print comparison
        print(f"Base Prediction: {base_prediction['score']:.1f} ({base_prediction['category']})")
        print(f"Component Analysis - Wit: {component_analysis['wit_level']:.1f}, Sarcasm: {component_analysis['sarcasm_level']:.1f}")
        print(f"Reverse Target: {reverse_prediction['desired_characteristic']} (Intensity: {reverse_prediction['intensity']:.1f})")
        print(f"Style Best Match: {style_comparison['best_match_style']} ({style_comparison['best_match_score']:.1f})")
    
    return comparison_results

def humor_trend_analysis():
    """Analyze humor trends across different styles."""
    print("\n📈 Humor Trend Analysis")
    print("=" * 50)
    
    # Initialize components
    text_processor = TextProcessor()
    predictor = ChucklenetPredictor()
    
    # Create joke sets by type
    joke_sets = {
        'puns': [
            "I'm reading a book about anti-gravity. It's impossible to put down!",
            "I used to be a baker, but I couldn't make enough dough.",
            "Why don't skeletons fight each other? They don't have the guts.",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "The bicycle couldn't stand on its own because it was two tired."
        ],
        'absurdist': [
            "I asked my dog what's two minus two. He said nothing.",
            "Why did the scarecrow win an award? Because he was outstanding in his field.",
            "I haven't slept for ten days, because that would be too long.",
            "What do you call a bear with no teeth? A gummy bear!",
            "I'm on a whiskey diet. I've lost three days already."
        ],
        'witty': [
            "Parallel lines have so much in common. It's a shame they'll never meet.",
            "I told my computer I needed a break, and now it won't stop sending me Kit-Kats.",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "I'm friends with 25 letters of the alphabet. I don't know why.",
            "Why don't scientists trust atoms? Because they make up everything!"
        ],
        'sarcasm': [
            "Oh great, another meeting. I just love sitting in a room doing nothing.",
            "Oh wonderful, traffic. I haven't sat in a car doing nothing for hours.",
            "Great, another Monday. I was really looking forward to getting up at 6 AM.",
            "Oh fantastic, I get to work overtime again. I love giving away my free time.",
            "Wonderful, the WiFi's down. I was really looking forward to being disconnected."
        ]
    }
    
    print("Analyzing humor trends across different styles...")
    
    trend_analysis = {}
    
    for humor_style, jokes in joke_sets.items():
        print(f"\n--- {humor_style.capitalize()} Style ({len(jokes)} jokes) ---")
        
        style_scores = []
        style_categories = []
        component_scores = {
            'wit': [],
            'sarcasm': [],
            'irony': [],
            'surprise': [],
            'absurdity': []
        }
        
        for joke in jokes:
            # Process joke
            text_features = text_processor.preprocess_text(joke)
            humor_prediction = predictor.predict_humor_strength(joke, text_features)
            component_analysis = predictor.analyze_humor_components(joke, text_features)
            
            style_scores.append(humor_prediction['score'])
            style_categories.append(humor_prediction['category'])
            
            # Collect component scores
            component_scores['wit'].append(component_analysis['wit_level'])
            component_scores['sarcasm'].append(component_analysis['sarcasm_level'])
            component_scores['irony'].append(component_analysis['irony_level'])
            component_scores['surprise'].append(component_analysis['surprise_level'])
            component_scores['absurdity'].append(component_analysis['absurdity_level'])
        
        # Calculate trend statistics
        trend_analysis[humor_style] = {
            'average_score': np.mean(style_scores),
            'score_std': np.std(style_scores),
            'score_range': (np.min(style_scores), np.max(style_scores)),
            'category_distribution': pd.Series(style_categories).value_counts().to_dict(),
            'component_averages': {
                component: np.mean(scores) for component, scores in component_scores.items()
            },
            'sample_scores': style_scores
        }
        
        # Print trend summary
        print(f"Average Score: {np.mean(style_scores):.1f}/100")
        print(f"Score Range: {np.min(style_scores):.1f} - {np.max(style_scores):.1f}")
        print(f"Most Common Category: {max(pd.Series(style_categories).value_counts(), key=pd.Series(style_categories).value_counts().get)}")
        
        component_summary = ", ".join([f"{component}: {np.mean(scores):.1f}" 
                                    for component, scores in component_scores.items()])
        print(f"Component Averages: {component_summary}")
    
    # Overall trend comparison
    print(f"\n🎯 Overall Trend Analysis:")
    print(f"Most Consistent Style: {min(trend_analysis.keys(), key=lambda k: trend_analysis[k]['score_std'])}")
    print(f"Most Variable Style: {max(trend_analysis.keys(), key=lambda k: trend_analysis[k]['score_std'])}")
    print(f"Highest Average Score: {max(trend_analysis.keys(), key=lambda k: trend_analysis[k]['average_score'])}")
    print(f"Most Predictable Category: {pd.Series([cat for style in trend_analysis.values() for cat in style['category_distribution'].keys()]).value_counts().index[0]}")
    
    return trend_analysis

def batch_analysis():
    """Perform batch analysis on multiple samples."""
    print("\n📦 Batch Analysis")
    print("=" * 50)
    
    # Initialize components
    text_processor = TextProcessor()
    predictor = ChucklenetPredictor()
    evaluator = EvaluationMetrics(save_results=True)
    
    # Batch of text samples
    batch_samples = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "I'm reading a book about anti-gravity. It's impossible to put down!",
        "I asked my dog what's two minus two. He said nothing.",
        "Why did the scarecrow win an award? Because he was outstanding in his field.",
        "I haven't slept for ten days, because that would be too long.",
        "What do you call a bear with no teeth? A gummy bear!",
        "I'm on a whiskey diet. I've lost three days already."
    ]
    
    print(f"Processing batch of {len(batch_samples)} samples...")
    
    batch_results = []
    
    # Process batch
    for i, text in enumerate(batch_samples, 1):
        print(f"\n--- Processing sample {i}/{len(batch_samples)} ---")
        
        # Process text
        text_features = text_processor.preprocess_text(text)
        
        # Predict
        humor_prediction = predictor.predict_humor_strength(text, text_features)
        
        # Analyze components
        component_analysis = predictor.analyze_humor_components(text, text_features)
        
        # Store result
        result = {
            'sample_num': i,
            'text': text,
            'prediction': humor_prediction,
            'components': component_analysis,
            'features': text_features['features']
        }
        
        batch_results.append(result)
        print(f"Score: {humor_prediction['score']:.1f} ({humor_prediction['category']})")
    
    # Batch evaluation
    print(f"\n📊 Batch Evaluation Results:")
    
    # Calculate batch statistics
    scores = [result['prediction']['score'] for result in batch_results]
    categories = [result['prediction']['category'] for result in batch_results]
    
    print(f"Average Score: {np.mean(scores):.1f}/100")
    print(f"Score Std: {np.std(scores):.1f}")
    print(f"Score Range: {np.min(scores):.1f} - {np.max(scores):.1f}")
    print(f"Category Distribution:")
    for category, count in pd.Series(categories).value_counts().items():
        print(f"  {category}: {count} ({count/len(categories):.1%})")
    
    # Component analysis summary
    component_summary = {}
    for component in ['wit_level', 'sarcasm_level', 'irony_level', 'surprise_level', 'absurdity_level']:
        component_values = [result['components'][component] for result in batch_results]
        component_summary[component] = {
            'mean': np.mean(component_values),
            'std': np.std(component_values),
            'min': np.min(component_values),
            'max': np.max(component_values)
        }
    
    print(f"\nComponent Analysis Summary:")
    for component, stats in component_summary.items():
        print(f"  {component}: {stats['mean']:.1f} (±{stats['std']:.1f})")
    
    # Save batch results
    batch_results_path = Path("evaluation_results/batch_analysis_results.json")
    import json
    with open(batch_results_path, 'w') as f:
        json.dump({
            'batch_results': batch_results,
            'statistics': {
                'average_score': np.mean(scores),
                'score_std': np.std(scores),
                'category_distribution': pd.Series(categories).value_counts().to_dict(),
                'component_summary': component_summary
            }
        }, f, indent=2)
    
    print(f"\n✅ Batch analysis completed. Results saved to: {batch_results_path}")
    
    return batch_results

def main():
    """Main function running all advanced analyses."""
    print("🎭 Chucklenet Predictor - Advanced Analysis")
    print("=" * 60)
    
    try:
        # Perform all advanced analyses
        print("\n🔬 Starting advanced analysis suite...")
        
        # 1. Advanced Text Analysis
        detailed_results = advanced_text_analysis()
        
        # 2. Comparative Model Analysis
        comparison_results = comparative_model_analysis()
        
        # 3. Humor Trend Analysis
        trend_analysis = humor_trend_analysis()
        
        # 4. Batch Analysis
        batch_results = batch_analysis()
        
        # Generate comprehensive report
        print("\n📋 Generating Comprehensive Report...")
        
        report = f"""
# Chucklenet Predictor - Advanced Analysis Report

## Summary
- Detailed text analysis: {len(detailed_results)} complex examples
- Comparative model analysis: {len(comparison_results)} samples
- Humor trend analysis: {len(trend_analysis)} humor styles
- Batch analysis: {len(batch_results)} samples

## Detailed Text Analysis Results
"""
        
        for result in detailed_results:
            report += f"""
### Example {result['example_num']}: {result['type']}
- Text: {result['text']}
- Expected: {result['expected_category']}
- Predicted: {result['predicted_score']:.1f} ({result['predicted_category']})
- Match: {'✅' if result['predicted_category'] == result['expected_category'] else '❌'}
- Confidence: {result['confidence']:.2f}
"""
        
        report += f"""
## Comparative Model Analysis
"""
        
        for result in comparison_results:
            report += f"""
### Sample {result['sample_num']}
- Base Prediction: {result['base_prediction']['score']:.1f} ({result['base_prediction']['category']})
- Component Analysis - Wit: {result['component_analysis']['wit_level']:.1f}, Sarcasm: {result['component_analysis']['sarcasm_level']:.1f}
- Reverse Target: {result['reverse_prediction']['desired_characteristic']} (Intensity: {result['reverse_prediction']['intensity']:.1f})
- Style Best Match: {result['style_comparison']['best_match_style']} ({result['style_comparison']['best_match_score']:.1f})
"""
        
        report += f"""
## Humor Trend Analysis
"""
        
        for style, analysis in trend_analysis.items():
            report += f"""
### {style.capitalize()} Style
- Average Score: {analysis['average_score']:.1f}/100
- Score Range: {analysis['score_range'][0]:.1f} - {analysis['score_range'][1]:.1f}
- Most Common Category: {max(analysis['category_distribution'], key=analysis['category_distribution'].get)}
"""
        
        # Save comprehensive report
        report_path = Path("evaluation_reports/advanced_analysis_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Advanced analysis completed successfully!")
        print(f"📄 Comprehensive report saved to: {report_path}")
        print("\n🎉 All analysis tasks completed!")
        
    except Exception as e:
        logger.error(f"Error in advanced analysis: {e}")
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("🎊 Chucklenet Predictor - Advanced Analysis Complete!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Advanced Analysis Failed")
        print("=" * 60)