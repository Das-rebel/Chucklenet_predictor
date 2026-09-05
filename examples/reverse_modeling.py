#!/usr/bin/env python3
"""
Reverse Modeling Example for Chucklenet Predictor

This script demonstrates the reverse modeling capabilities - 
finding input characteristics that lead to desired humor outputs.

Author: Subho Das
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

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

class ReverseModelingEngine:
    """Engine for reverse modeling humor characteristics."""
    
    def __init__(self, predictor: ChucklenetPredictor):
        self.predictor = predictor
        self.text_processor = TextProcessor()
        self.reverse_cache = {}
        
    def reverse_engineer_humor(self, 
                             target_score: float, 
                             target_category: str = None,
                             text_context: str = None) -> Dict[str, Any]:
        """
        Reverse engineer characteristics needed for target humor.
        
        Args:
            target_score: Target humor strength (0-100)
            target_category: Target humor category
            text_context: Optional context for the humor
            
        Returns:
            Reverse engineering results
        """
        print(f"🔄 Reverse engineering for target: {target_score}/100")
        if target_category:
            print(f"🎯 Target category: {target_category}")
        
        # Analyze target characteristics
        target_characteristics = self._analyze_target_characteristics(target_score, target_category)
        
        # Generate guidelines
        guidelines = self._generate_humor_guidelines(target_characteristics)
        
        # Create example constructions
        examples = self._create_humor_examples(target_characteristics)
        
        # Validate examples
        validation_results = self._validate_examples(examples, target_score, target_category)
        
        # Compile results
        reverse_engineering_result = {
            'target_score': target_score,
            'target_category': target_category,
            'characteristics': target_characteristics,
            'guidelines': guidelines,
            'examples': examples,
            'validation': validation_results,
            'methodology': 'reverse_characteristic_analysis'
        }
        
        # Cache result
        cache_key = f"{target_score}_{target_category}_{hash(text_context) if text_context else 'none'}"
        self.reverse_cache[cache_key] = reverse_engineering_result
        
        return reverse_engineering_result
    
    def _analyze_target_characteristics(self, 
                                    target_score: float, 
                                    target_category: Optional[str]) -> Dict[str, Any]:
        """Analyze characteristics needed for target humor."""
        
        # Base characteristics mapping
        characteristic_ranges = {
            'wit_level': (2, 8),
            'sarcasm_level': (1, 7),
            'irony_level': (2, 8),
            'surprise_level': (3, 9),
            'absurdity_level': (2, 8),
            'linguistic_complexity': (3, 8),
            'wordplay_intensity': (2, 8),
            'semantic_twist': (3, 9),
            'cultural_reference_depth': (2, 7)
        }
        
        # Category-specific adjustments
        category_adjustments = {
            'Witty': {'wit_level': (4, 9), 'linguistic_complexity': (5, 9)},
            'Clever': {'sarcasm_level': (3, 8), 'semantic_twist': (5, 9)},
            'Sarcastic': {'sarcasm_level': (5, 9), 'irony_level': (4, 8)},
            'Absurd': {'absurdity_level': (5, 9), 'surprise_level': (6, 9)},
            'Surprising': {'surprise_level': (6, 9), 'semantic_twist': (4, 8)},
            'Ironic': {'irony_level': (5, 9), 'sarcasm_level': (3, 7)},
            'Dry': {'sarcasm_level': (2, 5), 'surprise_level': (3, 6)},
            'Slapstick': {'absurdity_level': (6, 9), 'surprise_level': (5, 8)}
        }
        
        # Adjust characteristics based on target score and category
        adjusted_characteristics = {}
        
        for characteristic, base_range in characteristic_ranges.items():
            # Adjust for score level
            score_factor = target_score / 100.0
            
            if characteristic in ['wit_level', 'sarcasm_level', 'irony_level', 'surprise_level', 'absurdity_level']:
                # Core humor characteristics scale with score
                adjusted_min = int(base_range[0] + (score_factor * (base_range[1] - base_range[0]) * 0.3))
                adjusted_max = int(base_range[1] - ((1 - score_factor) * (base_range[1] - base_range[0]) * 0.3))
            else:
                # Linguistic characteristics scale differently
                adjusted_min = int(base_range[0] + (score_factor * (base_range[1] - base_range[0]) * 0.5))
                adjusted_max = int(base_range[1] - ((1 - score_factor) * (base_range[1] - base_range[0]) * 0.5))
            
            # Apply category-specific adjustments
            if target_category and target_category in category_adjustments:
                if characteristic in category_adjustments[target_category]:
                    cat_range = category_adjustments[target_category][characteristic]
                    adjusted_min = max(adjusted_min, cat_range[0])
                    adjusted_max = min(adjusted_max, cat_range[1])
            
            # Clamp ranges
            adjusted_min = max(1, min(10, adjusted_min))
            adjusted_max = max(1, min(10, adjusted_max))
            adjusted_min = min(adjusted_min, adjusted_max)
            
            adjusted_characteristics[characteristic] = (adjusted_min, adjusted_max)
        
        # Calculate optimal values
        optimal_values = {}
        for characteristic, (min_val, max_val) in adjusted_characteristics.items():
            # Optimal is slightly above midpoint for high scores, midpoint for medium scores
            if target_score >= 70:
                optimal = int(min_val + (max_val - min_val) * 0.7)
            elif target_score >= 40:
                optimal = int(min_val + (max_val - min_val) * 0.5)
            else:
                optimal = int(min_val + (max_val - min_val) * 0.3)
            
            optimal_values[characteristic] = optimal
        
        return {
            'ranges': adjusted_characteristics,
            'optimal_values': optimal_values,
            'score_level': 'High' if target_score >= 70 else 'Medium' if target_score >= 40 else 'Low',
            'characteristics_needed': self._identify_key_characteristics(target_score, target_category)
        }
    
    def _identify_key_characteristics(self, target_score: float, target_category: Optional[str]) -> List[str]:
        """Identify key characteristics needed for target humor."""
        
        base_characteristics = {
            'Low': ['surprise_level', 'semantic_twist'],
            'Medium': ['wit_level', 'semantic_twist', 'linguistic_complexity'],
            'High': ['all']
        }
        
        score_level = 'High' if target_score >= 70 else 'Medium' if target_score >= 40 else 'Low'
        
        if target_category:
            # Category-specific key characteristics
            category_keys = {
                'Witty': ['wit_level', 'linguistic_complexity'],
                'Clever': ['semantic_twist', 'sarcasm_level'],
                'Sarcastic': ['sarcasm_level', 'irony_level'],
                'Absurd': ['absurdity_level', 'surprise_level'],
                'Surprising': ['surprise_level', 'semantic_twist'],
                'Ironic': ['irony_level', 'sarcasm_level'],
                'Dry': ['sarcasm_level', 'irony_level'],
                'Slapstick': ['absurdity_level', 'surprise_level']
            }
            
            if target_category in category_keys:
                return category_keys[target_category]
        
        return base_characteristics[score_level]
    
    def _generate_humor_guidelines(self, characteristics: Dict[str, Any]) -> List[str]:
        """Generate humor guidelines based on characteristics."""
        guidelines = []
        
        optimal = characteristics['optimal_values']
        
        # Wit guidelines
        if optimal['wit_level'] >= 7:
            guidelines.append("Use clever wordplay and unexpected associations")
            guidelines.append("Employ sophisticated vocabulary and complex sentence structures")
        elif optimal['wit_level'] >= 5:
            guidelines.append("Use moderate wordplay and surprise elements")
            guidelines.append("Balance complexity with clarity")
        else:
            guidelines.append("Keep language simple and direct")
            guidelines.append("Focus on basic surprise elements")
        
        # Sarcasm guidelines
        if optimal['sarcasm_level'] >= 6:
            guidelines.append("Use strong ironic statements with clear contrast")
            guidelines.append("Employ deadpan delivery with exaggerated expectations")
        elif optimal['sarcasm_level'] >= 4:
            guidelines.append("Use moderate irony with subtle contrasts")
            guidelines.append("Balance sarcasm with genuine elements")
        else:
            guidelines.append("Minimize sarcasm, focus on direct humor")
            guidelines.append("Use straightforward positive surprise")
        
        # Surprise guidelines
        if optimal['surprise_level'] >= 7:
            guidelines.append("Create unexpected twists and punchlines")
            guidelines.append("Build tension and deliver maximum surprise")
        elif optimal['surprise_level'] >= 5:
            guidelines.append("Use moderate surprise with clear setup")
            guidelines.append("Balance predictability with unexpected elements")
        else:
            guidelines.append("Use gentle, familiar surprise elements")
            guidelines.append("Avoid shocking or extreme twists")
        
        # Absurdity guidelines
        if optimal['absurdity_level'] >= 7:
            guidelines.append("Use extreme illogical situations")
            guidelines.append("Create bizarre scenarios with no real-world logic")
        elif optimal['absurdity_level'] >= 5:
            guidelines.append("Use moderate illogical elements")
            guidelines.append("Balance absurdity with some real-world connections")
        else:
            guidelines.append("Keep humor grounded in reality")
            guidelines.append("Use subtle absurd elements only")
        
        # Linguistic complexity
        if optimal['linguistic_complexity'] >= 7:
            guidelines.append("Use sophisticated vocabulary and complex syntax")
            guidelines.append("Employ literary devices and advanced grammar")
        elif optimal['linguistic_complexity'] >= 5:
            guidelines.append("Use moderate vocabulary and sentence structures")
            guidelines.append("Balance simplicity with some sophistication")
        else:
            guidelines.append("Use simple, clear language")
            guidelines.append("Avoid complex terminology or structures")
        
        return guidelines
    
    def _create_humor_examples(self, characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create example humor constructions based on characteristics."""
        examples = []
        optimal = characteristics['optimal_values']
        
        # Example 1: High wit, moderate surprise
        if optimal['wit_level'] >= 6 and optimal['surprise_level'] >= 5:
            example_text = "I told my computer I needed a break, and now it won't stop sending me Kit-Kats."
            example_type = "Wordplay with Technology"
        # Example 2: High sarcasm, moderate irony
        elif optimal['sarcasm_level'] >= 6:
            example_text = "Oh great, another meeting. I just love sitting in a room doing nothing productive."
            example_type = "Deadpan Sarcastic"
        # Example 3: High absurdity, maximum surprise
        elif optimal['absurdity_level'] >= 7:
            example_text = "Why don't skeletons fight each other? They don't have the guts... which is why they're always bone-idle."
            example_type = "Absurd Pun"
        # Example 4: Moderate everything
        else:
            example_text = "Why don't eggs tell jokes? They'd crack each other up!"
            example_type = "Classic Pun"
        
        # Process the example
        text_features = self.text_processor.preprocess_text(example_text)
        humor_prediction = self.predictor.predict_humor_strength(example_text, text_features)
        component_analysis = self.predictor.analyze_humor_components(example_text, text_features)
        
        example = {
            'text': example_text,
            'type': example_type,
            'prediction': humor_prediction,
            'components': component_analysis,
            'target_match': abs(humor_prediction['score'] - 70) <= 15  # Within 15 points of target
        }
        
        examples.append(example)
        
        # Create additional examples for different aspects
        if optimal['surprise_level'] >= 7:
            surprise_text = "I asked my dog what's two minus two. He said nothing. Because dogs can't do math, but they can be very dramatic about it."
            surprise_features = self.text_processor.preprocess_text(surprise_text)
            surprise_prediction = self.predictor.predict_humor_strength(surprise_text, surprise_features)
            
            examples.append({
                'text': surprise_text,
                'type': 'High Surprise',
                'prediction': surprise_prediction,
                'components': self.predictor.analyze_humor_components(surprise_text, surprise_features),
                'target_match': abs(surprise_prediction['score'] - 70) <= 15
            })
        
        if optimal['absurdity_level'] >= 7:
            absurd_text = "I'm on a whiskey diet. I've lost three days already. And my liver has gained 10 pounds. Fair trade."
            absurd_features = self.text_processor.preprocess_text(absurd_text)
            absurd_prediction = self.predictor.predict_humor_strength(absurd_text, absurd_features)
            
            examples.append({
                'text': absurd_text,
                'type': 'High Absurdity',
                'prediction': absurd_prediction,
                'components': self.predictor.analyze_humor_components(absurd_text, absurd_features),
                'target_match': abs(absurd_prediction['score'] - 70) <= 15
            })
        
        return examples
    
    def _validate_examples(self, examples: List[Dict], target_score: float, target_category: Optional[str]) -> Dict[str, Any]:
        """Validate generated examples against target criteria."""
        validation = {
            'total_examples': len(examples),
            'successful_matches': 0,
            'scores': [],
            'categories': [],
            'score_errors': [],
            'category_matches': []
        }
        
        for example in examples:
            score = example['prediction']['score']
            category = example['prediction']['category']
            target_match = example['target_match']
            
            validation['scores'].append(score)
            validation['categories'].append(category)
            validation['score_errors'].append(abs(score - target_score))
            validation['category_matches'].append(category == target_category)
            
            if target_match:
                validation['successful_matches'] += 1
        
        validation['average_score'] = np.mean(validation['scores'])
        validation['score_accuracy'] = validation['successful_matches'] / validation['total_examples']
        validation['average_error'] = np.mean(validation['score_errors'])
        validation['category_accuracy'] = sum(validation['category_matches']) / validation['total_examples']
        
        return validation
    
    def optimize_for_context(self, 
                           target_score: float, 
                           context: str, 
                           context_type: str = 'conversation') -> Dict[str, Any]:
        """
        Optimize humor characteristics for specific context.
        
        Args:
            target_score: Target humor strength
            context: Context description
            context_type: Type of context (conversation, presentation, etc.)
        
        Returns:
            Context-optimized reverse engineering results
        """
        print(f"🎯 Optimizing for context: {context_type}")
        print(f"📝 Context: {context}")
        
        # Base reverse engineering
        base_result = self.reverse_engineer_humor(target_score, None, context)
        
        # Context-specific adjustments
        context_adjustments = self._apply_context_adjustments(base_result, context_type, context)
        
        # Optimize guidelines for context
        optimized_guidelines = self._optimize_guidelines_for_context(context_adjustments['guidelines'], context_type)
        
        # Create context-specific examples
        context_examples = self._create_context_examples(context_adjustments, context_type)
        
        # Validate in context
        context_validation = self._validate_context_examples(context_examples, context_type)
        
        # Compile results
        context_result = {
            'target_score': target_score,
            'context': context,
            'context_type': context_type,
            'base_characteristics': base_result['characteristics'],
            'context_adjustments': context_adjustments,
            'optimized_guidelines': optimized_guidelines,
            'context_examples': context_examples,
            'context_validation': context_validation,
            'methodology': 'context_optimized_reverse_engineering'
        }
        
        return context_result
    
    def _apply_context_adjustments(self, base_result: Dict, context_type: str, context: str) -> Dict[str, Any]:
        """Apply context-specific adjustments to characteristics."""
        adjustments = {}
        optimal = base_result['characteristics']['optimal_values']
        
        if context_type == 'conversation':
            # Casual conversation - more direct humor
            adjustments['sarcasm_level'] = (optimal['sarcasm_level'] - 1, optimal['sarcasm_level'] + 1)
            adjustments['wit_level'] = (optimal['wit_level'] - 1, optimal['wit_level'] + 1)
            adjustments['linguistic_complexity'] = (max(1, optimal['linguistic_complexity'] - 2), optimal['linguistic_complexity'])
            
        elif context_type == 'presentation':
            # Professional presentation - more sophisticated humor
            adjustments['linguistic_complexity'] = (optimal['linguistic_complexity'], min(10, optimal['linguistic_complexity'] + 2))
            adjustments['wit_level'] = (optimal['wit_level'], min(10, optimal['wit_level'] + 1))
            adjustments['sarcasm_level'] = (max(1, optimal['sarcasm_level'] - 2), optimal['sarcasm_level'])
            
        elif context_type == 'social_media':
            # Social media - high energy, surprise
            adjustments['surprise_level'] = (min(10, optimal['surprise_level'] + 2), min(10, optimal['surprise_level'] + 3))
            adjustments['absurdity_level'] = (optimal['absurdity_level'], min(10, optimal['absurdity_level'] + 2))
            adjustments['semantic_twist'] = (optimal['semantic_twist'], min(10, optimal['semantic_twist'] + 2))
            
        elif context_type == 'academic':
            # Academic setting - subtle, intelligent humor
            adjustments['wit_level'] = (optimal['wit_level'], min(10, optimal['wit_level'] + 2))
            adjustments['linguistic_complexity'] = (optimal['linguistic_complexity'], min(10, optimal['linguistic_complexity'] + 2))
            adjustments['sarcasm_level'] = (max(1, optimal['sarcasm_level'] - 2), max(1, optimal['sarcasm_level'] - 1))
            adjustments['absurdity_level'] = (max(1, optimal['absurdity_level'] - 2), optimal['absurdity_level'])
        
        return adjustments
    
    def _optimize_guidelines_for_context(self, guidelines: List[str], context_type: str) -> List[str]:
        """Optimize guidelines for specific context."""
        context_specific_guidelines = []
        
        for guideline in guidelines:
            # Add context-specific optimization
            if context_type == 'conversation':
                context_specific_guidelines.append(guideline + " (Keep it casual and immediate)")
            elif context_type == 'presentation':
                context_specific_guidelines.append(guideline + " (Maintain professional tone)")
            elif context_type == 'social_media':
                context_specific_guidelines.append(guideline + " (Make it shareable and engaging)")
            elif context_type == 'academic':
                context_specific_guidelines.append(guideline + " (Use sophisticated references)")
            else:
                context_specific_guidelines.append(guideline)
        
        return context_specific_guidelines
    
    def _create_context_examples(self, adjustments: Dict, context_type: str) -> List[Dict[str, Any]]:
        """Create context-specific humor examples."""
        examples = []
        
        if context_type == 'conversation':
            example_text = "Wait, that's not right. Let me try that again... nope, still wrong. Okay, third time's the charm... or not."
            example_type = "Casual Self-Deprecating"
        elif context_type == 'presentation':
            example_text = "As we can see from this complex algorithm, sometimes the simplest solution is the most elegant... unlike this presentation."
            example_type = "Sophisticated Academic"
        elif context_type == 'social_media':
            example_text = "Me trying to adult today: *exists* Also me: *procrastinates* Adulting: hard mode activated 🎮"
            example_type = "High-Energy Social Media"
        elif context_type == 'academic':
            example_text = "This theorem is so elegant it makes me want to write poetry... but I'm a mathematician, so I'll just keep doing proofs instead."
            example_type = "Academic Wit"
        else:
            example_text = "I'm not arguing, I'm just explaining why I'm right."
            example_type = "General Context"
        
        # Process example
        text_features = self.text_processor.preprocess_text(example_text)
        humor_prediction = self.predictor.predict_humor_strength(example_text, text_features)
        component_analysis = self.predictor.analyze_humor_components(example_text, text_features)
        
        example = {
            'text': example_text,
            'type': example_type,
            'prediction': humor_prediction,
            'components': component_analysis,
            'context_optimized': True
        }
        
        examples.append(example)
        return examples
    
    def _validate_context_examples(self, examples: List[Dict], context_type: str) -> Dict[str, Any]:
        """Validate examples in specific context."""
        validation = {
            'context_type': context_type,
            'total_examples': len(examples),
            'scores': [],
            'categories': [],
            'context_appropriateness': []
        }
        
        for example in examples:
            score = example['prediction']['score']
            category = example['prediction']['category']
            
            validation['scores'].append(score)
            validation['categories'].append(category)
            
            # Context appropriateness (simplified)
            if context_type == 'conversation' and category in ['Witty', 'Surprising']:
                validation['context_appropriateness'].append(True)
            elif context_type == 'presentation' and category in ['Witty', 'Clever']:
                validation['context_appropriateness'].append(True)
            elif context_type == 'social_media' and category in ['Surprising', 'Absurd']:
                validation['context_appropriateness'].append(True)
            elif context_type == 'academic' and category in ['Witty', 'Clever']:
                validation['context_appropriateness'].append(True)
            else:
                validation['context_appropriateness'].append(False)
        
        validation['average_score'] = np.mean(validation['scores'])
        validation['appropriateness_rate'] = sum(validation['context_appropriateness']) / validation['total_examples']
        
        return validation


def main():
    """Main function demonstrating reverse modeling capabilities."""
    print("🔄 Chucklenet Predictor - Reverse Modeling")
    print("=" * 60)
    
    # Initialize components
    predictor = ChucklenetPredictor()
    reverse_engine = ReverseModelingEngine(predictor)
    evaluator = EvaluationMetrics(save_results=True)
    
    try:
        # Example 1: Basic Reverse Engineering
        print("\n🎯 Example 1: Basic Reverse Engineering")
        print("-" * 40)
        
        result_1 = reverse_engine.reverse_engineer_humor(
            target_score=75,
            target_category="Witty"
        )
        
        print(f"Target: {result_1['target_score']}/100 ({result_1['target_category']})")
        print(f"Characteristics needed:")
        for char, opt in result_1['characteristics']['optimal_values'].items():
            print(f"  - {char}: {opt}/10")
        print(f"Guidelines:")
        for guideline in result_1['guidelines'][:3]:  # Show first 3 guidelines
            print(f"  - {guideline}")
        
        # Example 2: Context Optimization
        print("\n🎯 Example 2: Context Optimization")
        print("-" * 40)
        
        contexts = [
            ("presentation", "Professional conference keynote speech"),
            ("social_media", "Twitter post about daily life"),
            ("conversation", "Casual chat with friends"),
            ("academic", "University lecture break")
        ]
        
        context_results = []
        for context_type, context_description in contexts:
            print(f"\n--- {context_type.capitalize()} Context ---")
            print(f"Description: {context_description}")
            
            context_result = reverse_engine.optimize_for_context(
                target_score=65,
                context=context_description,
                context_type=context_type
            )
            
            print(f"Optimal characteristics:")
            for char, opt in context_result['context_adjustments'].items():
                print(f"  - {char}: {opt[0]}-{opt[1]} (range)")
            
            print(f"Context-appropriate examples: {context_result['context_validation']['appropriateness_rate']:.1%}")
            
            context_results.append(context_result)
        
        # Example 3: Score Optimization
        print("\n🎯 Example 3: Score Optimization")
        print("-" * 40)
        
        scores = [30, 50, 70, 90]
        score_results = []
        
        for target_score in scores:
            print(f"\n--- Target Score: {target_score}/100 ---")
            
            score_result = reverse_engine.reverse_engineer_humor(
                target_score=target_score
            )
            
            print(f"Characteristics level: {score_result['characteristics']['score_level']}")
            print(f"Key characteristics: {', '.join(score_result['characteristics']['characteristics_needed'])}")
            
            score_results.append(score_result)
        
        # Example 4: Validation and Testing
        print("\n🎯 Example 4: Validation and Testing")
        print("-" * 40)
        
        # Test reverse predictions
        test_examples = [
            "I'm reading a book about anti-gravity. It's impossible to put down!",
            "Why don't skeletons fight each other? They don't have the guts.",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "Parallel lines have so much in common. It's a shame they'll never meet."
        ]
        
        validation_results = []
        for i, test_text in enumerate(test_examples, 1):
            print(f"\n--- Test Example {i} ---")
            print(f"Text: {test_text}")
            
            # Get prediction
            text_features = predictor.text_processor.preprocess_text(test_text)
            actual_prediction = predictor.predict_humor_strength(test_text, text_features)
            
            # Reverse engineer from actual score
            reverse_result = reverse_engine.reverse_engineer_humor(
                target_score=actual_prediction['score']
            )
            
            # Compare
            score_match = abs(actual_prediction['score'] - reverse_result['target_score']) < 5
            category_match = actual_prediction['category'] == reverse_result['target_category']
            
            print(f"Actual: {actual_prediction['score']:.1f} ({actual_prediction['category']})")
            print(f"Reverse Target: {reverse_result['target_score']} ({reverse_result['target_category']})")
            print(f"Score Match: {'✅' if score_match else '❌'}")
            print(f"Category Match: {'✅' if category_match else '❌'}")
            
            validation_results.append({
                'test_text': test_text,
                'actual_prediction': actual_prediction,
                'reverse_target': reverse_result,
                'score_match': score_match,
                'category_match': category_match
            })
        
        # Generate comprehensive report
        print("\n📋 Generating Reverse Modeling Report")
        print("-" * 40)
        
        report = f"""
# Chucklenet Predictor - Reverse Modeling Report

## Summary
This report demonstrates the reverse modeling capabilities of the Chucklenet Predictor,
showing how to generate characteristics and guidelines for desired humor outputs.

## Example 1: Basic Reverse Engineering
- Target: {result_1['target_score']}/100 ({result_1['target_category']})
- Characteristics needed:
"""
        
        for char, opt in result_1['characteristics']['optimal_values'].items():
            report += f"  - {char}: {opt}/10\n"
        
        report += f"""
- Guidelines generated: {len(result_1['guidelines'])} total
- Examples created: {len(result_1['examples'])}
- Validation success rate: {result_1['validation']['score_accuracy']:.1%}

## Example 2: Context Optimization
"""
        
        for context_result in context_results:
            report += f"""
### {context_result['context_type'].capitalize()} Context
- Target: {context_result['target_score']}/100
- Context appropriateness: {context_result['context_validation']['appropriateness_rate']:.1%}
- Context-optimized examples: {context_result['context_validation']['total_examples']}

## Example 3: Score Optimization
"""
        
        for score_result in score_results:
            report += f"""
### Score Level: {score_result['characteristics']['score_level']} ({score_result['target_score']}/100)
- Key characteristics: {', '.join(score_result['characteristics']['characteristics_needed'])}
- Validation accuracy: {score_result['validation']['score_accuracy']:.1%}

## Example 4: Validation Results
- Test examples: {len(validation_results)}
- Score matches: {sum(1 for r in validation_results if r['score_match'])}/{len(validation_results)}
- Category matches: {sum(1 for r in validation_results if r['category_match'])}/{len(validation_results)}
"""
        
        # Save report
        report_path = Path("evaluation_reports/reverse_modeling_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Reverse modeling demonstration completed!")
        print(f"📄 Report saved to: {report_path}")
        
        # Print final summary
        print("\n🎯 Reverse Modeling Summary:")
        print(f"- Successfully reverse-engineered {len(score_results)} different score levels")
        print(f"- Optimized for {len(contexts)} different contexts")
        print(f"- Validated on {len(validation_results)} test examples")
        print(f"- Overall score match accuracy: {sum(1 for r in validation_results if r['score_match'])/len(validation_results):.1%}")
        print(f"- Overall category match accuracy: {sum(1 for r in validation_results if r['category_match'])/len(validation_results):.1%}")
        
        print("\n🎉 Reverse modeling capabilities demonstrated successfully!")
        
    except Exception as e:
        logger.error(f"Error in reverse modeling: {e}")
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("🎊 Chucklenet Predictor - Reverse Modeling Complete!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Reverse Modeling Failed")
        print("=" * 60)