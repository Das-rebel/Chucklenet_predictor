"""
Evaluation Metrics for Chucklenet Predictor

This module provides comprehensive evaluation metrics and benchmarking capabilities
for humor prediction models including statistical analysis, human evaluation, and 
performance comparison.

Author: Subho Das
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, cohen_kappa_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
import logging
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for humor prediction models.
    """
    
    def __init__(self, save_results: bool = True, output_dir: str = "evaluation_results"):
        """
        Initialize evaluation metrics.
        
        Args:
            save_results: Whether to save evaluation results
            output_dir: Directory to save results
        """
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Evaluation history
        self.evaluation_history = []
        
        # Define humor strength bins for classification
        self.humor_bins = [
            (0, 20, "Very Low"),
            (20, 40, "Low"),
            (40, 60, "Moderate"),
            (60, 80, "High"),
            (80, 100, "Very High")
        ]
        
        logger.info("EvaluationMetrics initialized")
    
    def evaluate_humor_classification(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray, 
                                     y_pred_proba: Optional[np.ndarray] = None,
                                     sample_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate humor prediction performance.
        
        Args:
            y_true: True humor strength values
            y_pred: Predicted humor strength values
            y_pred_proba: Predicted probabilities (for probabilistic models)
            sample_weights: Sample weights for weighted metrics
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            # Ensure numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Validate inputs
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have the same length")
            
            # Basic metrics
            mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weights)
            mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weights)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred, sample_weight=sample_weights)
            
            # Classification metrics (binned humor strength)
            y_true_binned = self._bin_humor_strength(y_true)
            y_pred_binned = self._bin_humor_strength(y_pred)
            
            accuracy = accuracy_score(y_true_binned, y_pred_binned, sample_weight=sample_weights)
            
            # Handle multi-class metrics
            unique_classes = np.unique(y_true_binned)
            if len(unique_classes) > 1:
                # Calculate weighted metrics
                f1_weighted = f1_score(y_true_binned, y_pred_binned, 
                                      average='weighted', sample_weight=sample_weights)
                f1_macro = f1_score(y_true_binned, y_pred_binned, 
                                   average='macro', sample_weight=sample_weights)
                f1_micro = f1_score(y_true_binned, y_pred_binned, 
                                   average='micro', sample_weight=sample_weights)
                
                precision_weighted = precision_score(y_true_binned, y_pred_binned, 
                                                  average='weighted', sample_weight=sample_weights)
                precision_macro = precision_score(y_true_binned, y_pred_binned, 
                                               average='macro', sample_weight=sample_weights)
                
                recall_weighted = recall_score(y_true_binned, y_pred_binned, 
                                            average='weighted', sample_weight=sample_weights)
                recall_macro = recall_score(y_true_binned, y_pred_binned, 
                                         average='macro', sample_weight=sample_weights)
                
                kappa = cohen_kappa_score(y_true_binned, y_pred_binned, weights='quadratic')
            else:
                # Single class case
                f1_weighted = f1_macro = f1_micro = 0
                precision_weighted = precision_macro = 0
                recall_weighted = recall_macro = 0
                kappa = 0
            
            # Confusion matrix - use only classes present in data
            present_labels = np.unique(np.concatenate([y_true_binned, y_pred_binned]))
            present_names = [self.humor_bins[label][2] for label in present_labels]
            cm = confusion_matrix(y_true_binned, y_pred_binned, labels=present_labels)
            
            # Classification report
            class_report = classification_report(y_true_binned, y_pred_binned, 
                                               labels=present_labels,
                                               target_names=present_names,
                                               output_dict=True, zero_division=0)
            
            # Error analysis
            errors = self._analyze_errors(y_true, y_pred)
            
            # Statistical tests
            correlation_metrics = self._calculate_correlations(y_true, y_pred)
            
            # Probabilistic metrics (if probabilities are provided)
            prob_metrics = {}
            if y_pred_proba is not None and y_pred_proba.shape[1] > 1:
                try:
                    # Convert to binary for AUC calculation (e.g., humorous vs non-humorous)
                    threshold = np.median(y_true)
                    y_true_binary = (y_true >= threshold).astype(int)
                    
                    # Check if we can compute AUC
                    if len(np.unique(y_true_binary)) > 1:
                        auc_roc = roc_auc_score(y_true_binary, y_pred_proba[:, 1])
                        prob_metrics['auc_roc'] = auc_roc
                        
                        # Precision-recall curve
                        ap_score = average_precision_score(y_true_binary, y_pred_proba[:, 1])
                        prob_metrics['average_precision'] = ap_score
                except Exception as e:
                    logger.warning(f"Error computing probabilistic metrics: {e}")
            
            # Compile results
            results = {
                'regression_metrics': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                },
                'classification_metrics': {
                    'accuracy': accuracy,
                    'f1_weighted': f1_weighted,
                    'f1_macro': f1_macro,
                    'f1_micro': f1_micro,
                    'precision_weighted': precision_weighted,
                    'precision_macro': precision_macro,
                    'recall_weighted': recall_weighted,
                    'recall_macro': recall_macro,
                    'kappa': kappa
                },
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'error_analysis': errors,
                'correlation_metrics': correlation_metrics,
                'probabilistic_metrics': prob_metrics,
                'num_samples': len(y_true),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add to history
            self.evaluation_history.append(results)
            
            # Save results if requested
            if self.save_results:
                self._save_evaluation_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in humor classification evaluation: {e}")
            return {'error': str(e)}
    
    def _bin_humor_strength(self, humor_strengths: np.ndarray) -> np.ndarray:
        """
        Convert continuous humor strength to discrete bins.
        
        Args:
            humor_strengths: Continuous humor strength values
            
        Returns:
            Binned humor strength values
        """
        binned = np.zeros(len(humor_strengths), dtype=int)
        
        for i, strength in enumerate(humor_strengths):
            for j, (lower, upper, _) in enumerate(self.humor_bins):
                if lower <= strength < upper:
                    binned[i] = j
                    break
            else:
                # Handle edge case
                binned[i] = len(self.humor_bins) - 1
        
        return binned
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze prediction errors.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Error analysis results
        """
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        # Error statistics
        error_stats = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'mean_absolute_error': np.mean(abs_errors),
            'median_absolute_error': np.median(abs_errors),
            'max_absolute_error': np.max(abs_errors),
            'min_absolute_error': np.min(abs_errors),
            'quartile_error': {
                'q1': np.percentile(abs_errors, 25),
                'q2': np.percentile(abs_errors, 50),
                'q3': np.percentile(abs_errors, 75),
                'q95': np.percentile(abs_errors, 95),
                'q99': np.percentile(abs_errors, 99)
            }
        }
        
        # Error distribution
        error_distribution = {
            'negative_errors': np.sum(errors < 0) / len(errors),
            'positive_errors': np.sum(errors > 0) / len(errors),
            'zero_errors': np.sum(errors == 0) / len(errors),
            'large_errors': np.sum(np.abs(errors) > 20) / len(errors)
        }
        
        # Error by true value range
        error_by_range = {}
        for lower, upper, label in self.humor_bins:
            mask = (y_true >= lower) & (y_true < upper)
            if np.sum(mask) > 0:
                error_by_range[label] = {
                    'mean_absolute_error': np.mean(abs_errors[mask]),
                    'sample_count': np.sum(mask),
                    'error_std': np.std(errors[mask])
                }
        
        return {
            'error_statistics': error_stats,
            'error_distribution': error_distribution,
            'error_by_range': error_by_range
        }
    
    def _calculate_correlations(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate correlation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Correlation metrics
        """
        correlations = {}
        
        try:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            correlations['pearson'] = {'correlation': pearson_corr, 'p_value': pearson_p}
            
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            correlations['spearman'] = {'correlation': spearman_corr, 'p_value': spearman_p}
            
            # Kendall's tau
            kendall_corr, kendall_p = kendalltau(y_true, y_pred)
            correlations['kendall'] = {'correlation': kendall_corr, 'p_value': kendall_p}
            
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
            correlations = {'error': str(e)}
        
        return correlations
    
    def evaluate_human_agreement(self, 
                               ai_scores: np.ndarray, 
                               human_scores: np.ndarray,
                               human_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate agreement between AI predictions and human ratings.
        
        Args:
            ai_scores: AI predicted humor scores
            human_scores: Human humor ratings
            human_weights: Weights for different human raters
            
        Returns:
            Human agreement evaluation results
        """
        try:
            # Convert to numpy arrays
            ai_scores = np.array(ai_scores)
            human_scores = np.array(human_scores)
            
            if len(ai_scores) != len(human_scores):
                raise ValueError("AI scores and human scores must have the same length")
            
            # Calculate agreement metrics
            agreement_metrics = self._calculate_agreement_metrics(ai_scores, human_scores, human_weights)
            
            # Agreement by humor strength range
            agreement_by_range = {}
            for lower, upper, label in self.humor_bins:
                mask = (ai_scores >= lower) & (ai_scores < upper)
                if np.sum(mask) > 0:
                    range_ai = ai_scores[mask]
                    range_human = human_scores[mask]
                    if human_weights is not None:
                        range_weights = human_weights[mask]
                    else:
                        range_weights = None
                    
                    range_agreement = self._calculate_agreement_metrics(range_ai, range_human, range_weights)
                    agreement_by_range[label] = range_agreement
            
            # Discrepancy analysis
            discrepancies = np.abs(ai_scores - human_scores)
            discrepancy_stats = {
                'mean_discrepancy': np.mean(discrepancies),
                'median_discrepancy': np.median(discrepancies),
                'std_discrepancy': np.std(discrepancies),
                'max_discrepancy': np.max(discrepancies),
                'discrepancy_distribution': {
                    'small': np.sum(discrepancies <= 5) / len(discrepancies),
                    'medium': np.sum((discrepancies > 5) & (discrepancies <= 15)) / len(discrepancies),
                    'large': np.sum(discrepancies > 15) / len(discrepancies)
                }
            }
            
            return {
                'agreement_metrics': agreement_metrics,
                'agreement_by_range': agreement_by_range,
                'discrepancy_analysis': discrepancy_stats,
                'num_samples': len(ai_scores),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in human agreement evaluation: {e}")
            return {'error': str(e)}
    
    def _calculate_agreement_metrics(self, 
                                    ai_scores: np.ndarray, 
                                    human_scores: np.ndarray,
                                    weights: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate agreement metrics between AI and human scores.
        """
        try:
            # Inter-rater reliability (treating AI as one rater, humans as others)
            if len(human_scores.shape) == 1:
                # Single human rater
                human_scores = human_scores.reshape(-1, 1)
            
            # Add AI scores as an additional rater
            combined_scores = np.column_stack([human_scores, ai_scores.reshape(-1, 1)])
            
            # Intraclass correlation coefficient (ICC)
            icc = self._calculate_icc(combined_scores)
            
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(ai_scores, human_scores.flatten())
            
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(ai_scores, human_scores.flatten())
            
            # Mean absolute difference
            mae = np.mean(np.abs(ai_scores - human_scores.flatten()))
            
            # Root mean square difference
            rmse = np.sqrt(np.mean((ai_scores - human_scores.flatten()) ** 2))
            
            return {
                'icc': icc,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'mean_absolute_difference': mae,
                'root_mean_square_difference': rmse
            }
            
        except Exception as e:
            logger.warning(f"Error calculating agreement metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_icc(self, ratings: np.ndarray) -> float:
        """
        Calculate Intraclass Correlation Coefficient (ICC).
        
        Args:
            ratings: Array of ratings (samples x raters)
            
        Returns:
            ICC value
        """
        try:
            # Two-way random effects model for absolute agreement
            n_samples, n_raters = ratings.shape
            
            # Calculate means
            mean_rating = np.mean(ratings)
            mean_sample = np.mean(ratings, axis=1)
            mean_rater = np.mean(ratings, axis=0)
            
            # Sum of squares
            ss_total = np.sum((ratings - mean_rating) ** 2)
            ss_samples = n_raters * np.sum((mean_sample - mean_rating) ** 2)
            ss_raters = n_samples * np.sum((mean_rater - mean_rating) ** 2)
            ss_error = ss_total - ss_samples - ss_raters
            
            # Mean squares
            ms_samples = ss_samples / (n_samples - 1)
            ms_raters = ss_raters / (n_raters - 1)
            ms_error = ss_error / ((n_samples - 1) * (n_raters - 1))
            
            # ICC (two-way random effects, absolute agreement)
            icc = (ms_samples - ms_error) / (ms_samples + (n_raters - 1) * ms_error)
            
            return icc
            
        except Exception as e:
            logger.warning(f"Error calculating ICC: {e}")
            return 0.0
    
    def evaluate_baselines(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate against baseline models.
        
        Args:
            y_true: True humor strength values
            y_pred: Predicted humor strength values
            
        Returns:
            Baseline comparison results
        """
        try:
            baseline_results = {}
            
            # Mean baseline
            mean_baseline = np.full_like(y_true, np.mean(y_true))
            mean_mae = mean_absolute_error(y_true, mean_baseline)
            mean_mse = mean_squared_error(y_true, mean_baseline)
            
            # Median baseline
            median_baseline = np.full_like(y_true, np.median(y_true))
            median_mae = mean_absolute_error(y_true, median_baseline)
            median_mse = mean_squared_error(y_true, median_baseline)
            
            # Random baseline (normally distributed around mean)
            random_baseline = np.random.normal(np.mean(y_true), np.std(y_true), len(y_true))
            random_baseline = np.clip(random_baseline, 0, 100)
            random_mae = mean_absolute_error(y_true, random_baseline)
            random_mse = mean_squared_error(y_true, random_baseline)
            
            # Linear regression baseline
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(y_true)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X, y_true)
            lr_pred = lr.predict(X)
            lr_mae = mean_absolute_error(y_true, lr_pred)
            lr_mse = mean_squared_error(y_true, lr_pred)
            
            # Calculate model performance relative to baselines
            model_mae = mean_absolute_error(y_true, y_pred)
            model_mse = mean_squared_error(y_true, y_pred)
            
            # Performance improvements
            improvements = {
                'vs_mean_mae': ((mean_mae - model_mae) / mean_mae) * 100,
                'vs_median_mae': ((median_mae - model_mae) / median_mae) * 100,
                'vs_random_mae': ((random_mae - model_mae) / random_mae) * 100,
                'vs_lr_mae': ((lr_mae - model_mae) / lr_mae) * 100,
                'vs_mean_mse': ((mean_mse - model_mse) / mean_mse) * 100,
                'vs_median_mse': ((median_mse - model_mse) / median_mse) * 100,
                'vs_random_mse': ((random_mse - model_mse) / random_mse) * 100,
                'vs_lr_mse': ((lr_mse - model_mse) / lr_mse) * 100
            }
            
            baseline_results = {
                'baseline_performance': {
                    'mean_baseline': {'mae': mean_mae, 'mse': mean_mse},
                    'median_baseline': {'mae': median_mae, 'mse': median_mse},
                    'random_baseline': {'mae': random_mae, 'mse': random_mse},
                    'linear_regression_baseline': {'mae': lr_mae, 'mse': lr_mse}
                },
                'model_performance': {
                    'mae': model_mae,
                    'mse': model_mse
                },
                'improvements': improvements,
                'num_samples': len(y_true),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            return baseline_results
            
        except Exception as e:
            logger.error(f"Error in baseline evaluation: {e}")
            return {'error': str(e)}
    
    def generate_evaluation_report(self, 
                                 results: Dict, 
                                 model_name: str = "Chucklenet Predictor",
                                 include_plots: bool = True) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            model_name: Name of the model
            include_plots: Whether to include plots in report
            
        Returns:
            Formatted evaluation report
        """
        try:
            report = f"""
# {model_name} - Evaluation Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Performance Summary

### Regression Metrics
- **Mean Absolute Error (MAE)**: {results.get('regression_metrics', {}).get('mae', 'N/A'):.4f}
- **Mean Squared Error (MSE)**: {results.get('regression_metrics', {}).get('mse', 'N/A'):.4f}
- **Root Mean Squared Error (RMSE)**: {results.get('regression_metrics', {}).get('rmse', 'N/A'):.4f}
- **R-squared (R²)**: {results.get('regression_metrics', {}).get('r2', 'N/A'):.4f}

### Classification Metrics
- **Accuracy**: {results.get('classification_metrics', {}).get('accuracy', 'N/A'):.4f}
- **F1 Score (Weighted)**: {results.get('classification_metrics', {}).get('f1_weighted', 'N/A'):.4f}
- **F1 Score (Macro)**: {results.get('classification_metrics', {}).get('f1_macro', 'N/A'):.4f}
- **Precision (Weighted)**: {results.get('classification_metrics', {}).get('precision_weighted', 'N/A'):.4f}
- **Recall (Weighted)**: {results.get('classification_metrics', {}).get('recall_weighted', 'N/A'):.4f}
- **Cohen's Kappa**: {results.get('classification_metrics', {}).get('kappa', 'N/A'):.4f}

### Sample Information
- **Total Samples**: {results.get('num_samples', 'N/A')}
- **Evaluation Date**: {results.get('evaluation_timestamp', 'N/A')}

## 🔍 Error Analysis

### Error Distribution
"""
            
            if 'error_analysis' in results:
                error_dist = results['error_analysis'].get('error_distribution', {})
                report += f"""
- **Negative Errors**: {error_dist.get('negative_errors', 0):.2%}
- **Positive Errors**: {error_dist.get('positive_errors', 0):.2%}
- **Zero Errors**: {error_dist.get('zero_errors', 0):.2%}
- **Large Errors (>20)**: {error_dist.get('large_errors', 0):.2%}
"""
            
            report += f"""
### Error Statistics
"""
            
            if 'error_analysis' in results:
                error_stats = results['error_analysis'].get('error_statistics', {})
                report += f"""
- **Mean Error**: {error_stats.get('mean_error', 0):.4f}
- **Median Error**: {error_stats.get('median_error', 0):.4f}
- **Std Error**: {error_stats.get('std_error', 0):.4f}
- **Mean Absolute Error**: {error_stats.get('mean_absolute_error', 0):.4f}
- **Median Absolute Error**: {error_stats.get('median_absolute_error', 0):.4f}
- **Max Absolute Error**: {error_stats.get('max_absolute_error', 0):.4f}
"""
            
            report += f"""
## 📈 Correlation Analysis

### Statistical Correlations
"""
            
            if 'correlation_metrics' in results:
                correlations = results['correlation_metrics']
                for corr_type, corr_data in correlations.items():
                    if isinstance(corr_data, dict) and 'correlation' in corr_data:
                        report += f"""
- **{corr_type.title()}**: {corr_data['correlation']:.4f} (p-value: {corr_data['p_value']:.4f})
"""
            
            report += f"""
## 🎯 Classification Report

### Performance by Humor Strength Category
"""
            
            if 'classification_report' in results:
                class_report = results['classification_report']
                for category in ['Very Low', 'Low', 'Moderate', 'High', 'Very High']:
                    if category in class_report:
                        metrics = class_report[category]
                        report += f"""
#### {category} Humor
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **F1-Score**: {metrics.get('f1-score', 0):.4f}
- **Support**: {metrics.get('support', 0)}
"""
            
            report += f"""
## 📋 Additional Metrics

### Probabilistic Metrics
"""
            
            if 'probabilistic_metrics' in results:
                prob_metrics = results['probabilistic_metrics']
                for metric, value in prob_metrics.items():
                    report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
            
            if 'baseline_performance' in results:
                report += f"""
### Baseline Comparisons
"""
                baseline_metrics = results['baseline_performance']
                model_metrics = results.get('model_performance', {})
                
                if 'mae' in model_metrics:
                    report += f"""
- **Model MAE**: {model_metrics['mae']:.4f}
- **Mean Baseline MAE**: {baseline_metrics['mean_baseline']['mae']:.4f}
- **Improvement**: {results.get('improvements', {}).get('vs_mean_mae', 0):.2f}%
"""
            
            report += f"""
## 📊 Overall Assessment

### Performance Rating
"""
            
            # Generate performance rating
            accuracy = results.get('classification_metrics', {}).get('accuracy', 0)
            mae = results.get('regression_metrics', {}).get('mae', 100)
            
            if accuracy >= 0.8 and mae <= 5.0:
                rating = "Excellent"
                emoji = "🌟"
            elif accuracy >= 0.7 and mae <= 10.0:
                rating = "Good"
                emoji = "✅"
            elif accuracy >= 0.6 and mae <= 15.0:
                rating = "Fair"
                emoji = "⚠️"
            else:
                rating = "Needs Improvement"
                emoji = "❌"
            
            report += f"""
{emoji} **{rating} Performance**
- **Accuracy**: {accuracy:.2%}
- **MAE**: {mae:.2f}

### Recommendations
"""
            
            # Generate recommendations
            recommendations = []
            
            if accuracy < 0.7:
                recommendations.append("- Consider improving model architecture or training data quality")
            
            if mae > 10:
                recommendations.append("- High error rate suggests need for better feature engineering or regularization")
            
            if results.get('error_analysis', {}).get('error_distribution', {}).get('large_errors', 0) > 0.1:
                recommendations.append("- High percentage of large errors indicates poor generalization on difficult cases")
            
            if not recommendations:
                recommendations.append("- Model shows good performance across metrics")
                recommendations.append("- Consider deployment for production use")
            
            for rec in recommendations:
                report += f"{rec}\n"
            
            report += f"""
---
*Report generated by Chucklenet Predictor Evaluation Metrics*
*Total evaluations in session: {len(self.evaluation_history)}*
"""
            
            # Save report if requested
            if self.save_results:
                report_path = self.output_dir / f"{model_name.lower().replace(' ', '_')}_report.md"
                with open(report_path, 'w') as f:
                    f.write(report)
                logger.info(f"Evaluation report saved to {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _save_evaluation_results(self, results: Dict):
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results to save
        """
        try:
            # Save as JSON
            json_path = self.output_dir / "evaluation_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary statistics
            summary_stats = {
                'total_evaluations': len(self.evaluation_history),
                'latest_evaluation': results.get('evaluation_timestamp'),
                'average_mae': np.mean([res.get('regression_metrics', {}).get('mae', 0) 
                                      for res in self.evaluation_history]),
                'average_accuracy': np.mean([res.get('classification_metrics', {}).get('accuracy', 0) 
                                           for res in self.evaluation_history]),
                'evaluation_dates': [res.get('evaluation_timestamp') for res in self.evaluation_history]
            }
            
            summary_path = self.output_dir / "evaluation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {json_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def create_evaluation_plots(self, results: Dict, save_path: Optional[str] = None):
        """
        Create evaluation plots (placeholder for actual implementation).
        
        Args:
            results: Evaluation results
            save_path: Path to save plots
        """
        try:
            # This would be implemented with matplotlib
            # Create plots for:
            # - Confusion matrix
            # - Error distribution
            # - Correlation scatter plot
            # - ROC curve (if probabilistic)
            # etc.
            
            logger.info("Evaluation plots not implemented in this version")
            
        except Exception as e:
            logger.error(f"Error creating evaluation plots: {e}")
    
    def cross_validate(self, 
                      X: np.ndarray, 
                      y: np.ndarray, 
                      model, 
                      cv_folds: int = 5,
                      scoring: str = 'accuracy') -> Dict:
        """
        Perform cross-validation evaluation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Model to evaluate
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            # Convert humor strength to bins for stratification
            y_binned = self._bin_humor_strength(y)
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y_binned, cv=cv, scoring=scoring)
            
            # Calculate statistics
            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'min_score': np.min(cv_scores),
                'max_score': np.max(cv_scores),
                'median_score': np.median(cv_scores),
                'cv_folds': cv_folds,
                'scoring': scoring,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add to history
            self.evaluation_history.append(cv_results)
            
            # Save results
            if self.save_results:
                cv_path = self.output_dir / "cross_validation_results.json"
                with open(cv_path, 'w') as f:
                    json.dump(cv_results, f, indent=2, default=str)
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}
    
    def compare_models(self, 
                      models: Dict[str, object],
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      model_names: Optional[List[str]] = None) -> Dict:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test: Test targets
            model_names: Optional custom model names
            
        Returns:
            Model comparison results
        """
        try:
            comparison_results = {}
            
            # Evaluate each model
            for i, (model_key, model) in enumerate(models.items()):
                model_name = model_names[i] if model_names else model_key
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = getattr(model, 'predict_proba', None)
                if y_pred_proba:
                    y_pred_proba = y_pred_proba(X_test)
                
                # Evaluate
                eval_result = self.evaluate_humor_classification(y_test, y_pred, y_pred_proba)
                comparison_results[model_name] = eval_result
            
            # Compare performance
            performance_comparison = {}
            
            for metric in ['accuracy', 'mae', 'r2']:
                model_scores = {}
                for model_name, result in comparison_results.items():
                    if metric in result.get('classification_metrics', {}):
                        score = result['classification_metrics'][metric]
                    elif metric in result.get('regression_metrics', {}):
                        score = result['regression_metrics'][metric]
                    else:
                        score = None
                    
                    if score is not None:
                        model_scores[model_name] = score
                
                if model_scores:
                    best_model = max(model_scores, key=model_scores.get)
                    performance_comparison[metric] = {
                        'scores': model_scores,
                        'best_model': best_model,
                        'best_score': model_scores[best_model]
                    }
            
            # Compile final results
            final_results = {
                'model_evaluations': comparison_results,
                'performance_comparison': performance_comparison,
                'num_models': len(models),
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save comparison
            if self.save_results:
                comparison_path = self.output_dir / "model_comparison.json"
                with open(comparison_path, 'w') as f:
                    json.dump(final_results, f, indent=2, default=str)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            return {'error': str(e)}
    
    def get_evaluation_summary(self) -> Dict:
        """
        Get summary of all evaluations performed.
        
        Returns:
            Evaluation summary
        """
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        try:
            # Extract key metrics from all evaluations
            mae_scores = []
            accuracy_scores = []
            f1_scores = []
            
            for evaluation in self.evaluation_history:
                if 'regression_metrics' in evaluation:
                    mae_scores.append(evaluation['regression_metrics'].get('mae', 0))
                
                if 'classification_metrics' in evaluation:
                    accuracy_scores.append(evaluation['classification_metrics'].get('accuracy', 0))
                    f1_scores.append(evaluation['classification_metrics'].get('f1_weighted', 0))
            
            # Calculate summary statistics
            summary = {
                'total_evaluations': len(self.evaluation_history),
                'performance_metrics': {
                    'mae': {
                        'mean': np.mean(mae_scores) if mae_scores else 0,
                        'std': np.std(mae_scores) if mae_scores else 0,
                        'min': np.min(mae_scores) if mae_scores else 0,
                        'max': np.max(mae_scores) if mae_scores else 0
                    },
                    'accuracy': {
                        'mean': np.mean(accuracy_scores) if accuracy_scores else 0,
                        'std': np.std(accuracy_scores) if accuracy_scores else 0,
                        'min': np.min(accuracy_scores) if accuracy_scores else 0,
                        'max': np.max(accuracy_scores) if accuracy_scores else 0
                    },
                    'f1': {
                        'mean': np.mean(f1_scores) if f1_scores else 0,
                        'std': np.std(f1_scores) if f1_scores else 0,
                        'min': np.min(f1_scores) if f1_scores else 0,
                        'max': np.max(f1_scores) if f1_scores else 0
                    }
                },
                'latest_evaluation': self.evaluation_history[-1].get('evaluation_timestamp'),
                'evaluation_history': [eval.get('evaluation_timestamp') for eval in self.evaluation_history]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating evaluation summary: {e}")
            return {'error': str(e)}
    
    def clear_history(self):
        """Clear evaluation history."""
        self.evaluation_history = []
        logger.info("Evaluation history cleared")
    
    def export_results(self, format: str = 'json', filename: Optional[str] = None) -> str:
        """
        Export evaluation results.
        
        Args:
            format: Export format ('json', 'csv', 'excel')
            filename: Optional filename
            
        Returns:
            Path to exported file
        """
        try:
            if filename is None:
                filename = f"evaluation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            if format.lower() == 'json':
                filepath = self.output_dir / f"{filename}.json"
                with open(filepath, 'w') as f:
                    json.dump(self.evaluation_history, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                # Flatten results for CSV export
                flattened_results = []
                for evaluation in self.evaluation_history:
                    flat_result = self._flatten_dict(evaluation)
                    flattened_results.append(flat_result)
                
                df = pd.DataFrame(flattened_results)
                filepath = self.output_dir / f"{filename}.csv"
                df.to_csv(filepath, index=False)
            
            elif format.lower() == 'excel':
                # Flatten results for Excel export
                flattened_results = []
                for evaluation in self.evaluation_history:
                    flat_result = self._flatten_dict(evaluation)
                    flattened_results.append(flat_result)
                
                df = pd.DataFrame(flattened_results)
                filepath = self.output_dir / f"{filename}.xlsx"
                df.to_excel(filepath, index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Results exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Handle list of dictionaries
                for i, item in enumerate(v):
                    items.extend(self._flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)