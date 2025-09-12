"""
Model Comparison Framework for Moral Value Classification

This module provides tools to compare different models for moral value classification
without requiring ground truth labels. It uses various evaluation metrics that can
assess model quality based on consistency, diversity, and semantic coherence.
"""

import torch
import numpy as np
import pandas as pd
from transformers import pipeline
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import os
from datetime import datetime

from constants import MORAL_FOUNDATIONS
from utils import save_csv_data, print_separator

warnings.filterwarnings('ignore')

class ModelComparisonFramework:
    """
    Framework for comparing different moral value classification models.
    """
    
    def __init__(self, test_data_path: str = "data/generated/content.csv"):
        """
        Initialize the comparison framework.
        
        Args:
            test_data_path: Path to test data
        """
        self.test_data_path = test_data_path
        self.logger = logging.getLogger(__name__)
        
        # Define models to compare
        self.models = {
            "facebook/bart-large-mnli": {
                "name": "BART-Large-MNLI",
                "description": "Current baseline model"
            },
            "roberta-large-mnli": {
                "name": "RoBERTa-Large-MNLI", 
                "description": "Alternative transformer model"
            },
            "microsoft/DialoGPT-medium": {
                "name": "DialoGPT-Medium",
                "description": "Conversational model"
            }
        }
        
        # Moral foundations
        self.moral_foundations = [
            "Care/Harm", "Fairness/Cheating", "Loyalty/Betrayal",
            "Authority/Subversion", "Sanctity/Degradation", "Liberty/Oppression"
        ]
        
        # Improved prompts for each moral foundation
        self.improved_prompts = {
            "Care/Harm": "This project focuses on caring for people, protecting safety, helping others, or preventing harm",
            "Fairness/Cheating": "This project promotes equality, justice, fair treatment, equal opportunities, or addresses unfairness",
            "Loyalty/Betrayal": "This project builds community bonds, strengthens group identity, or promotes loyalty to community",
            "Authority/Subversion": "This project respects established systems, follows rules, or maintains social order",
            "Sanctity/Degradation": "This project protects sacred values, maintains purity, or prevents degradation of important things",
            "Liberty/Oppression": "This project promotes freedom, autonomy, individual rights, or opposes oppression"
        }
        
        # Results storage
        self.results = {}
        self.test_data = None
        
    def load_test_data(self, sample_size: int = 100) -> pd.DataFrame:
        """
        Load test data for evaluation.
        
        Args:
            sample_size: Number of samples to use for testing
            
        Returns:
            DataFrame with test data
        """
        try:
            if os.path.exists(self.test_data_path):
                df = pd.read_csv(self.test_data_path)
                # Sample if needed
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                self.test_data = df
                self.logger.info(f"Loaded {len(df)} test samples")
                return df
            else:
                self.logger.warning(f"Test data not found at {self.test_data_path}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            return pd.DataFrame()
    
    def create_classifier(self, model_name: str):
        """
        Create a classifier for a specific model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Pipeline classifier
        """
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            return classifier
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def classify_with_model(self, classifier, text: str) -> Dict:
        """
        Classify text using a specific model.
        
        Args:
            classifier: Model classifier
            text: Text to classify
            
        Returns:
            Classification results
        """
        try:
            candidate_labels = list(self.improved_prompts.keys())
            
            result = classifier(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template="This text is about: {}",
                multi_label=False
            )
            
            scores = result['scores']
            labels = result['labels']
            
            # Create score dictionary
            moral_scores = dict(zip(labels, scores))
            
            # Find dominant foundation
            dominant_foundation = max(moral_scores.items(), key=lambda x: x[1])
            
            # Find secondary foundation
            sorted_foundations = sorted(moral_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_foundation = sorted_foundations[1] if len(sorted_foundations) > 1 else (None, 0.0)
            
            return {
                'dominant_foundation': dominant_foundation[0],
                'confidence': dominant_foundation[1],
                'secondary_foundation': secondary_foundation[0],
                'secondary_confidence': secondary_foundation[1],
                'all_foundation_scores': moral_scores,
                'raw_scores': moral_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error in classification: {e}")
            return {
                'error': f'Classification failed: {str(e)}',
                'dominant_foundation': 'Unknown',
                'confidence': 0.0,
                'secondary_foundation': 'Unknown',
                'secondary_confidence': 0.0,
                'all_foundation_scores': {}
            }
    
    def evaluate_model(self, model_name: str) -> Dict:
        """
        Evaluate a single model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Create classifier
        classifier = self.create_classifier(model_name)
        if classifier is None:
            return {'error': f'Failed to load model {model_name}'}
        
        # Initialize results
        model_results = {
            'model_name': model_name,
            'classifications': [],
            'dominant_foundations': [],
            'confidences': [],
            'score_matrices': []
        }
        
        # Classify all test samples
        for idx, row in self.test_data.iterrows():
            text = row.get('description', '')
            if not text:
                continue
                
            result = self.classify_with_model(classifier, text)
            
            if 'error' not in result:
                model_results['classifications'].append(result)
                model_results['dominant_foundations'].append(result['dominant_foundation'])
                model_results['confidences'].append(result['confidence'])
                
                # Store score matrix for similarity analysis
                scores = [result['all_foundation_scores'].get(foundation, 0.0) 
                         for foundation in self.moral_foundations]
                model_results['score_matrices'].append(scores)
        
        # Calculate evaluation metrics
        evaluation_metrics = self.calculate_evaluation_metrics(model_results)
        model_results.update(evaluation_metrics)
        
        return model_results
    
    def calculate_evaluation_metrics(self, model_results: Dict) -> Dict:
        """
        Calculate evaluation metrics for a model.
        
        Args:
            model_results: Results from model evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not model_results['classifications']:
            return {'error': 'No successful classifications'}
        
        # 1. Distribution Analysis
        foundation_counts = Counter(model_results['dominant_foundations'])
        total_samples = len(model_results['dominant_foundations'])
        
        distribution_metrics = {
            'foundation_distribution': dict(foundation_counts),
            'foundation_percentages': {k: v/total_samples*100 for k, v in foundation_counts.items()},
            'distribution_entropy': self.calculate_entropy(foundation_counts.values()),
            'max_foundation_percentage': max(foundation_counts.values()) / total_samples * 100
        }
        
        # 2. Confidence Analysis
        confidences = model_results['confidences']
        confidence_metrics = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'high_confidence_rate': sum(1 for c in confidences if c > 0.7) / len(confidences),
            'low_confidence_rate': sum(1 for c in confidences if c < 0.3) / len(confidences)
        }
        
        # 3. Consistency Analysis (using score matrices)
        score_matrix = np.array(model_results['score_matrices'])
        
        # Calculate pairwise similarities between classifications
        similarities = cosine_similarity(score_matrix)
        np.fill_diagonal(similarities, 0)  # Remove self-similarities
        
        consistency_metrics = {
            'mean_pairwise_similarity': np.mean(similarities),
            'std_pairwise_similarity': np.std(similarities),
            'score_matrix_variance': np.var(score_matrix, axis=0).tolist()
        }
        
        # 4. Diversity Analysis
        diversity_metrics = {
            'unique_foundations_used': len(foundation_counts),
            'foundation_coverage': len(foundation_counts) / len(self.moral_foundations),
            'gini_coefficient': self.calculate_gini_coefficient(list(foundation_counts.values()))
        }
        
        return {
            'distribution_metrics': distribution_metrics,
            'confidence_metrics': confidence_metrics,
            'consistency_metrics': consistency_metrics,
            'diversity_metrics': diversity_metrics
        }
    
    def calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution."""
        total = sum(values)
        if total == 0:
            return 0
        probabilities = [v/total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probabilities)
    
    def calculate_gini_coefficient(self, values: List[int]) -> float:
        """Calculate Gini coefficient for distribution inequality."""
        if not values:
            return 0
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def compare_models(self) -> Dict:
        """
        Compare all models and generate comprehensive analysis.
        
        Returns:
            Comparison results
        """
        self.logger.info("Starting model comparison...")
        
        # Load test data
        if self.test_data is None:
            self.load_test_data()
        
        if self.test_data.empty:
            return {'error': 'No test data available'}
        
        # Evaluate each model
        for model_name in self.models.keys():
            self.logger.info(f"Evaluating {model_name}...")
            self.results[model_name] = self.evaluate_model(model_name)
        
        # Generate comparison analysis
        comparison_analysis = self.generate_comparison_analysis()
        
        return {
            'model_results': self.results,
            'comparison_analysis': comparison_analysis
        }
    
    def generate_comparison_analysis(self) -> Dict:
        """
        Generate comprehensive comparison analysis.
        
        Returns:
            Comparison analysis results
        """
        analysis = {
            'summary': {},
            'recommendations': [],
            'detailed_comparison': {}
        }
        
        # Compare key metrics across models
        metrics_comparison = {}
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
                
            metrics_comparison[model_name] = {
                'distribution_entropy': results['distribution_metrics']['distribution_entropy'],
                'mean_confidence': results['confidence_metrics']['mean_confidence'],
                'consistency': results['consistency_metrics']['mean_pairwise_similarity'],
                'diversity': results['diversity_metrics']['foundation_coverage'],
                'max_foundation_percentage': results['distribution_metrics']['max_foundation_percentage']
            }
        
        # Find best model for each metric
        best_models = {}
        for metric in ['distribution_entropy', 'mean_confidence', 'consistency', 'diversity']:
            if metric == 'max_foundation_percentage':
                # Lower is better for this metric
                best_model = min(metrics_comparison.items(), key=lambda x: x[1][metric])
            else:
                # Higher is better for other metrics
                best_model = max(metrics_comparison.items(), key=lambda x: x[1][metric])
            best_models[metric] = best_model[0]
        
        # Generate recommendations
        recommendations = []
        
        # Check for balanced distribution
        balanced_models = []
        for model_name, metrics in metrics_comparison.items():
            if metrics['max_foundation_percentage'] < 40:  # No foundation dominates too much
                balanced_models.append(model_name)
        
        if balanced_models:
            recommendations.append(f"Models with balanced distribution: {', '.join(balanced_models)}")
        
        # Check for high confidence
        high_confidence_models = []
        for model_name, metrics in metrics_comparison.items():
            if metrics['mean_confidence'] > 0.6:
                high_confidence_models.append(model_name)
        
        if high_confidence_models:
            recommendations.append(f"Models with high confidence: {', '.join(high_confidence_models)}")
        
        # Overall recommendation
        if metrics_comparison:
            # Simple scoring system
            scores = {}
            for model_name, metrics in metrics_comparison.items():
                score = (
                    metrics['distribution_entropy'] * 0.3 +
                    metrics['mean_confidence'] * 0.3 +
                    metrics['consistency'] * 0.2 +
                    metrics['diversity'] * 0.2 -
                    metrics['max_foundation_percentage'] * 0.01  # Penalty for dominance
                )
                scores[model_name] = score
            
            best_overall = max(scores.items(), key=lambda x: x[1])
            recommendations.append(f"Overall best model: {best_overall[0]} (score: {best_overall[1]:.3f})")
        
        analysis['summary'] = {
            'best_models': best_models,
            'metrics_comparison': metrics_comparison
        }
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def generate_visualizations(self, output_dir: str = "results/comparisons"):
        """
        Generate visualization plots for model comparison.
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Distribution comparison
        self.plot_foundation_distributions(output_dir)
        
        # 2. Confidence comparison
        self.plot_confidence_comparison(output_dir)
        
        # 3. Consistency analysis
        self.plot_consistency_analysis(output_dir)
        
        # 4. Overall metrics comparison
        self.plot_overall_metrics(output_dir)
    
    def plot_foundation_distributions(self, output_dir: str):
        """Plot foundation distributions for all models."""
        plt.figure(figsize=(15, 8))
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'error' in results:
                continue
                
            plt.subplot(1, 3, i+1)
            
            distribution = results['distribution_metrics']['foundation_distribution']
            foundations = list(distribution.keys())
            counts = list(distribution.values())
            
            plt.bar(foundations, counts)
            plt.title(f'{self.models[model_name]["name"]}\nFoundation Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/foundation_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confidence_comparison(self, output_dir: str):
        """Plot confidence distributions for all models."""
        plt.figure(figsize=(12, 6))
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
                
            confidences = results['confidences']
            plt.hist(confidences, alpha=0.7, label=self.models[model_name]["name"], bins=20)
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/confidence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_consistency_analysis(self, output_dir: str):
        """Plot consistency analysis using PCA."""
        plt.figure(figsize=(12, 8))
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if 'error' in results or not results['score_matrices']:
                continue
                
            plt.subplot(1, 3, i+1)
            
            # PCA for visualization
            score_matrix = np.array(results['score_matrices'])
            pca = PCA(n_components=2)
            reduced_scores = pca.fit_transform(score_matrix)
            
            plt.scatter(reduced_scores[:, 0], reduced_scores[:, 1], alpha=0.6)
            plt.title(f'{self.models[model_name]["name"]}\nScore Space (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/consistency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_overall_metrics(self, output_dir: str):
        """Plot overall metrics comparison."""
        metrics = ['distribution_entropy', 'mean_confidence', 'consistency', 'diversity']
        model_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
                
            model_names.append(self.models[model_name]["name"])
            for metric in metrics:
                if metric == 'distribution_entropy':
                    value = results['distribution_metrics']['distribution_entropy']
                elif metric == 'mean_confidence':
                    value = results['confidence_metrics']['mean_confidence']
                elif metric == 'consistency':
                    value = results['consistency_metrics']['mean_pairwise_similarity']
                elif metric == 'diversity':
                    value = results['diversity_metrics']['foundation_coverage']
                
                metric_values[metric].append(value)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model_name in enumerate(model_names):
            values = [metric_values[metric][i] for metric in metrics]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison', size=15, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/overall_metrics_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: str = "results/comparisons"):
        """
        Save comparison results to files.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"{output_dir}/model_comparison_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary report
        summary_file = f"{output_dir}/comparison_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("MODEL COMPARISON SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"Model: {self.models[model_name]['name']}\n")
                f.write(f"Description: {self.models[model_name]['description']}\n")
                
                if 'error' in results:
                    f.write(f"Status: ERROR - {results['error']}\n")
                else:
                    f.write(f"Status: SUCCESS\n")
                    f.write(f"Samples processed: {len(results['classifications'])}\n")
                    f.write(f"Distribution entropy: {results['distribution_metrics']['distribution_entropy']:.3f}\n")
                    f.write(f"Mean confidence: {results['confidence_metrics']['mean_confidence']:.3f}\n")
                    f.write(f"Consistency: {results['consistency_metrics']['mean_pairwise_similarity']:.3f}\n")
                    f.write(f"Diversity: {results['diversity_metrics']['foundation_coverage']:.3f}\n")
                
                f.write("\n" + "-" * 30 + "\n\n")
        
        self.logger.info(f"Results saved to {output_dir}")
        return results_file, summary_file


def main():
    """
    Run the model comparison framework.
    """
    print("Model Comparison Framework for Moral Value Classification")
    print("=" * 60)
    
    # Initialize framework
    framework = ModelComparisonFramework()
    
    # Run comparison
    print("Running model comparison...")
    comparison_results = framework.compare_models()
    
    if 'error' in comparison_results:
        print(f"Error: {comparison_results['error']}")
        return
    
    # Generate visualizations
    print("Generating visualizations...")
    framework.generate_visualizations()
    
    # Save results
    print("Saving results...")
    results_file, summary_file = framework.save_results()
    
    # Print summary
    print("\nComparison completed!")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    
    # Print recommendations
    if 'comparison_analysis' in comparison_results:
        analysis = comparison_results['comparison_analysis']
        print("\nRecommendations:")
        for rec in analysis.get('recommendations', []):
            print(f"  • {rec}")


if __name__ == "__main__":
    main()
