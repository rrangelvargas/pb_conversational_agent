"""
Enhanced Moral Value Classifier

This version supports multiple models and provides comprehensive evaluation metrics
without requiring ground truth labels.
"""

import torch
import numpy as np
from transformers import pipeline
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from constants import MORAL_FOUNDATIONS
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class EnhancedMoralValueClassifier:
    """
    Enhanced moral value classifier with support for multiple models and evaluation.
    """
    
    def __init__(self, model_name: str = "roberta-large-mnli", model_type: str = "moral_foundations"):
        """
        Initialize the enhanced classifier.
        
        Args:
            model_name: Name of the model to use
            model_type: Type of classification to perform
        """
        self.model_name = model_name
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Load the model
        self.logger.info(f"Loading enhanced {model_type} model: {model_name}")
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Improved prompts for each moral foundation
        self.improved_prompts = {
            "Care/Harm": "This project focuses on caring for people, protecting safety, helping others, or preventing harm",
            "Fairness/Cheating": "This project promotes equality, justice, fair treatment, equal opportunities, or addresses unfairness",
            "Loyalty/Betrayal": "This project builds community bonds, strengthens group identity, or promotes loyalty to community",
            "Authority/Subversion": "This project respects established systems, follows rules, or maintains social order",
            "Sanctity/Degradation": "This project protects sacred values, maintains purity, or prevents degradation of important things",
            "Liberty/Oppression": "This project promotes freedom, autonomy, individual rights, or opposes oppression"
        }
        
        # Conservative calibration factors for realistic balance
        self.calibration_factors = {
            "Care/Harm": 0.9,
            "Fairness/Cheating": 1.4,
            "Loyalty/Betrayal": 1.2,
            "Authority/Subversion": 1.1,
            "Sanctity/Degradation": 1.2,
            "Liberty/Oppression": 1.3
        }
        
        # Evaluation metrics storage
        self.evaluation_history = []
        
    def classify_moral_foundations(self, text: str, apply_calibration: bool = True) -> Dict:
        """
        Classify text using enhanced moral foundations analysis.
        
        Args:
            text: Text to classify
            apply_calibration: Whether to apply calibration factors
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Use improved prompts for classification
            candidate_labels = list(self.improved_prompts.keys())
            
            # Classify with improved prompts
            result = self.classifier(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template="This text is about: {}",
                multi_label=False
            )
            
            # Extract scores
            scores = result['scores']
            labels = result['labels']
            
            # Create score dictionary
            moral_scores = dict(zip(labels, scores))
            
            # Apply calibration if requested
            if apply_calibration:
                calibrated_scores = {}
                for foundation, score in moral_scores.items():
                    calibration = self.calibration_factors.get(foundation, 1.0)
                    calibrated_scores[foundation] = min(1.0, score * calibration)
            else:
                calibrated_scores = moral_scores.copy()
            
            # Find dominant foundation
            dominant_foundation = max(calibrated_scores.items(), key=lambda x: x[1])
            
            # Find secondary foundation
            sorted_foundations = sorted(calibrated_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_foundation = sorted_foundations[1] if len(sorted_foundations) > 1 else (None, 0.0)
            
            # Generate analysis
            analysis = self._generate_enhanced_analysis(dominant_foundation, secondary_foundation, calibrated_scores, text)
            
            # Store evaluation data
            self._store_evaluation_data(text, dominant_foundation, calibrated_scores)
            
            return {
                'dominant_foundation': dominant_foundation[0],
                'confidence': dominant_foundation[1],
                'secondary_foundation': secondary_foundation[0],
                'secondary_confidence': secondary_foundation[1],
                'all_foundation_scores': calibrated_scores,
                'analysis': analysis,
                'raw_scores': moral_scores,
                'calibrated_scores': calibrated_scores,
                'model_used': self.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error in moral foundations classification: {e}")
            return {
                'error': f'Classification failed: {str(e)}',
                'dominant_foundation': 'Unknown',
                'confidence': 0.0,
                'secondary_foundation': 'Unknown',
                'secondary_confidence': 0.0,
                'all_foundation_scores': {},
                'model_used': self.model_name
            }
    
    def _generate_enhanced_analysis(self, dominant_foundation: Tuple, secondary_foundation: Tuple, scores: Dict, text: str):
        """
        Generate enhanced analysis using both dominant and secondary foundations.
        
        Args:
            dominant_foundation: Tuple of (foundation_name, score)
            secondary_foundation: Tuple of (foundation_name, score)
            scores: Dictionary of all foundation scores
            text: Original text
            
        Returns:
            Analysis string
        """
        foundation_name, score = dominant_foundation
        secondary_name, secondary_score = secondary_foundation
        
        # Get foundation descriptions
        foundation_info = MORAL_FOUNDATIONS.get(foundation_name.lower().replace('/', '_'), {})
        secondary_info = MORAL_FOUNDATIONS.get(secondary_name.lower().replace('/', '_'), {}) if secondary_name else {}
        
        description = foundation_info.get('description', 'This moral foundation')
        secondary_description = secondary_info.get('description', 'This moral foundation') if secondary_name else ''
        
        # Generate analysis based on score strength
        if score > 0.6:
            strength = "strongly"
        elif score > 0.4:
            strength = "moderately"
        else:
            strength = "slightly"
        
        analysis = f"This text {strength} concerns {description.lower()}. "
        
        # Add secondary foundation if it's significant
        if secondary_name and secondary_score > 0.25:
            if secondary_score > 0.4:
                secondary_strength = "strongly"
            elif secondary_score > 0.3:
                secondary_strength = "moderately"
            else:
                secondary_strength = "slightly"
            
            analysis += f"It also {secondary_strength} shows elements of {secondary_description.lower()}. "
        
        # Add model information
        analysis += f"[Analysis by {self.model_name}]"
        
        return analysis
    
    def _store_evaluation_data(self, text: str, dominant_foundation: Tuple, scores: Dict):
        """
        Store evaluation data for analysis.
        
        Args:
            text: Original text
            dominant_foundation: Dominant foundation and score
            scores: All foundation scores
        """
        evaluation_data = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'dominant_foundation': dominant_foundation[0],
            'dominant_score': dominant_foundation[1],
            'all_scores': scores,
            'model_used': self.model_name
        }
        
        self.evaluation_history.append(evaluation_data)
    
    def get_evaluation_metrics(self) -> Dict:
        """
        Calculate evaluation metrics from stored data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.evaluation_history:
            return {'error': 'No evaluation data available'}
        
        # Extract metrics
        dominant_foundations = [data['dominant_foundation'] for data in self.evaluation_history]
        dominant_scores = [data['dominant_score'] for data in self.evaluation_history]
        
        # Distribution analysis
        foundation_counts = {}
        for foundation in self.improved_prompts.keys():
            foundation_counts[foundation] = dominant_foundations.count(foundation)
        
        total_samples = len(dominant_foundations)
        
        # Calculate metrics
        metrics = {
            'total_samples': total_samples,
            'foundation_distribution': foundation_counts,
            'foundation_percentages': {k: v/total_samples*100 for k, v in foundation_counts.items()},
            'mean_confidence': np.mean(dominant_scores),
            'std_confidence': np.std(dominant_scores),
            'max_foundation_percentage': max(foundation_counts.values()) / total_samples * 100,
            'distribution_entropy': self._calculate_entropy(foundation_counts.values()),
            'model_used': self.model_name
        }
        
        return metrics
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution."""
        total = sum(values)
        if total == 0:
            return 0
        probabilities = [v/total for v in values if v > 0]
        return -sum(p * np.log2(p) for p in probabilities)
    
    def classify_text(self, text: str, apply_calibration: bool = True) -> Dict:
        """
        Main classification method.
        
        Args:
            text: Text to classify
            apply_calibration: Whether to apply calibration factors
            
        Returns:
            Classification results
        """
        if self.model_type == "moral_foundations":
            return self.classify_moral_foundations(text, apply_calibration)
        else:
            return {'error': f'Unknown model type: {self.model_type}'}
    
    def save_evaluation_report(self, output_path: str = None):
        """
        Save evaluation report to file.
        
        Args:
            output_path: Path to save the report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/evaluation/evaluation_{self.model_name.replace('/', '_')}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = {
            'model_info': {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'metrics': self.get_evaluation_metrics(),
            'evaluation_history': self.evaluation_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation report saved to: {output_path}")
        return output_path


def create_classifier_comparison(test_texts: List[str], models: List[str] = None) -> Dict:
    """
    Create a comparison of different classifiers on the same test texts.
    
    Args:
        test_texts: List of texts to test
        models: List of model names to compare
        
    Returns:
        Comparison results
    """
    if models is None:
        models = [
            "facebook/bart-large-mnli",
            "roberta-large-mnli", 
            "microsoft/DialoGPT-medium"
        ]
    
    comparison_results = {}
    
    for model_name in models:
        try:
            print(f"Testing model: {model_name}")
            classifier = EnhancedMoralValueClassifier(model_name)
            
            model_results = {
                'classifications': [],
                'metrics': {}
            }
            
            # Classify all test texts
            for text in test_texts:
                result = classifier.classify_text(text)
                model_results['classifications'].append(result)
            
            # Get evaluation metrics
            model_results['metrics'] = classifier.get_evaluation_metrics()
            
            comparison_results[model_name] = model_results
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            comparison_results[model_name] = {'error': str(e)}
    
    return comparison_results


def main():
    """
    Test the enhanced classifier.
    """
    print("Testing Enhanced Moral Value Classifier")
    print("=" * 55)
    
    # Test with different models
    test_texts = [
        "Equal access to education for all students regardless of background",
        "Economic fairness and job training opportunities for everyone",
        "Community safety and crime prevention programs",
        "Environmental protection and sustainability initiatives",
        "Cultural diversity and inclusion programs",
        "Supporting local businesses and economic development",
        "Preserving historical landmarks and cultural heritage",
        "Improving public transportation and reducing traffic congestion"
    ]
    
    models_to_test = [
        "facebook/bart-large-mnli",
        "roberta-large-mnli"
    ]
    
    print("Running model comparison...")
    comparison_results = create_classifier_comparison(test_texts, models_to_test)
    
    # Print results
    for model_name, results in comparison_results.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            continue
        
        # Print metrics
        metrics = results['metrics']
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Mean confidence: {metrics['mean_confidence']:.3f}")
        print(f"Distribution entropy: {metrics['distribution_entropy']:.3f}")
        print(f"Max foundation percentage: {metrics['max_foundation_percentage']:.1f}%")
        
        print("\nFoundation Distribution:")
        for foundation, count in metrics['foundation_distribution'].items():
            percentage = metrics['foundation_percentages'][foundation]
            print(f"  • {foundation}: {count} ({percentage:.1f}%)")
        
        # Print sample classifications
        print(f"\nSample Classifications:")
        for i, classification in enumerate(results['classifications'][:3]):
            if 'error' not in classification:
                print(f"  {i+1}. '{test_texts[i][:50]}...'")
                print(f"     → {classification['dominant_foundation']} ({classification['confidence']:.3f})")
                print(f"     → {classification['analysis']}")


if __name__ == "__main__":
    main()
