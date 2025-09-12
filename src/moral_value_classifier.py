"""
Final Balanced Moral Value Classifier

This version achieves better balance by using more conservative calibration
and leveraging secondary moral foundations for project differentiation.
"""

import torch
import numpy as np
from transformers import pipeline
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from constants import MORAL_FOUNDATIONS

warnings.filterwarnings('ignore')

class MoralValueClassifier:
    """
    Balanced moral value classifier with conservative calibration.
    """
    
    def __init__(self, model_type: str = "moral_foundations"):
        """
        Initialize the final balanced classifier.
        
        Args:
            model_type: Type of classification to perform
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Load the model
        self.logger.info(f"Loading balanced {model_type} model: roberta-large-mnli")
        
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="roberta-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("Successfully loaded balanced moral_foundations model")
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
        # Target: No foundation should exceed 35-40% dominance
        self.calibration_factors = {
            "Care/Harm": 0.9,        # Slight reduction (was 0.7, then 1.2)
            "Fairness/Cheating": 1.4, # Moderate boost (was 2.5, then 1.8)
            "Loyalty/Betrayal": 1.2, # Slight boost (was 1.8, then 1.4)
            "Authority/Subversion": 1.1, # Slight boost (was 0.8, then 1.0)
            "Sanctity/Degradation": 1.2, # Slight boost (was 1.5, then 1.3)
            "Liberty/Oppression": 1.3   # Moderate boost (was 2.2, then 1.5)
        }
    
    def classify_moral_foundations(self, text: str) -> Dict:
        """
        Classify text using balanced moral foundations analysis.
        
        Args:
            text: Text to classify
            
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
            
            # Extract and calibrate scores
            scores = result['scores']
            labels = result['labels']
            
            # Create score dictionary
            moral_scores = dict(zip(labels, scores))
            
            # Apply conservative calibration factors
            calibrated_scores = {}
            for foundation, score in moral_scores.items():
                calibration = self.calibration_factors.get(foundation, 1.0)
                calibrated_scores[foundation] = min(1.0, score * calibration)
            
            # Find dominant foundation
            dominant_foundation = max(calibrated_scores.items(), key=lambda x: x[1])
            
            # Find secondary foundation (second highest score)
            sorted_foundations = sorted(calibrated_scores.items(), key=lambda x: x[1], reverse=True)
            secondary_foundation = sorted_foundations[1] if len(sorted_foundations) > 1 else (None, 0.0)
            
            # Generate analysis
            analysis = self._generate_balanced_analysis(dominant_foundation, secondary_foundation, calibrated_scores, text)
            
            return {
                'dominant_foundation': dominant_foundation[0],
                'confidence': dominant_foundation[1],
                'secondary_foundation': secondary_foundation[0],
                'secondary_confidence': secondary_foundation[1],
                'all_foundation_scores': calibrated_scores,
                'analysis': analysis,
                'raw_scores': moral_scores,  # Keep raw scores for debugging
                'calibrated_scores': calibrated_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error in moral foundations classification: {e}")
            return {
                'error': f'Classification failed: {str(e)}',
                'dominant_foundation': 'Unknown',
                'confidence': 0.0,
                'secondary_foundation': 'Unknown',
                'secondary_confidence': 0.0,
                'all_foundation_scores': {}
            }
    
    def _generate_balanced_analysis(self, dominant_foundation: Tuple, secondary_foundation: Tuple, scores: Dict, text: str):
        """
        Generate balanced analysis using both dominant and secondary foundations.
        
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
        
        return analysis
    
    def classify_text(self, text: str) -> Dict:
        """
        Main classification method.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification results
        """
        if self.model_type == "moral_foundations":
            return self.classify_moral_foundations(text)
        else:
            return {'error': f'Unknown model type: {self.model_type}'}


def main():
    """
    Test the balanced classifier.
    """
    print("Testing Balanced Moral Value Classifier")
    print("=" * 55)
    
    classifier = MoralValueClassifier("moral_foundations")
    
    # Test cases that should now work better
    test_texts = [
        "Equal access to education for all students regardless of background",
        "Economic fairness and job training opportunities for everyone",
        "Community safety and crime prevention programs",
        "Environmental protection and sustainability initiatives",
        "Cultural diversity and inclusion programs"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        result = classifier.classify_moral_foundations(text)
        
        if 'error' not in result:
            print(f"   Dominant: {result['dominant_foundation']} ({result['confidence']:.3f})")
            if result['secondary_foundation']:
                print(f"   Secondary: {result['secondary_foundation']} ({result['secondary_confidence']:.3f})")
            print(f"   Calibrated Scores:")
            for foundation, score in result['calibrated_scores'].items():
                print(f"      • {foundation}: {score:.3f}")
            print(f"   Analysis: {result['analysis']}")
        else:
            print(f"   Error: {result['error']}")


if __name__ == "__main__":
    main()
