"""
Moral Value Extractor - A system for identifying moral values in text using AI models.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import warnings

from constants import (
    DEFAULT_MODEL_NAME, 
    DEFAULT_THRESHOLD, 
    DEFAULT_TOP_K,
    MODEL_CONFIG,
    MORAL_FOUNDATION_KEYWORDS,
    GENERAL_VALUE_KEYWORDS,
    KEYWORD_CONFIDENCE_SCORES,
    MORAL_RECOMMENDATIONS,
    ANALYSIS_THRESHOLDS
)

warnings.filterwarnings('ignore')


class MoralValueExtractor:
    """
    A value extraction model based on moral reasoning for identifying moral values in text.
    
    This model uses a pre-trained RoBERTa model fine-tuned on moral stories 
    combined with keyword analysis to identify moral values, ethical principles, 
    and moral reasoning patterns in text based on Moral Foundations Theory.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the value extractor using a pre-trained moral reasoning model.
        
        Args:
            model_name: HuggingFace model name (default: RoBERTa model fine-tuned on moral stories)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        print(f"Loading moral reasoning model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def extract_values(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
        """
        Extract moral values from input text using the moral reasoning model.
        
        Args:
            text: Input text to analyze
            threshold: Confidence threshold for value detection
            
        Returns:
            Dictionary mapping value categories to confidence scores
        """
        if not text.strip():
            return {}
        
        # Get moral reasoning predictions
        moral_scores = self._get_moral_reasoning_predictions(text)
        
        # Analyze text for general moral values using keyword matching
        general_values = self._analyze_general_values(text)
        
        # Combine both approaches
        value_scores = {**moral_scores, **general_values}
        
        # Filter by threshold
        return self._filter_values_by_threshold(value_scores, threshold)
    
    def _get_moral_reasoning_predictions(self, text: str) -> Dict[str, float]:
        """
        Get predictions from the moral reasoning model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of moral foundation scores
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            **MODEL_CONFIG
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            scores = probabilities.cpu().numpy()[0]
        
        # Map model outputs to moral foundations
        value_scores = {}
        
        # The moral stories model predicts multiple moral dimensions
        # We'll map these to our moral foundations based on the highest scores
        if len(scores) > 0:
            # Find the highest scoring dimension
            max_score_idx = np.argmax(scores)
            max_score = float(scores[max_score_idx])
            
            # Map to appropriate moral foundation based on score strength
            if max_score > 0.6:
                # Map high scores to Care/Harm foundation
                value_scores['Care/Harm'] = max_score
            elif max_score > 0.4:
                # Map medium scores to Fairness/Cheating foundation
                value_scores['Fairness/Cheating'] = max_score
        
        return value_scores
    
    def _analyze_general_values(self, text: str) -> Dict[str, float]:
        """
        Analyze text for general moral values using keyword matching and context.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of detected values with confidence scores
        """
        text_lower = text.lower()
        value_scores = {}
        
        for value, keywords in GENERAL_VALUE_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                value_scores[value] = KEYWORD_CONFIDENCE_SCORES["general_values"]
        
        return value_scores
    
    def _filter_values_by_threshold(self, values: Dict[str, float], threshold: float) -> Dict[str, float]:
        """
        Filter values by confidence threshold.
        
        Args:
            values: Dictionary of values and scores
            threshold: Confidence threshold
            
        Returns:
            Filtered dictionary
        """
        return {k: v for k, v in values.items() if v >= threshold}
    
    def batch_extract_values(self, texts: List[str], threshold: float = DEFAULT_THRESHOLD) -> List[Dict[str, float]]:
        """
        Extract values from multiple texts in batch.
        
        Args:
            texts: List of input texts
            threshold: Confidence threshold for value detection
            
        Returns:
            List of dictionaries mapping value categories to confidence scores
        """
        results = []
        for text in texts:
            values = self.extract_values(text, threshold)
            results.append(values)
        return results
    
    def get_top_values(self, text: str, top_k: int = DEFAULT_TOP_K, threshold: float = DEFAULT_THRESHOLD) -> List[Tuple[str, float]]:
        """
        Get top-k most confident value predictions.
        
        Args:
            text: Input text
            top_k: Number of top values to return
            threshold: Confidence threshold
            
        Returns:
            List of (value, confidence) tuples sorted by confidence
        """
        values = self.extract_values(text, threshold)
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
        return sorted_values[:top_k]
    
    def analyze_moral_dilemma(self, text: str) -> Dict[str, any]:
        """
        Analyze text for moral dilemmas and conflicting values.
        
        Args:
            text: Input text describing a moral situation
            
        Returns:
            Dictionary containing dilemma analysis
        """
        values = self.extract_values(text, threshold=0.3)
        
        # Categorize values by confidence
        high_confidence_values, medium_confidence_values = self._categorize_values_by_confidence(values)
        
        # Identify potential conflicts
        potential_conflicts = self._identify_value_conflicts(high_confidence_values)
        
        analysis = {
            'primary_values': high_confidence_values,
            'secondary_values': medium_confidence_values,
            'potential_conflicts': potential_conflicts,
            'moral_complexity': self._calculate_moral_complexity(values),
            'recommendation': self._generate_moral_recommendation(values)
        }
        
        return analysis
    
    def _categorize_values_by_confidence(self, values: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        Categorize values by confidence levels.
        
        Args:
            values: Dictionary of values and scores
            
        Returns:
            Tuple of (high_confidence_values, medium_confidence_values)
        """
        high_confidence = [k for k, v in values.items() if v > ANALYSIS_THRESHOLDS["high_confidence"]]
        medium_confidence = [k for k, v in values.items() if ANALYSIS_THRESHOLDS["medium_confidence"] < v <= ANALYSIS_THRESHOLDS["high_confidence"]]
        
        return high_confidence, medium_confidence
    
    def _identify_value_conflicts(self, high_confidence_values: List[str]) -> List[str]:
        """
        Identify potential conflicts between detected values.
        
        Args:
            high_confidence_values: List of high confidence values
            
        Returns:
            List of potential conflicts
        """
        conflicts = []
        
        # Check for common value conflicts
        if 'Justice' in high_confidence_values and 'Compassion' in high_confidence_values:
            conflicts.append("Justice vs Compassion")
        if 'Authority' in high_confidence_values and 'Liberty' in high_confidence_values:
            conflicts.append("Authority vs Liberty")
        
        return conflicts
    
    def _calculate_moral_complexity(self, values: Dict[str, float]) -> float:
        """
        Calculate moral complexity score based on number of detected values.
        
        Args:
            values: Dictionary of detected values and scores
            
        Returns:
            Normalized complexity score (0.0 to 1.0)
        """
        return len(values) / 10.0
    
    def _generate_moral_recommendation(self, values: Dict[str, float]) -> str:
        """
        Generate a moral recommendation based on detected values.
        
        Args:
            values: Dictionary of detected values and scores
            
        Returns:
            String recommendation
        """
        if not values:
            return "Insufficient information to provide moral guidance."
        
        top_value = max(values.items(), key=lambda x: x[1])
        return MORAL_RECOMMENDATIONS.get(top_value[0], "Consider the moral implications carefully.")
