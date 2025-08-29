"""
Moral Value Classification System

A system for classifying moral values in text using Jonathan Haidt's
Moral Foundations Theory with state-of-the-art pre-trained models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Dict, List
import warnings
import logging
from constants import MORAL_FOUNDATIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class MoralValueClassifier:
    """
    A moral value classification system based on Jonathan Haidt's Moral Foundations Theory.
    
    This system classifies text according to the 6 core moral foundations:
    1. Care/Harm - Promoting well-being, avoiding harm
    2. Fairness/Cheating - Justice, equality, rights
    3. Loyalty/Betrayal - Trust, commitment, allegiance
    4. Authority/Subversion - Leadership, hierarchy, respect
    5. Sanctity/Degradation - Purity, sacredness, spirituality
    6. Liberty/Oppression - Freedom, autonomy, independence
    """
    
    def __init__(self, model_type: str = "moral_foundations"):
        """
        Initialize the moral value classifier.
        
        Args:
            model_type: Type of model to use ("moral_foundations" or "value_alignment")
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.labels = []
        
        # Import moral foundations from constants
        self.moral_foundations = MORAL_FOUNDATIONS
        
        # Model configurations
        self.model_configs = {
            "moral_foundations": {
                "model_name": "facebook/bart-large-mnli",
                "description": "BART model for zero-shot moral foundations classification",
                "task": "zero_shot_classification"
            },
            "value_alignment": {
                "model_name": "facebook/bart-large-mnli",
                "description": "BART model for general value alignment classification",
                "labels": ["morally good", "morally neutral", "morally bad"],
                "task": "zero_shot_classification"
            }
        }
        
        # Load the specified model
        self._load_model()
    
    def _load_model(self):
        """Load the specified pre-trained model."""
        try:
            config = self.model_configs[self.model_type]
            model_name = config["model_name"]
            
            logger.info(f"Loading {self.model_type} model: {model_name}")
            
            if config["task"] == "zero_shot_classification":
                self.classifier = pipeline("zero-shot-classification", model=model_name)
            
            logger.info(f"Successfully loaded {self.model_type} model")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def classify_moral_foundations(self, text: str) -> Dict:
        """
        Classify text according to Moral Foundations Theory.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing moral foundation scores and analysis
        """
        try:
            if not text or not text.strip():
                return {"error": "Empty or invalid text provided"}
            
            # Use zero-shot classification with moral foundation labels
            candidate_labels = [foundation["name"] for foundation in self.moral_foundations.values()]
            
            result = self.classifier(text, candidate_labels)
            
            # Process results and add detailed analysis
            foundation_scores = {}
            for i, label in enumerate(result["labels"]):
                score = result["scores"][i]
                foundation_scores[label] = score
            
            # Find the dominant foundation
            dominant_foundation = result["labels"][0] if result["labels"] else "unknown"
            dominant_score = result["scores"][0] if result["scores"] else 0.0
            
            # Analyze keywords for additional insights
            keyword_analysis = self._analyze_keywords(text)
            
            return {
                "text": text,
                "dominant_foundation": dominant_foundation,
                "dominant_score": dominant_score,
                "all_foundation_scores": foundation_scores,
                "keyword_analysis": keyword_analysis,
                "moral_analysis": self._generate_moral_analysis(dominant_foundation, dominant_score, keyword_analysis),
                "model_type": self.model_type
            }
                
        except Exception as e:
            logger.error(f"Error in moral foundations classification: {e}")
            return {"error": f"Moral foundations classification failed: {str(e)}"}
    
    def _analyze_keywords(self, text: str) -> Dict:
        """Analyze text for moral foundation keywords."""
        text_lower = text.lower()
        keyword_matches = {}
        
        for foundation_key, foundation_data in self.moral_foundations.items():
            positive_matches = [word for word in foundation_data["positive_keywords"] if word in text_lower]
            negative_matches = [word for word in foundation_data["negative_keywords"] if word in text_lower]
            
            keyword_matches[foundation_data["name"]] = {
                "positive_keywords": positive_matches,
                "negative_keywords": negative_matches,
                "total_matches": len(positive_matches) + len(negative_matches)
            }
        
        return keyword_matches
    
    def _generate_moral_analysis(self, dominant_foundation: str, score: float, keyword_analysis: Dict) -> Dict:
        """Generate detailed moral analysis based on classification results."""
        analysis = {
            "primary_foundation": dominant_foundation,
            "confidence": score,
            "interpretation": "",
            "key_insights": [],
            "moral_recommendation": ""
        }
        
        # Generate interpretation based on dominant foundation
        foundation_descriptions = {
            "Care/Harm": "This text primarily concerns the well-being and protection of others.",
            "Fairness/Cheating": "This text primarily concerns justice, equality, and fair treatment.",
            "Loyalty/Betrayal": "This text primarily concerns trust, commitment, and group allegiance.",
            "Authority/Subversion": "This text primarily concerns respect for authority and social order.",
            "Sanctity/Degradation": "This text primarily concerns purity, sacredness, and spiritual values.",
            "Liberty/Oppression": "This text primarily concerns freedom, autonomy, and individual rights."
        }
        
        analysis["interpretation"] = foundation_descriptions.get(dominant_foundation, "This text has unclear moral foundations.")
        
        # Add key insights from keyword analysis
        for foundation_name, keyword_data in keyword_analysis.items():
            if keyword_data["total_matches"] > 0:
                insight = f"{foundation_name}: Found {keyword_data['total_matches']} relevant keywords"
                if keyword_data["positive_keywords"]:
                    insight += f" (positive: {', '.join(keyword_data['positive_keywords'])})"
                if keyword_data["negative_keywords"]:
                    insight += f" (negative: {', '.join(keyword_data['negative_keywords'])})"
                analysis["key_insights"].append(insight)
        
        # Generate moral recommendation
        if score > 0.7:
            analysis["moral_recommendation"] = f"Strong evidence for {dominant_foundation} foundation. Consider this foundation when making moral judgments about this text."
        elif score > 0.5:
            analysis["moral_recommendation"] = f"Moderate evidence for {dominant_foundation} foundation. This foundation likely plays a role in the moral content."
        else:
            analysis["moral_recommendation"] = "Weak evidence for any single moral foundation. Consider multiple foundations or contextual factors."
        
        return analysis
    
    def classify_text(self, text: str) -> Dict:
        """
        Classify moral values in the given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing classification results
        """
        if self.model_type == "moral_foundations":
            return self.classify_moral_foundations(text)
        elif self.model_type == "value_alignment":
            return self._classify_value_alignment(text)
        else:
            return {"error": f"Unknown model type: {self.model_type}"}
    
    def _classify_value_alignment(self, text: str) -> Dict:
        """Classify text for general value alignment."""
        try:
            candidate_labels = ["morally good", "morally neutral", "morally bad"]
            
            result = self.classifier(text, candidate_labels)
            
            return {
                "text": text,
                "predicted_label": result["labels"][0] if result["labels"] else "unknown",
                "confidence": result["scores"][0] if result["scores"] else 0.0,
                "all_scores": dict(zip(result["labels"], result["scores"])) if result["labels"] else {},
                "model_type": self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error in value alignment classification: {e}")
            return {"error": f"Value alignment classification failed: {str(e)}"}
    
    def batch_classify(self, texts: List[str]) -> List[Dict]:
        """
        Classify multiple texts in batch.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of classification results
        """
        results = []
        for text in texts:
            result = self.classify_text(text)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        config = self.model_configs.get(self.model_type, {})
        return {
            "model_type": self.model_type,
            "model_name": config.get("model_name", "unknown"),
            "description": config.get("description", "No description available"),
            "moral_foundations": list(self.moral_foundations.keys()),
            "task": config.get("task", "unknown"),
            "model_loaded": self.classifier is not None
        }
    
    def get_moral_foundations_info(self) -> Dict:
        """Get detailed information about Moral Foundations Theory."""
        return {
            "theory": "Jonathan Haidt's Moral Foundations Theory",
            "description": "A psychological theory that identifies six moral foundations that influence human moral reasoning",
            "foundations": self.moral_foundations
        }


def main():
    """
    Demonstration of the Moral Foundations Theory-based classifier.
    """
    print("🤖 Moral Foundations Theory Classifier")
    print("=" * 50)
    print("Based on Jonathan Haidt's research on moral psychology")
    print()
    
    # Initialize the classifier with moral foundations
    classifier = MoralValueClassifier("moral_foundations")
    
    # Get theory information
    theory_info = classifier.get_moral_foundations_info()
    print(f"📚 {theory_info['theory']}")
    print(f"📖 {theory_info['description']}")
    print()
    
    # Test texts covering different moral foundations
    test_texts = [
        "The doctor showed great compassion while treating the patient, ensuring their safety and wellbeing.",
        "The judge made a fair decision based on evidence, treating everyone equally under the law.",
        "The soldier remained loyal to his unit, never betraying their trust even under pressure.",
        "The student respected the teacher's authority and followed the established classroom rules.",
        "The ceremony was sacred and pure, honoring the spiritual traditions of the community.",
        "The activist fought for freedom and individual rights, opposing oppressive government policies."
    ]
    
    print(f"📝 Testing with {len(test_texts)} texts covering different moral foundations...")
    print()
    
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. Text: {text}")
        result = classifier.classify_moral_foundations(text)
        
        if "error" not in result:
            print(f"   🎯 Dominant Foundation: {result['dominant_foundation']}")
            print(f"   📊 Confidence: {result['dominant_score']:.3f}")
            print(f"   💡 Analysis: {result['moral_analysis']['interpretation']}")
            
            # Show top 3 foundation scores
            all_scores = result['all_foundation_scores']
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   📈 Top Foundations:")
            for foundation, score in sorted_scores:
                print(f"      • {foundation}: {score:.3f}")
        else:
            print(f"   ❌ Error: {result['error']}")
        
        print()
    
    print(f"✅ Model loaded: {classifier.get_model_info()['model_name']}")
    print(f"🔬 Moral Foundations: {', '.join(classifier.get_model_info()['moral_foundations'])}")


if __name__ == "__main__":
    main()
