"""
Moral Value-Based Conversational Agent for Participatory Budget Project Recommendations
This agent uses an ensemble of specialized models and a two-step filtering and
scoring system to recommend projects.
"""

import pandas as pd
from typing import List, Dict
import warnings
import os
import sys
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from constants import MORAL_FOUNDATIONS_TO_USE, CATEGORY_KEYWORDS, SYNTHETIC_DATA_PATH, POLAND_DATA_PATH, WORLDWIDE_DATA_PATH
from utils import load_csv_data

warnings.filterwarnings('ignore')
class ProjectRecommender:
    """
    A conversational agent that recommends projects based on an ensemble of moral value classifiers.
    """
    
    def __init__(self, dataset_type: str = "synthetic", category_weight: float = 5.0, keyword_weight: float = 3.0, moral_weight: float = 2.0):
        self.dataset_type = dataset_type
        self.category_weight = category_weight
        self.keyword_weight = keyword_weight
        self.moral_weight = moral_weight
        
        paths = {
            "synthetic": SYNTHETIC_DATA_PATH,
            "poland": POLAND_DATA_PATH,
            "worldwide": WORLDWIDE_DATA_PATH
        }
        self.projects_csv_path = paths.get(dataset_type)

        print(f"Loading {dataset_type} projects...")
        self.projects_df = load_csv_data(self.projects_csv_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, RobertaForSequenceClassification] = {}
        self.tokenizer = None
        self._load_model_ensemble()
        
        self.projects_dict = self.projects_df.set_index('project_id').to_dict('index')
        self.moral_foundation_names = MORAL_FOUNDATIONS_TO_USE
        
        print(f"Agent initialized with {len(self.projects_df)} projects.")

    def _load_model_ensemble(self):
        print("Loading moral value classification model ensemble...")
        models_loaded = 0
        for foundation in MORAL_FOUNDATIONS_TO_USE:
            model_path = f"../models/best_roberta_model_{foundation}"
            try:
                model = RobertaForSequenceClassification.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                self.models[foundation] = model
                
                if self.tokenizer is None:
                    self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
                
                models_loaded += 1
            except Exception as e:
                print(f"Failed to load model for '{foundation}': {e}")
        
        print(f"Successfully loaded {models_loaded}/{len(MORAL_FOUNDATIONS_TO_USE)} specialized models.")

    def analyze_user_moral_values(self, user_input: str) -> Dict:
        print(f"\nAnalyzing user input: '{user_input}'")
        all_scores = {}
        for foundation, model in self.models.items():
            inputs = self.tokenizer(user_input, truncation=True, padding=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                score = probabilities[0][1].item()
            
            all_scores[foundation] = score

        sorted_foundations = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
        dominant_foundation, confidence = sorted_foundations[0]
        
        return {
            'dominant_foundation': dominant_foundation,
            'confidence': confidence,
            'all_foundation_scores': all_scores
        }

    def _calculate_moral_alignment_score(self, project: pd.Series, user_moral_scores: Dict) -> float:
        alignment = sum(project.get(f'moral_score_{f}', 0.0) * user_moral_scores.get(f, 0.0) for f in self.moral_foundation_names)
        return alignment

    def _count_keyword_matches(self, user_input: str, project: pd.Series) -> int:
        user_input_lower = user_input.lower()
        
        # Use English name for Poland dataset, otherwise use regular name
        project_name = project.get('name_english', project['name'])
        project_text = f"{project_name} {project.get('description', '')}".lower()
        
        user_keywords = {kw for cat_kws in CATEGORY_KEYWORDS.values() for kw in cat_kws if kw in user_input_lower}

        matches = sum(1 for kw in user_keywords if kw in project_text)
        return matches

    def _calculate_category_match_score(self, user_input: str, project: pd.Series) -> float:
        """
        Calculate category match score (0 or 1) based on whether the project's category
        matches the user's interest category.
        """
        user_input_lower = user_input.lower()
        project_category = project.get('category', '')
        
        # Find which category the user is interested in based on their keywords
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in user_input_lower for kw in keywords):
                # Return 1.0 if project category matches user interest, 0.0 otherwise
                return 1.0 if project_category == category else 0.0
        
        # If no category keywords found in user input, return 0.0
        return 0.0

    def _calculate_normalized_keyword_score(self, user_input: str, project: pd.Series) -> float:
        """
        Calculate normalized keyword score (0-1) based on the ratio of matched keywords
        to total possible keywords from user input.
        """
        user_input_lower = user_input.lower()
        
        # Use English name for Poland dataset, otherwise use regular name
        project_name = project.get('name_english', project['name'])
        project_text = f"{project_name} {project.get('description', '')}".lower()
        
        # Get all keywords that appear in user input
        user_keywords = {kw for cat_kws in CATEGORY_KEYWORDS.values() for kw in cat_kws if kw in user_input_lower}
        
        if not user_keywords:
            return 0.0
        
        # Count all keyword matches in project text
        matches = sum(1 for kw in user_keywords if kw in project_text)
        
        # Normalize to 0-1 range
        return matches / len(user_keywords)

    def find_matching_projects(self, preferences: Dict, top_n: int = 5) -> List[Dict]:
        if not preferences.get('moral_values', {}).get('all_foundation_scores'):
            return []
        
        user_moral_scores = preferences['moral_values']['all_foundation_scores']
        user_input = preferences['raw_input']
        
        # Score all projects using the three-component system
        scored_projects = []
        for _, project in self.projects_df.iterrows():
            # Calculate all three scores
            category_score = self._calculate_category_match_score(user_input, project)
            keyword_score = self._calculate_normalized_keyword_score(user_input, project)
            moral_score = self._calculate_moral_alignment_score(project, user_moral_scores)
            
            # Calculate weighted combination
            total_score = (category_score * self.category_weight) + (keyword_score * self.keyword_weight) + (moral_score * self.moral_weight)
            
            project_dict = project.to_dict()
            # Use English name for Poland dataset to match ground truth
            if 'name_english' in project_dict and pd.notna(project_dict['name_english']):
                project_dict['name'] = project_dict['name_english']
            project_dict['final_score'] = total_score
            project_dict['category_match_score'] = category_score
            project_dict['keyword_match_score'] = keyword_score
            project_dict['moral_alignment_score'] = moral_score
            project_dict['raw_keyword_matches'] = self._count_keyword_matches(user_input, project)
            scored_projects.append(project_dict)

        # Sort by total score and return top_n
        scored_projects.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_projects[:top_n]

    def generate_recommendations(self, user_input: str, top_n: int = 5) -> Dict:
        print("\nGenerating project recommendations...")
        moral_analysis = self.analyze_user_moral_values(user_input)
        
        preferences = {
            'moral_values': moral_analysis,
            'raw_input': user_input
        }
        
        matching_projects = self.find_matching_projects(preferences, top_n)
        
        if matching_projects:
            print(f"Found {len(matching_projects)} relevant projects.")
        else:
            print("No projects found matching your preferences.")
            
        return {'preferences': preferences, 'recommendations': matching_projects}

    def chat_interface(self):
        print("\nWelcome to the Moral Value-Based Project Recommender!")
        print("Tell me what's important to you (e.g., 'helping children', 'protecting nature'). Type 'quit' to exit.")
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input:
                continue
            
            results = self.generate_recommendations(user_input)
            if results['recommendations']:
                dominant_foundation = results['preferences']['moral_values']['dominant_foundation']
                print(f"\nThank you for sharing what's important to you. My analysis suggests you have a strong focus on {dominant_foundation}. Based on that, here are some projects that seem to align with your values. Please review them to see if any are a good fit:")                
                for i, p in enumerate(results['recommendations'], 1):
                    print(f"\n{i}. {p['name']} (Final Score: {p['final_score']:.2f})")
                    print(f"   Category: {p['category']}   |   Cost: ${p.get('cost', 'N/A'):,}")
                    description = p.get('description', '')
                    print(f"   Description: {description if description and description.strip() else 'N/A'}")
                    print(f"   Category Score: {p['category_match_score']:.2f} | Keyword Score: {p['keyword_match_score']:.2f} - {p['raw_keyword_matches']} match(es) | Moral Score: {p['moral_alignment_score']:.2f}")
            else:
                print("\nI couldn't find specific projects, but I'm learning more about what you value.")

def main():
    """Main function to run the conversational agent."""
    dataset_type = "synthetic"
    try:
        choice = input("Select dataset (1-synthetic, 2-poland, 3-worldwide) [default: 1]: ").strip()
        if choice == "2": dataset_type = "poland"
        elif choice == "3": dataset_type = "worldwide"
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    print(f"\nInitializing agent with {dataset_type} dataset...")
    recommender = ProjectRecommender(dataset_type=dataset_type)
    recommender.chat_interface()

if __name__ == "__main__":
    main()