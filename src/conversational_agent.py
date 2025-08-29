"""
Moral Value-Based Conversational Agent for Participatory Budget Project Recommendations

This agent uses Jonathan Haidt's Moral Foundations Theory to analyze user input
and recommend participatory budget projects that align with their moral values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
from collections import Counter

from moral_value_classifier import MoralValueClassifier
from constants import MORAL_FOUNDATIONS
from utils import load_csv_data, print_success, print_error, print_info

warnings.filterwarnings('ignore')

class MoralValueProjectRecommender:
    """
    A conversational agent that recommends participatory budget projects based on 
    users' moral values detected through Moral Foundations Theory analysis.
    """
    
    def __init__(self, projects_csv_path: str = "data/generated/content.csv"):
        """
        Initialize the moral value-based project recommender.
        
        Args:
            projects_csv_path: Path to the projects CSV file
        """
        self.projects_csv_path = projects_csv_path
        
        # Load data
        print_info("Loading participatory budget projects...")
        self.projects_df = load_csv_data(projects_csv_path)
        
        # Initialize moral value classifier
        print_info("Loading moral value classification model...")
        self.moral_classifier = MoralValueClassifier("moral_foundations")
        
        # Create project mapping
        self.projects_dict = self.projects_df.set_index('project_id').to_dict('index')
        
        # Get available categories
        self.available_categories = sorted(self.projects_df['category'].unique())
        
        # Get moral foundation names from constants
        self.moral_foundation_names = [foundation["name"] for foundation in MORAL_FOUNDATIONS.values()]
        
        print_success(f"Agent initialized with {len(self.projects_df)} projects")
        print_info(f"Available categories: {', '.join(self.available_categories)}")
        print_info(f"Using Moral Foundations Theory: {', '.join(self.moral_foundation_names)}")
    
    def analyze_user_moral_values(self, user_input: str) -> Dict:
        """
        Analyze user input to extract moral values using Moral Foundations Theory.
        
        Args:
            user_input: User's conversational input
            
        Returns:
            Dictionary containing moral foundation analysis
        """
        print(f"\nAnalyzing user input: '{user_input}'")
        
        # Use the moral value classifier
        result = self.moral_classifier.classify_moral_foundations(user_input)
        
        if "error" in result:
            print_error(f"Failed to analyze moral values: {result['error']}")
            return {}
        
        print(f"Detected moral values:")
        print(f"   Dominant Foundation: {result['dominant_foundation']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        if result.get('secondary_foundation'):
            print(f"   Secondary Foundation: {result['secondary_foundation']} ({result['secondary_confidence']:.3f})")
        print(f"   Analysis: {result['analysis']}")
        
        # Show top 3 foundation scores
        all_scores = result['all_foundation_scores']
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Top Foundations:")
        for foundation, score in sorted_scores:
            print(f"      • {foundation}: {score:.3f}")
        
        return result
    
    def extract_project_preferences(self, user_input: str) -> Dict:
        """
        Extract project preferences from user input.
        
        Args:
            user_input: User's conversational input
            
        Returns:
            Dictionary containing extracted preferences
        """
        # Store user input for category matching
        self.current_user_input = user_input
        
        # Analyze moral values
        moral_analysis = self.analyze_user_moral_values(user_input)
        
        # Extract category preferences using keyword matching
        category_preferences = self._extract_category_preferences(user_input)
        
        # Extract demographic targets if mentioned
        demographic_targets = self._extract_demographic_targets(user_input)
        
        # Extract cost preferences if mentioned
        cost_preferences = self._extract_cost_preferences(user_input)
        
        return {
            'moral_values': moral_analysis,
            'category_preferences': category_preferences,
            'demographic_targets': demographic_targets,
            'cost_preferences': cost_preferences,
            'raw_input': user_input
        }
    
    def _extract_category_preferences(self, text: str) -> List[str]:
        """Extract category preferences from text using keyword matching."""
        text_lower = text.lower()
        preferences = []
        
        # Define keywords for each category
        category_keywords = {
            "Education": ['education', 'learning', 'school', 'students', 'training', 'skills', 'knowledge'],
            "Health": ['health', 'healthcare', 'medical', 'wellness', 'fitness', 'mental health', 'care'],
            "Environment, public health & safety": ['environment', 'environmental', 'safety', 'health', 'pollution', 'air quality', 'green'],
            "Facilities, parks & recreation": ['parks', 'recreation', 'facilities', 'playground', 'sports', 'fitness', 'leisure'],
            "Streets, Sidewalks & Transit": ['streets', 'sidewalks', 'transit', 'transportation', 'traffic', 'walking', 'biking'],
            "urban greenery": ['greenery', 'trees', 'plants', 'green space', 'nature', 'forest', 'garden'],
            "sport": ['sports', 'athletics', 'fitness', 'exercise', 'training', 'competition'],
            "public space": ['public space', 'plaza', 'square', 'gathering', 'community space'],
            "public transit and roads": ['transit', 'roads', 'transportation', 'buses', 'trains', 'infrastructure'],
            "welfare": ['welfare', 'social services', 'support', 'assistance', 'help', 'community'],
            "environmental protection": ['environmental protection', 'conservation', 'sustainability', 'climate', 'wildlife', 'environment', 'environmental'],
            "culture": ['culture', 'cultural', 'arts', 'heritage', 'tradition', 'creative'],
            "Culture & community": ['community', 'culture', 'social', 'neighborhood', 'togetherness']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                preferences.append(category)
        
        return preferences
    
    def _extract_demographic_targets(self, text: str) -> List[str]:
        """Extract demographic targets from text."""
        text_lower = text.lower()
        targets = []
        
        demographic_keywords = {
            "children": ['children', 'kids', 'youth', 'young', 'students', 'school age'],
            "youth": ['youth', 'teenagers', 'adolescents', 'young people', 'students'],
            "adults": ['adults', 'working age', 'professionals', 'parents'],
            "seniors": ['seniors', 'elderly', 'older adults', 'retired', 'aging'],
            "people with disabilities": ['disabilities', 'disabled', 'accessibility', 'inclusive', 'special needs']
        }
        
        for target, keywords in demographic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                targets.append(target)
        
        return targets
    
    def _extract_cost_preferences(self, text: str) -> Dict:
        """Extract cost preferences from text."""
        text_lower = text.lower()
        preferences = {}
        
        # Check for cost-related keywords
        if any(word in text_lower for word in ['expensive', 'high cost', 'budget', 'affordable', 'cheap', 'low cost']):
            if any(word in text_lower for word in ['expensive', 'high cost']):
                preferences['cost_preference'] = 'high'
            elif any(word in text_lower for word in ['affordable', 'cheap', 'low cost']):
                preferences['cost_preference'] = 'low'
            else:
                preferences['cost_preference'] = 'any'
        
        return preferences
    
    def find_matching_projects(self, preferences: Dict, top_n: int = 5) -> List[Dict]:
        """
        Find projects that match the user's moral values and preferences.
        
        Args:
            preferences: User preferences including moral values
            top_n: Number of top projects to return
            
        Returns:
            List of matching projects with scores
        """
        if not preferences or 'moral_values' not in preferences:
            return []
        
        moral_analysis = preferences['moral_values']
        if not moral_analysis or 'all_foundation_scores' not in moral_analysis:
            return []
        
        # Get moral foundation scores
        user_moral_scores = moral_analysis['all_foundation_scores']
        
        # Calculate project scores based on moral alignment
        project_scores = []
        
        for _, project in self.projects_df.iterrows():
            # Start with moral alignment score
            score = self._calculate_moral_alignment_score(project, user_moral_scores)
            
            # Apply category preference bonus
            if preferences.get('category_preferences'):
                if project['category'] in preferences['category_preferences']:
                    score *= 1.2  # 20% bonus for category match
            
            # Apply demographic target bonus
            if preferences.get('demographic_targets'):
                project_targets = str(project.get('target', '')).split(',')
                if any(target.strip() in preferences['demographic_targets'] for target in project_targets):
                    score *= 1.1  # 10% bonus for demographic match
            
            # Apply cost preference bonus
            if preferences.get('cost_preferences', {}).get('cost_preference') == 'low':
                # Prefer lower cost projects
                cost_factor = 1.0 / (1.0 + (project['cost'] / 1000000))  # Normalize by 1M
                score *= cost_factor
            elif preferences.get('cost_preferences', {}).get('cost_preference') == 'high':
                # Prefer higher cost projects
                cost_factor = 1.0 + (project['cost'] / 1000000)  # Normalize by 1M
                score *= cost_factor
            
            project_scores.append({
                'project_id': project['project_id'],
                'name': project['name'],
                'category': project['category'],
                'cost': project['cost'],
                'description': project['description'],
                'target': project.get('target', ''),
                'moral_value': project.get('moral_value', ''),
                'score': score,
                'moral_alignment': self._get_moral_alignment_explanation(project, user_moral_scores)
            })
        
        # Sort by score and return top N
        project_scores.sort(key=lambda x: x['score'], reverse=True)
        return project_scores[:top_n]
    
    def _calculate_moral_alignment_score(self, project: pd.Series, user_moral_scores: Dict) -> float:
        """Calculate how well a project aligns with user's moral values with improved balance."""
        score = 0.0
        
        # Get project category
        project_category = project.get('category', '').lower()
        
        # Check if project has moral foundation scores
        for foundation_name in self.moral_foundation_names:
            score_column = f'moral_score_{foundation_name}'
            if score_column in project:
                project_score = project[score_column]
                user_score = user_moral_scores.get(foundation_name, 0.0)
                
                # IMPROVED MORAL FOUNDATION SCORING
                foundation_weight = 1.0
                
                # Boost relevant foundations based on query type
                if any(keyword in self._get_user_input_keywords().lower() for keyword in ['education', 'learning', 'school', 'teach', 'student', 'equal opportunities', 'opportunities']):
                    # Education queries
                    if foundation_name == 'Fairness/Cheating':
                        foundation_weight = 1.8
                    elif foundation_name == 'Liberty/Oppression':
                        foundation_weight = 1.6
                    elif foundation_name == 'Care/Harm':
                        foundation_weight = 1.2
                elif any(keyword in self._get_user_input_keywords().lower() for keyword in ['environment', 'environmental', 'sustainability', 'conservation', 'climate', 'wildlife', 'green']):
                    # Environmental queries
                    if foundation_name == 'Sanctity/Degradation':
                        foundation_weight = 1.8  # Boost sanctity for environmental protection
                    elif foundation_name == 'Care/Harm':
                        foundation_weight = 1.6  # Boost care for environmental protection
                    elif foundation_name == 'Authority/Subversion':
                        foundation_weight = 1.4  # Boost authority for environmental regulation
                
                # Calculate alignment with foundation weighting
                if project_score > 0.7 and user_score > 0.4:
                    alignment = project_score * user_score * 1.5 * foundation_weight
                elif project_score > 0.5 and user_score > 0.3:
                    alignment = project_score * user_score * 1.2 * foundation_weight
                else:
                    alignment = project_score * user_score * foundation_weight
                
                score += alignment
        
        # CATEGORY BOOSTING - Give significant weight to category matching
        category_boost = 1.0
        
        # Education-related queries
        if any(keyword in self._get_user_input_keywords().lower() for keyword in ['education', 'learning', 'school', 'teach', 'student', 'equal opportunities', 'opportunities']):
            if 'education' in project_category:
                category_boost = 2.5  # Major boost for education projects
            elif any(edu_keyword in project_category for edu_keyword in ['culture', 'arts', 'music', 'science']):
                category_boost = 1.8  # Good boost for related categories
        
        # Environmental-related queries
        elif any(keyword in self._get_user_input_keywords().lower() for keyword in ['environment', 'environmental', 'sustainability', 'conservation', 'climate', 'wildlife', 'green']):
            if any(env_keyword in project_category for env_keyword in ['environment', 'environmental', 'protection', 'sustainability']):
                category_boost = 2.5  # Major boost for environmental projects
            elif any(env_keyword in project_category for env_keyword in ['parks', 'greenery', 'nature', 'conservation']):
                category_boost = 2.0  # Good boost for nature-related categories
        
        # Apply category boost
        score *= category_boost
        
        # Add base score for having any moral foundation alignment
        if score > 0:
            score += 0.01  # Small base score to differentiate from zero
        
        return score
    
    def _get_user_input_keywords(self) -> str:
        """Extract keywords from the most recent user input for category matching."""
        # Store the user's input for category matching
        if hasattr(self, 'current_user_input'):
            return self.current_user_input
        return ""
    
    def _get_moral_alignment_explanation(self, project: pd.Series, user_moral_scores: Dict) -> str:
        """Generate explanation of moral alignment between project and user."""
        alignments = []
        
        for foundation_name in self.moral_foundation_names:
            score_column = f'moral_score_{foundation_name}'
            if score_column in project:
                project_score = project[score_column]
                user_score = user_moral_scores.get(foundation_name, 0.0)
                
                if project_score > 0.5 and user_score > 0.3:
                    alignments.append(f"Strong {foundation_name} alignment")
                elif project_score > 0.3 and user_score > 0.2:
                    alignments.append(f"Moderate {foundation_name} alignment")
        
        if alignments:
            return "; ".join(alignments[:3])  # Top 3 alignments
        else:
            return "Limited moral alignment"
    
    def generate_recommendations(self, user_input: str, top_n: int = 5) -> Dict:
        """
        Generate project recommendations based on user input.
        
        Args:
            user_input: User's conversational input
            top_n: Number of top recommendations to return
            
        Returns:
            Dictionary containing recommendations and analysis
        """
        print(f"\nGenerating project recommendations...")
        
        # Extract preferences
        preferences = self.extract_project_preferences(user_input)
        
        # Find matching projects
        matching_projects = self.find_matching_projects(preferences, top_n)
        
        # Generate recommendation summary
        if matching_projects:
            dominant_foundation = preferences['moral_values'].get('dominant_foundation', 'Unknown')
            print_success(f"Found {len(matching_projects)} projects matching your {dominant_foundation} values!")
        else:
            print_error("No projects found matching your preferences.")
        
        return {
            'preferences': preferences,
            'recommendations': matching_projects,
            'total_projects_analyzed': len(self.projects_df)
        }
    
    def chat_interface(self):
        """
        Interactive chat interface for the moral value-based project recommender.
        """
        print("Welcome to the Moral Value-Based Project Recommender!")
        print("=" * 70)
        print("I'll help you find participatory budget projects that align with your moral values.")
        print("Tell me what's important to you, and I'll suggest relevant projects.")
        print("\nExample inputs:")
        print("   • 'I care about helping children and families in need'")
        print("   • 'Environmental protection and sustainability are important to me'")
        print("   • 'I want to support education and equal opportunities for everyone'")
        print("   • 'Community safety and public health are my priorities'")
        print("\nType 'quit' to exit.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the Moral Value-Based Project Recommender!")
                    break
                
                if not user_input:
                    continue
                
                # Generate recommendations
                results = self.generate_recommendations(user_input, top_n=5)
                
                if results['recommendations']:
                    print(f"\nTop {len(results['recommendations'])} Project Recommendations:")
                    print("-" * 80)
                    
                    for i, project in enumerate(results['recommendations'], 1):
                        print(f"\n{i}. {project['name']}")
                        print(f"   Category: {project['category']}")
                        print(f"   Cost: ${project['cost']:,}")
                        print(f"   Target: {project['target']}")
                        print(f"   Moral Alignment: {project['moral_alignment']}")
                        print(f"   Match Score: {project['score']:.3f}")
                        print(f"   Description: {project['description'][:150]}...")
                
                print(f"\nAnalyzed {results['total_projects_analyzed']} total projects")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print_error(f"Error processing your request: {e}")


def main():
    """
    Main function to demonstrate the moral value-based project recommender.
    """
    print("Moral Value-Based Project Recommender")
    print("=" * 50)
    
    # Initialize the recommender
    recommender = MoralValueProjectRecommender()
    
    # Test with a sample input
    test_input = "I care about helping children and families in need, and environmental protection is important to me."
    print(f"\nTesting with: '{test_input}'")
    
    results = recommender.generate_recommendations(test_input, top_n=3)
    
    if results['recommendations']:
        print(f"\nFound {len(results['recommendations'])} matching projects!")
        print("\nTop recommendation:")
        top_project = results['recommendations'][0]
        print(f"{top_project['name']}")
        print(f"   Category: {top_project['category']}")
        print(f"   Moral Alignment: {top_project['moral_alignment']}")
        print(f"   Match Score: {top_project['score']:.3f}")
    
    # Start interactive chat
    print(f"\nStarting interactive chat interface...")
    recommender.chat_interface()


if __name__ == "__main__":
    main()
