"""
Conversational Agent for Project Recommendation based on Moral Values and Preferences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
from collections import Counter

from moral_value_extractor import MoralValueExtractor
from constants import MORAL_FOUNDATION_KEYWORDS, GENERAL_VALUE_KEYWORDS

warnings.filterwarnings('ignore')

class ProjectRecommendationAgent:
    """
    A conversational agent that recommends projects based on user's moral values and preferences.
    """
    
    def __init__(self, votes_csv_path: str, projects_csv_path: str):
        """
        Initialize the recommendation agent.
        
        Args:
            votes_csv_path: Path to the votes CSV file
            projects_csv_path: Path to the projects CSV file
        """
        self.votes_csv_path = votes_csv_path
        self.projects_csv_path = projects_csv_path
        
        # Load data
        print("🔄 Loading voting data and projects...")
        self.votes_df = pd.read_csv(votes_csv_path)
        self.projects_df = pd.read_csv(projects_csv_path)
        
        # Initialize moral value extractor
        print("🔄 Initializing moral value extractor...")
        self.moral_extractor = MoralValueExtractor()
        
        # Create project mapping
        self.projects_dict = self.projects_df.set_index('project_id').to_dict('index')
        
        # Get available categories
        self.available_categories = sorted(self.projects_df['category'].unique())
        
        print(f"✅ Agent initialized with {len(self.votes_df)} votes and {len(self.projects_df)} projects")
        print(f"📊 Available categories: {', '.join(self.available_categories)}")
    
    def extract_user_preferences(self, user_input: str) -> Dict:
        """
        Extract user preferences from conversational input.
        
        Args:
            user_input: User's conversational input
            
        Returns:
            Dictionary containing extracted preferences
        """
        print(f"\n🔍 Analyzing user input: '{user_input}'")
        
        # Extract moral values
        moral_scores = self.moral_extractor.extract_values(user_input, threshold=0.1)
        print(f"📊 Detected moral values: {list(moral_scores.keys())}")
        
        # Extract category preferences using keyword matching
        category_preferences = self._extract_category_preferences(user_input)
        print(f"🏷️  Detected category preferences: {category_preferences}")
        
        # Extract demographic info if mentioned
        demographics = self._extract_demographics(user_input)
        
        return {
            'moral_values': moral_scores,
            'category_preferences': category_preferences,
            'demographics': demographics,
            'raw_input': user_input
        }
    
    def _extract_category_preferences(self, text: str) -> List[str]:
        """Extract category preferences from text using keyword matching."""
        text_lower = text.lower()
        preferences = []
        
        # Define keywords for each category with better health vs environment distinction
        category_keywords = {
            'Environment, public health & safety': {
                'health_focus': ['health', 'wellness', 'medical', 'healthcare', 'safety', 'well-being'],
                'environment_focus': ['environment', 'green', 'clean', 'renewable', 'pollution', 'climate', 'sustainability'],
                'general': ['public', 'safety']
            },
            'Culture & community': ['culture', 'community', 'social', 'people', 'together', 'unity', 'diversity', 'heritage'],
            'Education': ['education', 'learning', 'school', 'students', 'knowledge', 'training', 'skills', 'academic'],
            'Facilities, parks & recreation': ['facilities', 'parks', 'recreation', 'playground', 'sports', 'leisure', 'entertainment'],
            'Streets, Sidewalks & Transit': ['streets', 'sidewalks', 'transit', 'transportation', 'walking', 'biking', 'roads', 'infrastructure'],
            'Streets': ['streets', 'roads', 'traffic', 'pavement', 'streetlights'],
            'Sidewalks & Transit': ['sidewalks', 'walking', 'transit', 'public transport', 'buses', 'trains']
        }
        
        # Special handling for Environment, public health & safety category
        env_health_category = 'Environment, public health & safety'
        if env_health_category in category_keywords:
            env_health_keywords = category_keywords[env_health_category]
            
            # Check if user mentions health more than environment
            health_count = sum(1 for keyword in env_health_keywords['health_focus'] if keyword in text_lower)
            env_count = sum(1 for keyword in env_health_keywords['environment_focus'] if keyword in text_lower)
            
            if health_count > env_count:
                # User focuses on health - add this category
                preferences.append(env_health_category)
            elif env_count > 0:
                # User focuses on environment - add this category
                preferences.append(env_health_category)
            elif any(keyword in text_lower for keyword in env_health_keywords['general']):
                # User mentions general safety/public - add this category
                preferences.append(env_health_category)
        
        # Check other categories
        for category, keywords in category_keywords.items():
            if category != env_health_category:  # Skip the one we already handled
                if any(keyword in text_lower for keyword in keywords):
                    preferences.append(category)
        
        return preferences
    
    def _extract_demographics(self, text: str) -> Dict:
        """Extract demographic information from text."""
        text_lower = text.lower()
        demographics = {}
        
        # Age extraction
        age_keywords = ['young', 'old', 'elderly', 'senior', 'youth', 'teen', 'adult', 'middle-aged']
        for keyword in age_keywords:
            if keyword in text_lower:
                if keyword in ['young', 'youth', 'teen']:
                    demographics['age_group'] = 'young'
                elif keyword in ['old', 'elderly', 'senior']:
                    demographics['age_group'] = 'senior'
                elif keyword in ['adult', 'middle-aged']:
                    demographics['age_group'] = 'adult'
                break
        
        # Education extraction
        education_keywords = ['college', 'university', 'graduate', 'high school', 'phd', 'degree']
        for keyword in education_keywords:
            if keyword in text_lower:
                demographics['education'] = keyword
                break
        
        # Gender extraction
        gender_keywords = ['male', 'female', 'man', 'woman', 'boy', 'girl']
        for keyword in gender_keywords:
            if keyword in text_lower:
                demographics['sex'] = keyword
                break
        
        return demographics
    
    def find_similar_voters(self, user_preferences: Dict, top_k: int = 50) -> pd.DataFrame:
        """
        Find voters with similar preferences to the user.
        
        Args:
            user_preferences: User's extracted preferences
            top_k: Number of similar voters to find
            
        Returns:
            DataFrame of similar voters
        """
        print(f"\n🔍 Finding voters with similar preferences...")
        
        # Calculate similarity scores
        similarity_scores = []
        
        for idx, voter in self.votes_df.iterrows():
            score = 0
            
            # Moral value similarity
            if user_preferences['moral_values']:
                user_top_moral = max(user_preferences['moral_values'].items(), key=lambda x: x[1])[0]
                if voter['top_moral_value'] == user_top_moral:
                    score += 3
            
            # Category similarity
            user_categories = user_preferences['category_preferences']
            voter_categories = [voter['top_category_1'], voter['top_category_2'], voter['top_category_3']]
            
            for user_cat in user_categories:
                if user_cat in voter_categories:
                    score += 2
            
            # Demographic similarity
            demographics = user_preferences['demographics']
            if demographics.get('age_group'):
                if demographics['age_group'] == 'young' and voter['age'] < 30:
                    score += 1
                elif demographics['age_group'] == 'senior' and voter['age'] > 60:
                    score += 1
                elif demographics['age_group'] == 'adult' and 30 <= voter['age'] <= 60:
                    score += 1
            
            if demographics.get('sex') and voter['sex']:
                if demographics['sex'].lower() in str(voter['sex']).lower():
                    score += 1
            
            similarity_scores.append((idx, score))
        
        # Sort by similarity score and get top voters
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_voter_indices = [idx for idx, score in similarity_scores[:top_k]]
        
        similar_voters = self.votes_df.loc[top_voter_indices].copy()
        similar_voters['similarity_score'] = [score for idx, score in similarity_scores[:top_k]]
        
        print(f"✅ Found {len(similar_voters)} similar voters")
        return similar_voters
    
    def get_project_recommendations(self, similar_voters: pd.DataFrame, top_k: int = 5) -> List[Dict]:
        """
        Get project recommendations based on similar voters.
        
        Args:
            similar_voters: DataFrame of similar voters
            top_k: Number of top projects to recommend
            
        Returns:
            List of recommended projects with scores
        """
        print(f"\n🎯 Generating project recommendations...")
        
        # Count project votes from similar voters
        project_votes = Counter()
        project_scores = {}
        
        for idx, voter in similar_voters.iterrows():
            vote_str = voter['vote']
            if pd.notna(vote_str):
                project_ids = [int(pid.strip()) for pid in str(vote_str).split(',') if pid.strip().isdigit()]
                
                for project_id in project_ids:
                    if project_id in self.projects_dict:
                        project_votes[project_id] += 1
                        
                        # Weight by similarity score
                        similarity = voter['similarity_score']
                        if project_id not in project_scores:
                            project_scores[project_id] = 0
                        project_scores[project_id] += similarity
        
        # Get top projects
        top_projects = []
        for project_id, vote_count in project_votes.most_common(top_k * 2):  # Get more to filter
            if project_id in self.projects_dict:
                project_data = self.projects_dict[project_id]
                
                # Calculate recommendation score
                base_score = vote_count
                similarity_bonus = project_scores.get(project_id, 0) / max(1, vote_count)
                final_score = base_score + similarity_bonus
                
                top_projects.append({
                    'project_id': project_id,
                    'name': project_data.get('name', 'Unknown'),
                    'category': project_data.get('category', 'Unknown'),
                    'description': project_data.get('description', 'No description'),
                    'cost': project_data.get('cost', 'Unknown'),
                    'vote_count': vote_count,
                    'similarity_bonus': similarity_bonus,
                    'final_score': final_score
                })
        
        # Sort by final score and return top_k
        top_projects.sort(key=lambda x: x['final_score'], reverse=True)
        recommendations = top_projects[:top_k]
        
        print(f"✅ Generated {len(recommendations)} project recommendations")
        return recommendations
    
    def chat_and_recommend(self, user_input: str) -> str:
        """
        Main conversational method that processes user input and returns recommendations.
        
        Args:
            user_input: User's conversational input
            
        Returns:
            Formatted response with recommendations
        """
        # Extract user preferences
        user_preferences = self.extract_user_preferences(user_input)
        
        # Find similar voters
        similar_voters = self.find_similar_voters(user_preferences)
        
        # Get project recommendations
        recommendations = self.get_project_recommendations(similar_voters)
        
        # Format response
        response = self._format_recommendations(user_preferences, recommendations)
        
        return response
    
    def _format_recommendations(self, user_preferences: Dict, recommendations: List[Dict]) -> str:
        """Format the recommendations into a readable response."""
        
        response = "🤖 **Project Recommendation Agent**\n\n"
        
        # User preferences summary
        response += "📋 **Your Preferences:**\n"
        if user_preferences['moral_values']:
            top_moral = max(user_preferences['moral_values'].items(), key=lambda x: x[1])[0]
            response += f"  • Primary moral value: {top_moral}\n"
        
        if user_preferences['category_preferences']:
            response += f"  • Category interests: {', '.join(user_preferences['category_preferences'])}\n"
        
        if user_preferences['demographics']:
            demo_str = ', '.join([f"{k}: {v}" for k, v in user_preferences['demographics'].items()])
            response += f"  • Demographics: {demo_str}\n"
        
        response += "\n🎯 **Top 5 Project Recommendations:**\n\n"
        
        for i, project in enumerate(recommendations, 1):
            response += f"{i}. **{project['name']}**\n"
            response += f"   📍 Category: {project['category']}\n"
            response += f"   💰 Cost: {project['cost']}\n"
            response += f"   📝 Description: {project['description']}\n" # Changed to show full description
            response += f"   ⭐ Score: {project['final_score']:.1f} (based on {project['vote_count']} similar votes)\n\n"
        
        response += "💡 *These recommendations are based on voters with similar moral values and preferences to yours.*"
        
        return response

def main():
    """Main function to run the conversational agent."""
    print("🚀 Starting Project Recommendation Agent...")
    
    # Initialize agent
    agent = ProjectRecommendationAgent(
        votes_csv_path='data/parsed/worldwide_mechanical-turk/votes.csv',
        projects_csv_path='data/parsed/worldwide_mechanical-turk/projects.csv'
    )
    
    print("\n💬 Welcome! I'm your Project Recommendation Agent.")
    print("Tell me about your values, interests, and what matters to you.")
    print("I'll recommend projects that align with your preferences!")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thanks for using the Project Recommendation Agent!")
                break
            
            if not user_input:
                continue
            
            # Get recommendations
            response = agent.chat_and_recommend(user_input)
            print(f"\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
