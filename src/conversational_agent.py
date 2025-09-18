"""
Moral Value-Based Conversational Agent for Participatory Budget Project Recommendations

This agent uses Jonathan Haidt's Moral Foundations Theory to analyze user input
and recommend participatory budget projects that align with their moral values.
"""

import pandas as pd
from typing import List, Dict
import warnings
import os

from moral_value_classifier import MoralValueClassifier
from constants import MORAL_FOUNDATIONS
from utils import load_csv_data

warnings.filterwarnings('ignore')

class MoralValueProjectRecommender:
    """
    A conversational agent that recommends participatory budget projects based on 
    users' moral values detected through Moral Foundations Theory analysis.
    """
    
    def __init__(self, projects_csv_path: str = "data/generated/balanced_synthetic_projects.csv", use_finetuned: bool = True):
        """
        Initialize the moral value-based project recommender.
        
        Args:
            projects_csv_path: Path to the projects CSV file
            use_finetuned: Whether to use fine-tuned model if available
        """
        self.projects_csv_path = projects_csv_path
        
        # Load data
        print("Loading participatory budget projects...")
        self.projects_df = load_csv_data(projects_csv_path)
        
        # Initialize moral value classifier
        print("Loading moral value classification model...")
        self.moral_classifier = MoralValueClassifier("roberta-large-mnli")
        
        # Try to load fine-tuned model if requested
        if use_finetuned:
            self._load_finetuned_model()
        
        # Create project mapping
        self.projects_dict = self.projects_df.set_index('project_id').to_dict('index')
        
        # Get available categories
        self.available_categories = sorted(self.projects_df['category'].unique())
        
        # Get moral foundation names from constants
        self.moral_foundation_names = [foundation["name"] for foundation in MORAL_FOUNDATIONS.values()]
        
        print(f"Agent initialized with {len(self.projects_df)} projects")
        print(f"Available categories: {', '.join(self.available_categories)}")
        print(f"Using Moral Foundations Theory: {', '.join(self.moral_foundation_names)}")
    
    def _load_finetuned_model(self):
        """Try to load the latest fine-tuned model."""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            print("No models directory found. Using zero-shot model.")
            return
        
        # Find the latest model
        model_dirs = [d for d in os.listdir(models_dir) if d.startswith('moral_value_roberta_mftc_')]
        if not model_dirs:
            print("No fine-tuned models found. Using zero-shot model.")
            return
        
        latest_model = sorted(model_dirs)[-1]
        model_path = os.path.join(models_dir, latest_model, 'final_model')
        
        try:
            print(f"Loading fine-tuned model from: {model_path}")
            self.moral_classifier.load_finetuned_model(model_path)
            self.use_finetuned = True
            print("Fine-tuned model loaded successfully!")
        except Exception as e:
            print(f"Failed to load fine-tuned model: {e}")
            print("Falling back to zero-shot model.")
            self.use_finetuned = False
    
    def analyze_user_moral_values(self, user_input: str) -> Dict:
        """
        Analyze user input to extract moral values using Moral Foundations Theory.
        
        Args:
            user_input: User's conversational input
            
        Returns:
            Dictionary containing moral foundation analysis
        """
        print(f"\nAnalyzing user input: '{user_input}'")
        
        # Use the appropriate classifier (fine-tuned or zero-shot)
        if hasattr(self, 'use_finetuned') and self.use_finetuned:
            result = self.moral_classifier.classify_with_finetuned(user_input)
        else:
            result = self.moral_classifier.classify_moral_foundations(user_input)
        
        if "error" in result:
            print(f"Failed to analyze moral values: {result['error']}")
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
        """Extract category preferences from text using balanced keyword matching."""
        text_lower = text.lower()
        preferences = []
        
        # Define keywords for each category - BALANCED and more specific
        category_keywords = {
            "Education": ['education', 'learning', 'school', 'students', 'training', 'skills', 'knowledge', 'teach', 'teaching', 'literacy', 'academic', 'curriculum', 'classroom', 'tutoring', 'mentoring', 'library', 'books', 'reading'],
            "Health": ['health', 'healthcare', 'medical', 'wellness', 'mental health', 'vaccination', 'clinic', 'hospital', 'doctor', 'nurse', 'therapy', 'treatment', 'medicine', 'preventive care', 'nutrition', 'diet', 'fitness', 'exercise', 'safety'],
            "Environment, public health & safety": ['safety', 'security', 'emergency', 'preparedness', 'disaster', 'hazardous', 'lead', 'contamination', 'monitoring', 'air quality', 'water quality', 'soil', 'abatement', 'pollution', 'waste', 'cleanup'],
            "Facilities, parks & recreation": ['parks', 'recreation', 'facilities', 'playground', 'leisure', 'outdoor', 'pool', 'splash', 'park', 'recreational', 'amenities', 'community center', 'benches', 'tables', 'picnic'],
            "Streets, Sidewalks & Transit": ['streets', 'sidewalks', 'transit', 'transportation', 'traffic', 'walking', 'biking', 'pedestrian', 'curb', 'ramp', 'accessibility', 'mobility', 'crossing', 'road', 'highway', 'bridge', 'walkable'],
            "urban greenery": ['greenery', 'trees', 'plants', 'green space', 'nature', 'forest', 'garden', 'canopy', 'vegetation', 'landscaping', 'green roof', 'rain garden', 'tree planting'],
            "sport": ['sports', 'athletics', 'fitness', 'exercise', 'training', 'competition', 'team', 'league', 'field', 'court', 'equipment', 'adaptive sports', 'tennis', 'soccer', 'basketball', 'swimming', 'basketball court'],
            "public space": ['public space', 'plaza', 'square', 'gathering', 'community space', 'downtown', 'bench', 'seating', 'wayfinding', 'signage', 'marketplace', 'civic', 'toilet', 'restroom'],
            "public transit and roads": ['transit', 'roads', 'transportation', 'buses', 'trains', 'infrastructure', 'bus shelter', 'bike lane', 'pedestrian', 'accessibility', 'real-time', 'app', 'metro', 'subway', 'monitors'],
            "welfare": ['welfare', 'social services', 'assistance', 'help', 'food', 'backpack', 'diaper', 'formula', 'emergency', 'shelter', 'voucher', 'low-income', 'poverty', 'homeless', 'charity', 'donation', 'laundry'],
            "environmental protection": ['environmental protection', 'conservation', 'sustainability', 'climate', 'wildlife', 'environment', 'environmental', 'green', 'eco', 'renewable', 'clean energy', 'pollution', 'waste reduction', 'recycling', 'carbon', 'energy conversion'],
            "culture": ['culture', 'cultural', 'arts', 'heritage', 'tradition', 'creative', 'art', 'music', 'festival', 'performance', 'theater', 'museum', 'gallery', 'artist', 'exhibition', 'concert'],
            "Culture & community": ['community', 'togetherness', 'neighborhood', 'local', 'multicultural', 'diversity', 'storytelling', 'oral history', 'community center', 'gathering', 'celebration', 'social', 'unity', 'bonds']
        }
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return categories with scores above threshold (more balanced)
        threshold = 1  # Require at least 1 keyword match
        for category, score in category_scores.items():
            if score >= threshold:
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
            
            # Apply category preference bonus - BALANCED weights to prevent overfitting
            if preferences.get('category_preferences'):
                if project['category'] in preferences['category_preferences']:
                    score *= 2.0  # Reduced from 3.5 to 2.0 for more balanced matching
                    
                    # Additional bonus for exact keyword matches
                    user_input_lower = preferences.get('raw_input', '').lower()
                    project_name_lower = project['name'].lower()
                    project_desc_lower = project['description'].lower()
                    
                    # Check for specific keyword matches in project name/description
                    category_keywords = {
                        "Education": ['education', 'learning', 'school', 'students', 'training', 'skills', 'knowledge', 'teach', 'teaching', 'literacy', 'academic', 'curriculum', 'classroom', 'tutoring', 'mentoring'],
                        "Health": ['health', 'healthcare', 'medical', 'wellness', 'mental health', 'vaccination', 'clinic', 'hospital', 'doctor', 'nurse', 'therapy', 'treatment', 'medicine', 'preventive care', 'nutrition', 'diet'],
                        "Environment, public health & safety": ['safety', 'security', 'emergency', 'preparedness', 'disaster', 'hazardous', 'lead', 'contamination', 'monitoring', 'air quality', 'water quality', 'soil', 'abatement'],
                        "Facilities, parks & recreation": ['parks', 'recreation', 'facilities', 'playground', 'leisure', 'outdoor', 'pool', 'splash', 'park', 'recreational', 'amenities', 'community center'],
                        "Streets, Sidewalks & Transit": ['streets', 'sidewalks', 'transit', 'transportation', 'traffic', 'walking', 'biking', 'pedestrian', 'curb', 'ramp', 'accessibility', 'mobility', 'crossing'],
                        "sport": ['sports', 'athletics', 'fitness', 'exercise', 'training', 'competition', 'team', 'league', 'field', 'court', 'equipment', 'adaptive sports', 'tennis', 'soccer', 'basketball'],
                        "public transit and roads": ['transit', 'roads', 'transportation', 'buses', 'trains', 'infrastructure', 'bus shelter', 'bike lane', 'pedestrian', 'accessibility', 'real-time', 'app'],
                        "welfare": ['welfare', 'social services', 'assistance', 'help', 'food', 'backpack', 'diaper', 'formula', 'emergency', 'shelter', 'voucher', 'low-income', 'poverty', 'homeless'],
                        "environmental protection": ['environmental protection', 'conservation', 'sustainability', 'climate', 'wildlife', 'environment', 'environmental', 'green', 'eco', 'renewable', 'clean energy', 'pollution', 'waste reduction', 'recycling'],
                        "culture": ['culture', 'cultural', 'arts', 'heritage', 'tradition', 'creative', 'art', 'music', 'festival', 'performance', 'theater', 'museum', 'gallery', 'artist'],
                        "Culture & community": ['community', 'togetherness', 'neighborhood', 'local', 'multicultural', 'diversity', 'storytelling', 'oral history', 'community center', 'gathering', 'celebration']
                    }
                    
                    # Check for keyword matches in project details
                    project_category_keywords = category_keywords.get(project['category'], [])
                    keyword_matches = 0
                    for keyword in project_category_keywords:
                        if keyword in project_name_lower or keyword in project_desc_lower:
                            keyword_matches += 1
                    
                    # Additional bonus for keyword matches - REDUCED to prevent overfitting
                    if keyword_matches > 0:
                        score *= (1.0 + keyword_matches * 0.15)  # Reduced from 0.3 to 0.15
                    
                    # Position-based bonus for high-priority project types
                    priority_keywords = {
                        "Education": ['education', 'learning', 'school', 'students', 'literacy', 'academic'],
                        "Health": ['health', 'healthcare', 'medical', 'wellness', 'clinic', 'vaccination'],
                        "environmental protection": ['environmental', 'conservation', 'sustainability', 'green', 'eco'],
                        "Culture & community": ['community', 'culture', 'arts', 'heritage', 'festival'],
                        "sport": ['sports', 'athletics', 'fitness', 'exercise', 'recreation'],
                        "Facilities, parks & recreation": ['parks', 'recreation', 'facilities', 'playground']
                    }
                    
                    # Check for high-priority keywords in project name/description
                    project_category = project['category']
                    priority_words = priority_keywords.get(project_category, [])
                    priority_matches = 0
                    for word in priority_words:
                        if word in project_name_lower or word in project_desc_lower:
                            priority_matches += 1
                    
                    # Additional bonus for priority keyword matches - REDUCED
                    if priority_matches > 0:
                        score *= (1.0 + priority_matches * 0.1)  # Reduced from 0.25 to 0.1
                    
                    # Semantic similarity bonus - check for prompt keywords in project details
                    prompt_words = user_input_lower.split()
                    semantic_matches = 0
                    for word in prompt_words:
                        if len(word) > 3:  # Only consider words longer than 3 characters
                            if word in project_name_lower or word in project_desc_lower:
                                semantic_matches += 1
                    
                    # Bonus for semantic matches - REDUCED
                    if semantic_matches > 0:
                        score *= (1.0 + semantic_matches * 0.08)  # Reduced from 0.15 to 0.08
            
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
        
        # Sort by score and return top N with diversity consideration
        project_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply diversity mechanism to prevent overfitting to same projects
        diverse_results = []
        used_categories = set()
        
        # First pass: select best project from each category
        for project in project_scores:
            if len(diverse_results) >= top_n:
                break
            if project['category'] not in used_categories:
                diverse_results.append(project)
                used_categories.add(project['category'])
        
        # Second pass: fill remaining slots with highest scoring projects
        for project in project_scores:
            if len(diverse_results) >= top_n:
                break
            if project not in diverse_results:
                diverse_results.append(project)
        
        return diverse_results[:top_n]
    
    def _calculate_moral_alignment_score(self, project: pd.Series, user_moral_scores: Dict) -> float:
        """Calculate how well a project aligns with user's moral values with balanced scoring."""
        score = 0.0
        
        # Get project category
        project_category = project.get('category', '').lower()
        
        # Check if project has valid moral foundation scores
        has_valid_scores = False
        for foundation_name in self.moral_foundation_names:
            score_column = f'moral_score_{foundation_name}'
            if score_column in project and project[score_column] > 0:
                has_valid_scores = True
                break
        
        if has_valid_scores:
            # Use moral foundation scores if available
            for foundation_name in self.moral_foundation_names:
                score_column = f'moral_score_{foundation_name}'
                if score_column in project:
                    project_score = project[score_column]
                    user_score = user_moral_scores.get(foundation_name, 0.0)
                    
                    # BALANCED MORAL FOUNDATION SCORING - Remove hardcoded biases
                    foundation_weight = 1.0
                    
                    # Only apply moderate boosts based on general query patterns, not specific keywords
                    user_input_lower = self._get_user_input_keywords().lower()
                    
                    # General pattern matching with reduced bias
                    if any(pattern in user_input_lower for pattern in ['education', 'learning', 'school', 'student']):
                        # Education queries - moderate boost only
                        if foundation_name == 'Fairness/Cheating':
                            foundation_weight = 1.3  # Reduced from 1.8
                        elif foundation_name == 'Care/Harm':
                            foundation_weight = 1.2  # Reduced from 1.6
                    elif any(pattern in user_input_lower for pattern in ['environment', 'environmental', 'sustainability', 'green']):
                        # Environmental queries - moderate boost only
                        if foundation_name == 'Sanctity/Degradation':
                            foundation_weight = 1.3  # Reduced from 1.8
                        elif foundation_name == 'Care/Harm':
                            foundation_weight = 1.6  # Boost care for environmental protection
                        elif foundation_name == 'Authority/Subversion':
                            foundation_weight = 1.4  # Boost authority for environmental regulation
                    
                    # Calculate weighted score
                    weighted_score = project_score * user_score * foundation_weight
                    score += weighted_score
        else:
            # FALLBACK: Use category-based scoring when moral scores are invalid
            score = self._calculate_category_based_score(project, user_moral_scores)
        
        return max(score, 0.1)  # Ensure minimum score
    
    def _calculate_category_based_score(self, project: pd.Series, user_moral_scores: Dict) -> float:
        """Fallback scoring based on category and project content when moral scores are invalid."""
        score = 0.5  # Base score
        
        project_category = project.get('category', '').lower()
        project_name = project.get('name', '').lower()
        project_desc = project.get('description', '').lower()
        user_input_lower = self._get_user_input_keywords().lower()
        
        # Category matching bonus
        if any(keyword in user_input_lower for keyword in ['education', 'learning', 'school', 'student']):
            if 'education' in project_category:
                score += 0.3
        elif any(keyword in user_input_lower for keyword in ['recreation', 'parks', 'sports', 'leisure']):
            if any(cat in project_category for cat in ['recreation', 'sport', 'facilities']):
                score += 0.3
        elif any(keyword in user_input_lower for keyword in ['transit', 'transportation', 'walking', 'biking']):
            if any(cat in project_category for cat in ['transit', 'roads', 'sidewalks']):
                score += 0.3
        elif any(keyword in user_input_lower for keyword in ['environment', 'environmental', 'green', 'sustainability']):
            if any(cat in project_category for cat in ['environment', 'protection']):
                score += 0.3
        elif any(keyword in user_input_lower for keyword in ['culture', 'cultural', 'community', 'arts']):
            if any(cat in project_category for cat in ['culture', 'community']):
                score += 0.3
        elif any(keyword in user_input_lower for keyword in ['health', 'healthcare', 'wellness', 'medical']):
            if 'health' in project_category:
                score += 0.3
        elif any(keyword in user_input_lower for keyword in ['public space', 'amenities', 'livable']):
            if 'public space' in project_category:
                score += 0.3
        
        # Keyword matching bonus
        keyword_matches = 0
        for word in user_input_lower.split():
            if len(word) > 3 and (word in project_name or word in project_desc):
                keyword_matches += 1
        
        score += keyword_matches * 0.1
        
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
            print(f"Found {len(matching_projects)} projects matching your {dominant_foundation} values!")
        else:
            print("No projects found matching your preferences.")
        
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
                        print(f"   Description: {project['description']}")
                
                print(f"\nAnalyzed {results['total_projects_analyzed']} total projects")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing your request: {e}")


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
