"""
Synthetic Data Generator

This module generates balanced synthetic participatory budget projects for testing
the moral value classifier and conversational agent.
"""

import pandas as pd
import random
from typing import List, Dict
import os
import sys
import json
from collections import Counter

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from moral_value_classifier import MoralValueClassifier

class SyntheticDataGenerator:
    """Generator for synthetic participatory budget projects."""
    
    def __init__(self):
        """Initialize the synthetic data generator."""
        
        # Initialize moral value classifier for scoring projects
        print("Initializing moral value classifier for project scoring...")
        self.moral_classifier = MoralValueClassifier("roberta-large-mnli")
        
        # Define categories with realistic project examples
        self.categories = {
            "Education": [
                "After-School Tutoring Program",
                "STEM Equipment for Elementary Schools", 
                "Adult Literacy Classes",
                "Library Renovation Project",
                "Computer Lab for Community Center",
                "Scholarship Fund for College",
                "Teacher Training Workshop",
                "Reading Program for Children",
                "Digital Learning Resources",
                "Career Counseling Services",
                "Language Learning Classes",
                "Special Education Support",
                "School Safety Improvements",
                "Educational Field Trips",
                "Parent Education Program"
            ],
            "Health": [
                "Community Health Clinic",
                "Mental Health Services",
                "Child Vaccination Program",
                "Senior Health Screenings",
                "Nutrition Education Classes",
                "Exercise Programs for Seniors",
                "Health Fair Organization",
                "Medication Assistance Program",
                "Dental Care for Children",
                "Mental Health Awareness Campaign",
                "Healthy Cooking Classes",
                "Fitness Equipment for Parks",
                "Health Education Materials",
                "Community Garden for Nutrition",
                "Wellness Center Development"
            ],
            "Environment, public health & safety": [
                "Air Quality Monitoring System",
                "Lead Paint Removal Program",
                "Emergency Preparedness Training",
                "Hazardous Waste Cleanup",
                "Water Quality Testing",
                "Safety Equipment for Fire Department",
                "Environmental Education Center",
                "Pollution Reduction Initiative",
                "Disaster Relief Fund",
                "Safety Lighting Installation",
                "Environmental Monitoring Station",
                "Hazardous Material Disposal",
                "Public Safety Campaign",
                "Emergency Response Training",
                "Environmental Health Assessment"
            ],
            "Facilities, parks & recreation": [
                "Community Center Renovation",
                "Playground Equipment Upgrade",
                "Park Benches and Tables",
                "Recreation Facility Maintenance",
                "Splash Pad Installation",
                "Picnic Area Development",
                "Sports Field Improvements",
                "Walking Trail Construction",
                "Restroom Facilities",
                "Outdoor Fitness Equipment",
                "Community Garden Space",
                "Amphitheater Construction",
                "Dog Park Development",
                "Skate Park Creation",
                "Recreation Program Equipment"
            ],
            "Streets, Sidewalks & Transit": [
                "Sidewalk Repair and Installation",
                "Street Lighting Improvements",
                "Bus Stop Shelter Construction",
                "Pedestrian Crossing Safety",
                "Bike Lane Development",
                "Traffic Signal Installation",
                "Road Resurfacing Project",
                "Accessibility Ramp Installation",
                "Parking Lot Improvements",
                "Transit Information System",
                "Street Tree Planting",
                "Traffic Calming Measures",
                "Public Transportation Enhancement",
                "Pedestrian Bridge Construction",
                "Transit Station Renovation"
            ],
            "Culture & community": [
                "Cultural Center Development",
                "Community Festival Funding",
                "Public Art Installation",
                "Heritage Preservation Project",
                "Multicultural Celebration",
                "Community Storytelling Program",
                "Local History Museum",
                "Arts and Crafts Workshop",
                "Community Theater Support",
                "Cultural Exchange Program",
                "Neighborhood Association Support",
                "Community Newsletter",
                "Local Artist Showcase",
                "Cultural Heritage Documentation",
                "Community Unity Events"
            ],
            "sport": [
                "Sports Equipment for Youth",
                "Basketball Court Renovation",
                "Tennis Court Construction",
                "Soccer Field Improvements",
                "Swimming Pool Maintenance",
                "Athletic Program Support",
                "Sports League Organization",
                "Fitness Center Equipment",
                "Adaptive Sports Program",
                "Sports Tournament Funding",
                "Athletic Scholarship Fund",
                "Sports Coaching Program",
                "Recreation League Support",
                "Sports Facility Upgrades",
                "Youth Sports Development"
            ]
        }
        
        # Define moral foundations with realistic distributions
        self.moral_foundations = [
            "Care/Harm",
            "Fairness/Cheating", 
            "Loyalty/Betrayal",
            "Authority/Subversion",
            "Sanctity/Degradation",
            "Liberty/Oppression"
        ]
        
        # Realistic moral value distribution (slightly biased toward Care/Harm for PB projects)
        self.moral_distribution = {
            "Care/Harm": 0.25,        # 25% - helping vulnerable people
            "Fairness/Cheating": 0.20, # 20% - equal access, social justice
            "Loyalty/Betrayal": 0.15,  # 15% - community bonds
            "Authority/Subversion": 0.15, # 15% - following regulations
            "Sanctity/Degradation": 0.15, # 15% - preserving heritage
            "Liberty/Oppression": 0.10   # 10% - individual freedom
        }
        
        # Secondary moral value mapping (projects often have multiple moral dimensions)
        self.secondary_mapping = {
            "Care/Harm": ["Fairness/Cheating", "Loyalty/Betrayal"],
            "Fairness/Cheating": ["Care/Harm", "Liberty/Oppression"],
            "Loyalty/Betrayal": ["Care/Harm", "Sanctity/Degradation"],
            "Authority/Subversion": ["Sanctity/Degradation", "Fairness/Cheating"],
            "Sanctity/Degradation": ["Loyalty/Betrayal", "Authority/Subversion"],
            "Liberty/Oppression": ["Fairness/Cheating", "Care/Harm"]
        }
    
    def generate_project_description(self, name: str, category: str, moral_value: str) -> str:
        """
        Generate a realistic project description based on name, category, and moral value.
        
        Args:
            name: Project name
            category: Project category
            moral_value: Primary moral value
            
        Returns:
            Generated project description
        """
        # Base descriptions by category
        category_descriptions = {
            "Education": f"This project focuses on educational initiatives that provide learning opportunities and academic support for community members.",
            "Health": f"This project addresses health and wellness needs in the community, providing access to healthcare services and promoting healthy lifestyles.",
            "Environment, public health & safety": f"This project enhances public safety and environmental health through monitoring, prevention, and emergency preparedness measures.",
            "Facilities, parks & recreation": f"This project improves recreational facilities and public spaces to enhance community quality of life and provide leisure opportunities.",
            "Streets, Sidewalks & Transit": f"This project enhances transportation infrastructure and pedestrian safety to improve mobility and accessibility throughout the community.",
            "Culture & community": f"This project strengthens community bonds and preserves cultural heritage through arts, events, and community-building activities.",
            "sport": f"This project promotes physical activity and sports participation by providing equipment, facilities, and programs for athletic development."
        }
        
        # Moral value specific additions
        moral_additions = {
            "Care/Harm": " The initiative prioritizes helping vulnerable populations and preventing harm to community members.",
            "Fairness/Cheating": " The project ensures equal access to resources and promotes social justice for all residents.",
            "Loyalty/Betrayal": " This effort builds community bonds and strengthens neighborhood identity and local pride.",
            "Authority/Subversion": " The project follows established regulations and maintains proper institutional oversight.",
            "Sanctity/Degradation": " This initiative preserves cultural heritage and protects important community values and traditions.",
            "Liberty/Oppression": " The project promotes individual freedom and autonomy while opposing restrictions on community members."
        }
        
        base_description = category_descriptions.get(category, f"This project addresses {category.lower()} needs in the community.")
        moral_addition = moral_additions.get(moral_value, "")
        
        return base_description + moral_addition
    
    def _score_project_with_classifier(self, project_name: str, project_description: str) -> Dict[str, float]:
        """
        Score a project using the moral value classifier.
        
        Args:
            project_name: Name of the project
            project_description: Description of the project
            
        Returns:
            Dictionary of moral foundation scores
        """
        # Combine name and description for classification
        project_text = f"{project_name}. {project_description}"
        
        # Get moral value classification
        result = self.moral_classifier.classify_moral_foundations(project_text)
        
        # Extract scores
        if 'all_foundation_scores' in result:
            return result['all_foundation_scores']
        else:
            # Fallback to default scores if classification fails
            return {
                "Care/Harm": 0.5,
                "Fairness/Cheating": 0.5,
                "Loyalty/Betrayal": 0.5,
                "Authority/Subversion": 0.5,
                "Sanctity/Degradation": 0.5,
                "Liberty/Oppression": 0.5
            }
    
    def generate_single_project(self, project_id: int, category: str, name: str) -> Dict:
        """
        Generate a single synthetic project.
        
        Args:
            project_id: Unique project ID
            category: Project category
            name: Project name
            
        Returns:
            Dictionary containing project data
        """
        # Select primary moral value based on distribution
        primary_moral = random.choices(
            list(self.moral_distribution.keys()),
            weights=list(self.moral_distribution.values()),
            k=1
        )[0]
        
        # Select secondary moral value
        secondary_options = self.secondary_mapping.get(primary_moral, self.moral_foundations)
        secondary_moral = random.choice([mv for mv in secondary_options if mv != primary_moral])
        
        # Generate description
        description = self.generate_project_description(name, category, primary_moral)
        
        # Score the project using the moral value classifier
        print(f"  Scoring project: {name}")
        moral_scores = self._score_project_with_classifier(name, description)
        
        # Generate other attributes
        cost = random.randint(50000, 500000)  # Realistic PB project costs
        latitude = round(random.uniform(40.7, 40.8), 3)  # NYC area
        longitude = round(random.uniform(-74.0, -73.9), 3)
        votes = random.randint(20, 100)
        selected = random.choice([0, 1])
        
        # Target demographics
        targets = ["seniors", "children", "families", "adults", "youth", "community"]
        target = random.choice(targets)
        
        return {
            'project_id': project_id,
            'category': category,
            'cost': cost,
            'latitude': latitude,
            'longitude': longitude,
            'name': name,
            'description': description,
            'selected': selected,
            'target': target,
            'votes': votes,
            'moral_value': primary_moral,
            'secondary_moral_value': secondary_moral,
            'source_files': 'synthetic_balanced',
            # Use actual moral scores from classifier
            'moral_score_Care/Harm': moral_scores.get('Care/Harm', 0.0),
            'moral_score_Fairness/Cheating': moral_scores.get('Fairness/Cheating', 0.0),
            'moral_score_Loyalty/Betrayal': moral_scores.get('Loyalty/Betrayal', 0.0),
            'moral_score_Authority/Subversion': moral_scores.get('Authority/Subversion', 0.0),
            'moral_score_Sanctity/Degradation': moral_scores.get('Sanctity/Degradation', 0.0),
            'moral_score_Liberty/Oppression': moral_scores.get('Liberty/Oppression', 0.0)
        }
    
    def generate_balanced_dataset(self, projects_per_category: int = 15) -> pd.DataFrame:
        """
        Generate a balanced synthetic dataset.
        
        Args:
            projects_per_category: Number of projects per category
            
        Returns:
            DataFrame containing synthetic projects
        """
        print(f"Generating balanced synthetic dataset with {projects_per_category} projects per category...")
        
        projects = []
        project_id = 1
        
        # Generate projects for each category
        total_projects = len(self.categories) * projects_per_category
        current_project = 0
        
        for category, project_names in self.categories.items():
            print(f"  Generating {len(project_names)} projects for {category}...")
            
            # Use all available project names for this category
            for name in project_names[:projects_per_category]:
                current_project += 1
                print(f"    [{current_project}/{total_projects}] Processing: {name}")
                
                project = self.generate_single_project(project_id, category, name)
                projects.append(project)
                project_id += 1
        
        # Create DataFrame
        df = pd.DataFrame(projects)
        
        print(f"Generated {len(df)} synthetic projects")
        print(f"Categories: {df['category'].unique()}")
        print(f"Moral values: {df['moral_value'].unique()}")
        
        # Show distribution
        print("\nMoral value distribution:")
        moral_counts = df['moral_value'].value_counts()
        for moral_value, count in moral_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {moral_value}: {count} ({percentage:.1f}%)")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "balanced_synthetic_projects.csv"):
        """
        Save the synthetic dataset to CSV.
        
        Args:
            df: DataFrame containing synthetic projects
            filename: Output filename
        """
        # Ensure data directory exists
        os.makedirs('data/generated', exist_ok=True)
        
        filepath = f'data/generated/{filename}'
        df.to_csv(filepath, index=False)
        print(f"Synthetic dataset saved to: {filepath}")
        
        return filepath
    
    def generate_and_save(self, projects_per_category: int = 15, filename: str = "balanced_synthetic_projects.csv") -> str:
        """
        Generate and save a balanced synthetic dataset.
        
        Args:
            projects_per_category: Number of projects per category
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        df = self.generate_balanced_dataset(projects_per_category)
        return self.save_dataset(df, filename)
    
    def generate_liberty_oppression_data(self, num_samples: int = 200) -> List[Dict]:
        """
        Generate synthetic Liberty/Oppression data to supplement the missing MFTC data.
        
        Args:
            num_samples: Number of synthetic liberty/oppression examples to generate
            
        Returns:
            List of synthetic data dictionaries
        """
        print(f"Generating {num_samples} synthetic Liberty/Oppression examples...")
        
        # Define liberty/oppression themes and examples
        liberty_themes = {
            "Individual Rights": [
                "Freedom of speech and expression",
                "Right to privacy and personal autonomy", 
                "Individual liberty and self-determination",
                "Personal freedom and choice",
                "Civil liberties and human rights",
                "Freedom from government overreach",
                "Individual autonomy and independence",
                "Personal rights and freedoms",
                "Liberty from oppression",
                "Individual self-expression"
            ],
            "Political Freedom": [
                "Democratic participation and voting rights",
                "Freedom of assembly and protest",
                "Political liberty and representation",
                "Right to dissent and criticize",
                "Freedom from political oppression",
                "Democratic rights and freedoms",
                "Political autonomy and choice",
                "Freedom of political expression",
                "Right to political participation",
                "Liberty from authoritarian control"
            ],
            "Economic Freedom": [
                "Economic liberty and free markets",
                "Freedom to choose employment",
                "Economic independence and autonomy",
                "Right to economic opportunity",
                "Freedom from economic oppression",
                "Economic self-determination",
                "Liberty in economic choices",
                "Freedom from economic constraints",
                "Economic rights and freedoms",
                "Individual economic liberty"
            ],
            "Social Freedom": [
                "Social liberty and personal choice",
                "Freedom from social oppression",
                "Right to social self-expression",
                "Social autonomy and independence",
                "Freedom from discrimination",
                "Social rights and freedoms",
                "Liberty in social relationships",
                "Freedom from social constraints",
                "Social self-determination",
                "Personal social liberty"
            ]
        }
        
        oppression_themes = {
            "Government Oppression": [
                "Government overreach and control",
                "Authoritarian restrictions on freedom",
                "Political oppression and suppression",
                "Government violation of rights",
                "State control and oppression",
                "Political tyranny and dictatorship",
                "Government infringement on liberty",
                "Authoritarian rule and control",
                "State oppression of citizens",
                "Government restriction of freedoms"
            ],
            "Social Oppression": [
                "Social discrimination and prejudice",
                "Systemic oppression and inequality",
                "Social control and restriction",
                "Oppression of minority groups",
                "Social tyranny and domination",
                "Discrimination and social injustice",
                "Oppression of vulnerable populations",
                "Social inequality and injustice",
                "Systemic social oppression",
                "Oppression of marginalized groups"
            ],
            "Economic Oppression": [
                "Economic exploitation and control",
                "Economic inequality and oppression",
                "Workplace oppression and control",
                "Economic discrimination and injustice",
                "Financial oppression and exploitation",
                "Economic tyranny and domination",
                "Oppression of workers and employees",
                "Economic injustice and inequality",
                "Financial control and oppression",
                "Economic suppression of rights"
            ],
            "Cultural Oppression": [
                "Cultural suppression and control",
                "Oppression of cultural expression",
                "Cultural tyranny and domination",
                "Suppression of cultural identity",
                "Cultural discrimination and injustice",
                "Oppression of cultural practices",
                "Cultural control and restriction",
                "Suppression of cultural rights",
                "Cultural inequality and oppression",
                "Oppression of cultural freedom"
            ]
        }
        
        synthetic_data = []
        
        # Generate liberty examples
        liberty_count = num_samples // 2
        for i in range(liberty_count):
            theme_category = random.choice(list(liberty_themes.keys()))
            theme_examples = liberty_themes[theme_category]
            
            # Create synthetic tweet-like text
            base_text = random.choice(theme_examples)
            
            # Add variations and context
            variations = [
                f"I believe in {base_text.lower()}",
                f"We need to protect {base_text.lower()}",
                f"{base_text} is fundamental to our society",
                f"Everyone deserves {base_text.lower()}",
                f"{base_text} should be a basic right",
                f"We must defend {base_text.lower()}",
                f"{base_text} is essential for freedom",
                f"Supporting {base_text.lower()} is important",
                f"{base_text} protects individual dignity",
                f"We cannot compromise on {base_text.lower()}"
            ]
            
            tweet_text = random.choice(variations)
            
            synthetic_data.append({
                "tweet_id": f"liberty_synthetic_{i+1:06d}",
                "tweet_text": tweet_text,
                "date": "2023-01-01T00:00:00.000Z",
                "annotations": [
                    {
                        "annotator": "synthetic_generator",
                        "annotation": "liberty"
                    }
                ],
                "corpus": "Synthetic_Liberty",
                "theme_category": theme_category
            })
        
        # Generate oppression examples
        oppression_count = num_samples - liberty_count
        for i in range(oppression_count):
            theme_category = random.choice(list(oppression_themes.keys()))
            theme_examples = oppression_themes[theme_category]
            
            # Create synthetic tweet-like text
            base_text = random.choice(theme_examples)
            
            # Add variations and context
            variations = [
                f"This is {base_text.lower()}",
                f"We must fight against {base_text.lower()}",
                f"{base_text} is unacceptable",
                f"We cannot tolerate {base_text.lower()}",
                f"{base_text} violates basic rights",
                f"We must resist {base_text.lower()}",
                f"{base_text} is a threat to freedom",
                f"We need to end {base_text.lower()}",
                f"{base_text} is unjust and wrong",
                f"We must oppose {base_text.lower()}"
            ]
            
            tweet_text = random.choice(variations)
            
            synthetic_data.append({
                "tweet_id": f"oppression_synthetic_{i+1:06d}",
                "tweet_text": tweet_text,
                "date": "2023-01-01T00:00:00.000Z",
                "annotations": [
                    {
                        "annotator": "synthetic_generator", 
                        "annotation": "oppression"
                    }
                ],
                "corpus": "Synthetic_Oppression",
                "theme_category": theme_category
            })
        
        print(f"Generated {len(synthetic_data)} synthetic Liberty/Oppression examples")
        return synthetic_data
    
    def combine_with_extracted_data(self, extracted_file: str = "data/mftc_extracted_continuous.json", 
                                  output_file: str = "data/mftc_extracted_with_liberty.json",
                                  liberty_samples: int = 200) -> str:
        """
        Combine existing extracted data with synthetic Liberty/Oppression data.
        
        Args:
            extracted_file: Path to existing extracted data
            output_file: Path for combined output file
            liberty_samples: Number of synthetic liberty examples to add
            
        Returns:
            Path to combined data file
        """
        print(f"Combining extracted data with synthetic Liberty/Oppression data...")
        
        # Load existing extracted data
        if os.path.exists(extracted_file):
            with open(extracted_file, 'r') as f:
                raw_extracted_data = json.load(f)
            
            # Filter out tweets with no text
            extracted_data = []
            total_tweets_before = 0
            total_tweets_after = 0
            
            for corpus in raw_extracted_data:
                corpus_name = corpus.get('Corpus', 'Unknown')
                tweets = corpus.get('Tweets', [])
                
                # Filter tweets with valid text
                valid_tweets = [
                    tweet for tweet in tweets 
                    if tweet.get('tweet_text', '') and tweet.get('tweet_text', '') != 'no tweet text available'
                ]
                
                total_tweets_before += len(tweets)
                total_tweets_after += len(valid_tweets)
                
                extracted_data.append({
                    "Corpus": corpus_name,
                    "Tweets": valid_tweets
                })
            
            print(f"Loaded existing data: {extracted_file}")
            print(f"Filtered tweets: {total_tweets_before} → {total_tweets_after} (removed {total_tweets_before - total_tweets_after} with no text)")
        else:
            print(f"Warning: {extracted_file} not found, creating new structure")
            extracted_data = [
                {"Corpus": "ALM", "Tweets": []},
                {"Corpus": "Baltimore", "Tweets": []},
                {"Corpus": "BLM", "Tweets": []},
                {"Corpus": "Davidson", "Tweets": []},
                {"Corpus": "Election", "Tweets": []},
                {"Corpus": "MeToo", "Tweets": []},
                {"Corpus": "Sandy", "Tweets": []}
            ]
        
        # Generate synthetic liberty/oppression data
        synthetic_data = self.generate_liberty_oppression_data(liberty_samples)
        
        # Add synthetic data to extracted data structure
        synthetic_corpus = {
            "Corpus": "Synthetic_Liberty_Oppression",
            "Tweets": synthetic_data
        }
        extracted_data.append(synthetic_corpus)
        
        # Save combined data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        
        # Print statistics
        total_tweets = sum(len(corpus.get('Tweets', [])) for corpus in extracted_data)
        liberty_tweets = len(synthetic_data)
        
        print(f"Combined data saved to: {output_file}")
        print(f"Total tweets: {total_tweets}")
        print(f"Added {liberty_tweets} synthetic Liberty/Oppression examples")
        
        return output_file

def main():
    """Main function to generate synthetic data."""
    generator = SyntheticDataGenerator()
    
    # Generate balanced dataset
    filepath = generator.generate_and_save(projects_per_category=15)
    
    print(f"\nSynthetic data generation complete!")
    print(f"File saved to: {filepath}")
    
    # Load and display summary
    df = pd.read_csv(filepath)
    print(f"\nDataset Summary:")
    print(f"  Total projects: {len(df)}")
    print(f"  Categories: {len(df['category'].unique())}")
    print(f"  Moral values: {len(df['moral_value'].unique())}")
    
    print(f"\nCategory distribution:")
    for category, count in df['category'].value_counts().items():
        print(f"  {category}: {count}")
    
    print(f"\nMoral value distribution:")
    for moral_value, count in df['moral_value'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {moral_value}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
