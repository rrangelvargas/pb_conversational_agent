"""
Reprocess Projects with Final Balanced Moral Value Classifier

This script reprocesses the content.csv projects using the final balanced classifier
to get better, more realistic moral foundation scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from moral_value_classifier import MoralValueClassifier
from utils import load_csv_data, save_csv_data, print_success, print_error, print_info

warnings.filterwarnings('ignore')

class ProjectReprocessor:
    """
    Reprocesses projects using the final balanced moral value classifier.
    """
    
    def __init__(self, projects_csv_path: str = "data/generated/content.csv"):
        """
        Initialize the final reprocessor.
        
        Args:
            projects_csv_path: Path to the projects CSV file
        """
        self.projects_csv_path = projects_csv_path
        
        # Load data
        print_info("Loading projects data...")
        self.projects_df = load_csv_data(projects_csv_path)
        
        # Initialize balanced moral value classifier
        print_info("Loading balanced moral value classification model...")
        self.moral_classifier = MoralValueClassifier("moral_foundations")
        
        # Get moral foundation names
        from constants import MORAL_FOUNDATIONS
        self.moral_foundation_names = [foundation["name"] for foundation in MORAL_FOUNDATIONS.values()]
        
        print_success(f"Loaded {len(self.projects_df)} projects")
        print_info(f"Using Moral Foundations: {', '.join(self.moral_foundation_names)}")
    
    def reprocess_projects(self):
        """
        Reprocess all projects with the final balanced classifier.
        """
        print(f"\nReprocessing {len(self.projects_df)} projects with balanced classifier...")
        print("   This will take a few minutes...")
        
        # Create backup
        backup_path = self.projects_csv_path.replace('.csv', '_before_balance.csv')
        print_info(f"Creating backup at: {backup_path}")
        self.projects_df.to_csv(backup_path, index=False)
        
        # Clear existing moral scores
        for foundation_name in self.moral_foundation_names:
            score_column = f'moral_score_{foundation_name}'
            if score_column in self.projects_df.columns:
                self.projects_df[score_column] = 0.0
        
        # Clear existing moral_value column
        if 'moral_value' in self.projects_df.columns:
            self.projects_df['moral_value'] = ''
        
        # Add secondary moral value column
        if 'secondary_moral_value' not in self.projects_df.columns:
            self.projects_df['secondary_moral_value'] = ''
        
        # Process each project
        processed_count = 0
        error_count = 0
        
        for idx, (project_idx, project) in enumerate(self.projects_df.iterrows()):
            try:
                # Get project description
                description = project.get('description', '')
                if not description:
                    print(f"Warning: Project '{project.get('name', 'Unknown')}' has no description")
                    continue
                
                        # Analyze project with balanced classifier
                result = self.moral_classifier.classify_moral_foundations(description)
                        
                if "error" in result:
                    print(f"Warning: Failed to analyze project '{project.get('name', 'Unknown')}': {result['error']}")
                    error_count += 1
                    continue
                
                # Extract calibrated scores
                calibrated_scores = result.get('calibrated_scores', {})
                
                # Update moral foundation scores
                for foundation_name in self.moral_foundation_names:
                    score_column = f'moral_score_{foundation_name}'
                    score = calibrated_scores.get(foundation_name, 0.0)
                    self.projects_df.at[project_idx, score_column] = score
                
                # Update dominant moral value
                dominant_foundation = result.get('dominant_foundation', '')
                if dominant_foundation:
                    self.projects_df.at[project_idx, 'moral_value'] = dominant_foundation
                
                # Update secondary moral value
                secondary_foundation = result.get('secondary_foundation', '')
                if secondary_foundation:
                    self.projects_df.at[project_idx, 'secondary_moral_value'] = secondary_foundation
                
                processed_count += 1
                
                # Show progress
                if processed_count % 10 == 0:
                    print(f"   Processed {processed_count}/{len(self.projects_df)} projects")
        
            except Exception as e:
                error_count += 1
                print_error(f"Failed to process project '{project.get('name', 'Unknown')}': {e}")
                continue
        
        # Save results
        print_info("Saving reprocessed projects...")
        self.projects_df.to_csv(self.projects_csv_path, index=False)
        
        # Summary
        print_success(f"\nProject reprocessing completed!")
        print(f"   Total projects processed: {processed_count}")
        print(f"   Errors encountered: {error_count}")
        print(f"   Results saved to: {self.projects_csv_path}")
        print(f"   Backup saved to: {backup_path}")
    
    def validate_balance(self):
        """Validate that the final balance has been achieved."""
        print(f"\nValidating balance...")
        
        # Check if moral score columns exist
        score_columns = [f'moral_score_{foundation}' for foundation in self.moral_foundation_names]
        missing_columns = [col for col in score_columns if col not in self.projects_df.columns]
        
        if missing_columns:
            print_error(f"Missing moral score columns: {missing_columns}")
            return False
        
        # Check score ranges
        print("Score Ranges After Balance:")
        for foundation in self.moral_foundation_names:
            score_column = f'moral_score_{foundation}'
            scores = self.projects_df[score_column]
            
            print(f"   {foundation}:")
            print(f"     Range: {scores.min():.3f} - {scores.max():.3f}")
            print(f"     Mean: {scores.mean():.3f}")
            print(f"     Non-zero scores: {(scores > 0).sum()}/{len(scores)}")
        
        # Check moral_value distribution
        if 'moral_value' in self.projects_df.columns:
            moral_values = self.projects_df['moral_value'].value_counts()
            print(f"\nDominant Moral Values After Balance:")
            for value, count in moral_values.items():
                percentage = count / len(self.projects_df) * 100
                print(f"   • {value}: {count} projects ({percentage:.1f}%)")
            
            # Check balance
            max_percentage = moral_values.max() / len(self.projects_df) * 100
            min_percentage = moral_values.min() / len(self.projects_df) * 100
            
            print(f"\nBalance Analysis:")
            print(f"   • Most common: {max_percentage:.1f}%")
            print(f"   • Least common: {min_percentage:.1f}%")
            print(f"   • Ratio: {max_percentage/min_percentage:.1f}:1")
            
            if max_percentage > 40:
                print(f"   Still some imbalance: {moral_values.idxmax()} is overrepresented")
            elif max_percentage > 35:
                print(f"   Moderate imbalance: {moral_values.idxmax()} is somewhat overrepresented")
            else:
                print(f"   Good balance achieved!")
        
        # Check secondary moral values
        if 'secondary_moral_value' in self.projects_df.columns:
            secondary_values = self.projects_df['secondary_moral_value'].value_counts()
            print(f"\nSecondary Moral Values:")
            for value, count in secondary_values.items():
                if value:  # Skip empty values
                    percentage = count / len(self.projects_df) * 100
                    print(f"   • {value}: {count} projects ({percentage:.1f}%)")
        
        return True
    
    def show_sample_results(self, n: int = 5):
        """Show sample projects with their final balanced moral scores."""
        print(f"\nSample reprocessed projects:")
        print("-" * 80)
        
        # Get random sample
        sample_df = self.projects_df.sample(n=min(n, len(self.projects_df)))
        
        for _, project in sample_df.iterrows():
            print(f"\n{project['name']}")
            print(f"   Category: {project['category']}")
            print(f"   Target: {project.get('target', 'N/A')}")
            print(f"   Dominant Moral Value: {project.get('moral_value', 'N/A')}")
            if project.get('secondary_moral_value'):
                print(f"   Secondary Moral Value: {project.get('secondary_moral_value', 'N/A')}")
            print(f"   Description: {project['description'][:100]}...")
            
            # Show moral scores
            print(f"   Moral Foundation Scores:")
            for foundation in self.moral_foundation_names:
                score_column = f'moral_score_{foundation}'
                if score_column in project:
                    score = project[score_column]
                    print(f"     • {foundation}: {score:.3f}")


def main():
    """
    Main function to reprocess projects with final balance.
    """
    print("Balanced Moral Value Classification Reprocessor")
    print("=" * 65)
    
    # Initialize reprocessor
    reprocessor = ProjectReprocessor()
    
    # Reprocess projects
    reprocessor.reprocess_projects()
    
    # Validate balance
    if reprocessor.validate_balance():
        print_success("Balance validation passed!")
        
        # Show sample results
        reprocessor.show_sample_results(n=5)
    else:
        print_error("Final balance validation failed!")


if __name__ == "__main__":
    main()