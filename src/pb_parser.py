#!/usr/bin/env python3
"""
PB Parser - Extract and convert participatory budgeting data from .pb files
to CSV format compatible with the conversational agent system.

This script processes Poland Warszawa and worldwide Mechanical Turk datasets
and converts them to the same format as synthetic data for comparison.
"""

import os
import csv
import re
import random
from typing import List, Dict, Any
import pandas as pd
from constants import CATEGORY_MAPPING, STANDARDIZED_CATEGORIES


class PBParser:
    """Parser for participatory budgeting .pb files."""
    
    def __init__(self, data_dir: str = "../data/raw"):
        self.data_dir = data_dir
        self.poland_files = []
        self.worldwide_files = []
        
    def find_files(self):
        """Find all Poland Warszawa and worldwide Mechanical Turk files."""
        for filename in os.listdir(self.data_dir):
            if filename.startswith("poland_warszawa") and filename.endswith(".pb"):
                self.poland_files.append(filename)
            elif filename.startswith("worldwide_mechanical") and filename.endswith(".pb"):
                self.worldwide_files.append(filename)
        
        print(f"Found {len(self.poland_files)} Poland Warszawa files")
        print(f"Found {len(self.worldwide_files)} worldwide Mechanical Turk files")
    
    def parse_pb_file(self, filename: str) -> Dict[str, Any]:
        """Parse a single .pb file and extract metadata and projects."""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sections
        sections = content.split('\n\n')
        metadata_section = sections[0]
        
        # Parse metadata
        metadata = {}
        for line in metadata_section.split('\n')[1:]:  # Skip META header
            if ';' in line:
                key, value = line.split(';', 1)
                metadata[key] = value
        
        # Find PROJECTS section
        projects = []
        if 'PROJECTS' in content:
            projects_start = content.find('PROJECTS')
            projects_end = content.find('VOTES', projects_start)
            if projects_end == -1:
                projects_end = len(content)
            
            projects_content = content[projects_start:projects_end].strip()
            lines = projects_content.split('\n')
            
            if len(lines) > 1:
                headers = lines[1].split(';')
                
                for line in lines[2:]:
                    if line.strip():
                        values = line.split(';')
                        project = dict(zip(headers, values))
                        projects.append(project)
        
        return {
            'metadata': metadata,
            'projects': projects,
            'filename': filename
        }
    
    
    def convert_to_csv_format(self, data: Dict[str, Any], dataset_type: str) -> List[Dict[str, Any]]:
        """Convert parsed PB data to CSV format compatible with synthetic data."""
        projects = []
        
        for i, project in enumerate(data['projects']):
            project_name = project.get('name', f'Project {i+1}')
            description = project.get('description', '')
            cost = project.get('cost', '0')
            
            # Convert cost to integer
            try:
                cost_int = int(float(cost))
            except (ValueError, TypeError):
                cost_int = random.randint(50000, 500000)  # Random cost if invalid
            
            # Use existing category if available, otherwise categorize
            raw_category = project.get('category', 'Other')
            
            # Map to standardized category using dictionary lookup
            category = CATEGORY_MAPPING.get(raw_category.lower(), "Other") if raw_category else "Other"
            
            # Generate random coordinates (approximate Warszawa area)
            latitude = random.uniform(52.1, 52.4)
            longitude = random.uniform(20.8, 21.3)
            
            # Generate random votes
            votes = random.randint(10, 200)
            
            # Determine target audience
            target_options = ['families', 'youth', 'seniors', 'community', 'residents']
            target = random.choice(target_options)
            
            csv_project = {
                'project_id': i + 1,
                'category': category,
                'cost': cost_int,
                'latitude': round(latitude, 3),
                'longitude': round(longitude, 3),
                'name': project_name,
                'description': description,
                'selected': random.choice([0, 1]),
                'target': target,
                'votes': votes,
                'source_files': dataset_type,
                'moral_score_Care': 0,
                'moral_score_Fairness': 0,
                'moral_score_Loyalty': 0,
                'moral_score_Authority': 0,
                'moral_score_Sanctity': 0,
                'moral_score_Non-Moral': 0
            }
            
            projects.append(csv_project)
        
        return projects
    
    def process_poland_dataset(self) -> List[Dict[str, Any]]:
        """Process all Poland Warszawa files and convert to CSV format."""
        all_projects = []
        all_projects_complete = []  # Store complete dataset without limits
        project_id_counter = 1
        
        # Category limits: 200 per standardized category except Other
        category_limits = {cat: 200 for cat in STANDARDIZED_CATEGORIES if cat != 'Other'}
        category_limits['Other'] = float('inf')  # Other gets unlimited until we reach 1500 total
        
        # Track projects per category
        category_counts = {cat: 0 for cat in category_limits.keys()}
        
        # First pass: collect all unique projects
        unique_projects = {}  # Dictionary to store unique projects by name
        
        for filename in self.poland_files:
            print(f"Processing Poland file: {filename}")
            try:
                data = self.parse_pb_file(filename)
                projects = self.convert_to_csv_format(data, 'poland_warszawa')
                
                # Add only unique projects (by name)
                for project in projects:
                    project_name = project['name']
                    if project_name not in unique_projects:
                        unique_projects[project_name] = project
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"Found {len(unique_projects)} unique projects from Poland dataset")
        
        # Second pass: apply category limits to unique projects
        for project_name, project in unique_projects.items():
            # Add to complete dataset first
            project_complete = project.copy()
            project_complete['project_id'] = project_id_counter
            all_projects_complete.append(project_complete)
            
            category = project['category']
            
            # Check if we can add this project to limited dataset
            # For "Other" category, only check if we haven't reached 1500 total
            if category == 'Other':
                can_add = len(all_projects) < 1500
            else:
                can_add = category_counts[category] < category_limits[category]
            
            if can_add:
                # Update project IDs to be unique across all files
                project['project_id'] = project_id_counter
                project_id_counter += 1
                
                all_projects.append(project)
                category_counts[category] += 1
                
                # Stop if we've reached 1500 total projects
                if len(all_projects) >= 1500:
                    break
            else:
                project_id_counter += 1
        
        # Trim to exactly 1500 projects if we have more
        if len(all_projects) > 1500:
            all_projects = all_projects[:1500]
        
        # Print category distribution
        print(f"\nFinal Poland dataset category distribution:")
        final_counts = {}
        for project in all_projects:
            category = project['category']
            final_counts[category] = final_counts.get(category, 0) + 1
        
        for category, count in sorted(final_counts.items()):
            print(f"  {category}: {count}")
        
        print(f"Total projects (limited): {len(all_projects)}")
        print(f"Total projects (complete): {len(all_projects_complete)}")
        
        # Save complete dataset
        self.save_to_csv(all_projects_complete, "../data/poland_warszawa_projects_complete.csv")
        
        return all_projects
    
    def process_worldwide_dataset(self) -> List[Dict[str, Any]]:
        """Process all worldwide Mechanical Turk files and convert to CSV format using only unique projects."""
        all_projects = []
        unique_projects = {}  # Dictionary to store unique projects by name
        project_id_counter = 1
        
        for filename in self.worldwide_files:
            print(f"Processing worldwide file: {filename}")
            try:
                data = self.parse_pb_file(filename)
                projects = self.convert_to_csv_format(data, 'worldwide_mechanical')
                
                # Add only unique projects (by name)
                for project in projects:
                    project_name = project['name']
                    if project_name not in unique_projects:
                        project['project_id'] = project_id_counter
                        project_id_counter += 1
                        unique_projects[project_name] = project
                        all_projects.append(project)
                        
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"Found {len(unique_projects)} unique projects from worldwide dataset")
        
        # Save complete dataset (worldwide dataset is already complete, no limits applied)
        self.save_to_csv(all_projects, "../data/worldwide_mechanical_projects_complete.csv")
        
        return all_projects
    
    def extract_original_categories(self, dataset_type: str) -> List[Dict[str, Any]]:
        """Extract original categories from PB files before mapping."""
        original_data = []
        unique_projects = {}  # Dictionary to store unique projects by name
        multiple_categories_count = 0
        
        if dataset_type == 'poland':
            files = self.poland_files
        elif dataset_type == 'worldwide':
            files = self.worldwide_files
        else:
            print(f"Unknown dataset type: {dataset_type}")
            return []
        
        for filename in files:
            print(f"Extracting original categories from: {filename}")
            try:
                data = self.parse_pb_file(filename)
                for project in data['projects']:
                    project_name = project.get('name', '')
                    original_category = project.get('category', 'Unknown')
                    
                    if project_name and original_category and original_category.strip():
                        # Check if project has multiple categories (comma-separated)
                        # For worldwide dataset, categories have commas in their names, so we need to be more careful
                        if dataset_type == 'worldwide':
                            # Worldwide categories are single categories with commas in names
                            has_multiple_categories = False
                        else:
                            # Poland dataset has actual comma-separated multiple categories
                            has_multiple_categories = ',' in original_category.strip()
                            if has_multiple_categories:
                                multiple_categories_count += 1
                        
                        # Only add unique projects (by name)
                        if project_name not in unique_projects:
                            unique_projects[project_name] = {
                                'name': project_name,
                                'description': project.get('description', ''),
                                'original_category': original_category.strip(),
                                'cost': project.get('cost', 0),
                                'votes': project.get('votes', 0),
                                'selected': project.get('selected', False),
                                'source_file': filename,
                                'has_multiple_categories': has_multiple_categories
                            }
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Convert unique projects to list
        original_data = list(unique_projects.values())
        
        print(f"Found {len(original_data)} unique projects")
        print(f"Projects with multiple categories: {multiple_categories_count}")
        
        return original_data

    def save_to_csv(self, projects: List[Dict[str, Any]], filename: str):
        """Save projects to CSV file."""
        if not projects:
            print(f"No projects to save for {filename}")
            return
        
        # Get fieldnames from first project
        fieldnames = list(projects[0].keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(projects)
        
        print(f"Saved {len(projects)} projects to {filename}")
    
    def run(self):
        """Main execution method."""
        print("Starting PB Parser...")
        
        # Find files
        self.find_files()
        
        # Extract original categories for visualization
        if self.poland_files:
            print("\nExtracting original Poland categories...")
            poland_original = self.extract_original_categories('poland')
            self.save_to_csv(poland_original, "../data/poland_warszawa_original_categories.csv")
        
        if self.worldwide_files:
            print("\nExtracting original worldwide categories...")
            worldwide_original = self.extract_original_categories('worldwide')
            self.save_to_csv(worldwide_original, "../data/worldwide_mechanical_original_categories.csv")
        
        # Process Poland dataset
        if self.poland_files:
            print("\nProcessing Poland Warszawa dataset...")
            poland_projects = self.process_poland_dataset()
            self.save_to_csv(poland_projects, "../data/poland_warszawa_projects.csv")
        
        # Process worldwide dataset
        if self.worldwide_files:
            print("\nProcessing worldwide Mechanical Turk dataset...")
            worldwide_projects = self.process_worldwide_dataset()
            self.save_to_csv(worldwide_projects, "../data/worldwide_mechanical_projects.csv")
        
        print("\nPB Parser completed!")


if __name__ == "__main__":
    parser = PBParser()
    parser.run()
