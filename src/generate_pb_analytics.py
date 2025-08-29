"""
Generate comprehensive analytics and visualizations for Poland Warsaw PB data.
This script analyzes projects.csv and votes.csv to create various graphs and insights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PolandWarsawAnalytics:
    """Class to generate analytics and visualizations for Poland Warsaw PB data."""
    
    def __init__(self, projects_path: str = "data/parsed/projects.csv", 
                 votes_path: str = "data/parsed/votes.csv",
                 output_dir: str = "results/analysis"):
        """
        Initialize the analytics class.
        
        Args:
            projects_path: Path to the projects CSV file
            votes_path: Path to the votes CSV file
            output_dir: Directory to save generated graphs
        """
        self.projects_path = projects_path
        self.votes_path = votes_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading Poland Warsaw PB data...")
        self.projects_df = pd.read_csv(projects_path)
        self.votes_df = pd.read_csv(votes_path)
        
        print(f"Loaded {len(self.projects_df)} projects and {len(self.votes_df)} votes")
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        print("Preprocessing data...")
        
        # Clean and expand categories
        self.projects_df['categories_expanded'] = self.projects_df['category'].apply(
            lambda x: [cat.strip() for cat in str(x).split(',') if cat.strip()] if pd.notna(x) else []
        )
        
        # Create category columns for each unique category
        all_categories = set()
        for categories in self.projects_df['categories_expanded']:
            all_categories.update(categories)
        
        self.all_categories = sorted(all_categories)
        print(f"Found {len(self.all_categories)} unique categories: {', '.join(self.all_categories)}")
        
        # Create binary columns for each category
        for category in self.all_categories:
            self.projects_df[f'category_{category.replace(" ", "_").replace("&", "and")}'] = \
                self.projects_df['categories_expanded'].apply(lambda x: category in x)
        
        # Clean votes data
        self.votes_df['age'] = pd.to_numeric(self.votes_df['age'], errors='coerce')
        self.votes_df['sex'] = self.votes_df['sex'].str.upper()
        
        # Create age groups
        self.votes_df['age_group'] = pd.cut(
            self.votes_df['age'], 
            bins=[0, 18, 30, 50, 65, 100], 
            labels=['Under 18', '18-30', '31-50', '51-65', '65+'],
            include_lowest=True
        )
        
        # Clean project costs
        self.projects_df['cost'] = pd.to_numeric(self.projects_df['cost'], errors='coerce')
        
        print("Data preprocessing completed")
    
    def generate_category_analysis(self):
        """Generate category-based analysis graphs."""
        print("Generating category analysis...")
        
        # 1. Number of projects per category
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Poland Warsaw PB - Category Analysis', fontsize=16, fontweight='bold')
        
        # Projects per category
        category_counts = []
        for category in self.all_categories:
            count = self.projects_df[f'category_{category.replace(" ", "_").replace("&", "and")}'].sum()
            category_counts.append((category, count))
        
        category_counts.sort(key=lambda x: x[1], reverse=True)
        categories, counts = zip(*category_counts)
        
        axes[0, 0].barh(range(len(categories)), counts, color='skyblue')
        axes[0, 0].set_yticks(range(len(categories)))
        axes[0, 0].set_yticklabels(categories)
        axes[0, 0].set_xlabel('Number of Projects')
        axes[0, 0].set_title('Number of Projects per Category')
        axes[0, 0].invert_yaxis()
        
        # 2. Average votes per category
        category_votes = []
        for category in self.all_categories:
            mask = self.projects_df[f'category_{category.replace(" ", "_").replace("&", "and")}']
            avg_votes = self.projects_df[mask]['votes'].mean()
            category_votes.append((category, avg_votes))
        
        category_votes.sort(key=lambda x: x[1], reverse=True)
        categories_v, votes_avg = zip(*category_votes)
        
        axes[0, 1].barh(range(len(categories_v)), votes_avg, color='lightgreen')
        axes[0, 1].set_yticks(range(len(categories_v)))
        axes[0, 1].set_yticklabels(categories_v)
        axes[0, 1].set_xlabel('Average Votes')
        axes[0, 1].set_title('Average Votes per Category')
        axes[0, 1].invert_yaxis()
        
        # 3. Average cost per category
        category_costs = []
        for category in self.all_categories:
            mask = self.projects_df[f'category_{category.replace(" ", "_").replace("&", "and")}']
            avg_cost = self.projects_df[mask]['cost'].mean()
            category_costs.append((category, avg_cost))
        
        category_costs.sort(key=lambda x: x[1], reverse=True)
        categories_c, costs_avg = zip(*category_costs)
        
        axes[1, 0].barh(range(len(categories_c)), costs_avg, color='salmon')
        axes[1, 0].set_yticks(range(len(categories_c)))
        axes[1, 0].set_yticklabels(categories_c)
        axes[1, 0].set_xlabel('Average Cost (PLN)')
        axes[1, 0].set_title('Average Cost per Category')
        axes[1, 0].invert_yaxis()
        
        # 4. Project success rate by category (selected vs not selected)
        success_rates = []
        for category in self.all_categories:
            mask = self.projects_df[f'category_{category.replace(" ", "_").replace("&", "and")}']
            if mask.sum() > 0:
                success_rate = self.projects_df[mask]['selected'].mean() * 100
                success_rates.append((category, success_rate))
        
        success_rates.sort(key=lambda x: x[1], reverse=True)
        categories_s, rates = zip(*success_rates)
        
        axes[1, 1].barh(range(len(categories_s)), rates, color='gold')
        axes[1, 1].set_yticks(range(len(categories_s)))
        axes[1, 1].set_yticklabels(categories_s)
        axes[1, 1].set_xlabel('Success Rate (%)')
        axes[1, 1].set_title('Project Success Rate by Category')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Category analysis graphs saved")
    
    def generate_demographic_voting_analysis(self):
        """Generate demographic voting analysis graphs."""
        print("Generating demographic voting analysis...")
        
        # Use pandas vectorized operations for better performance
        print("  Processing votes data efficiently...")
        
        # Create a merged dataset for analysis
        # First, expand votes to individual project votes
        vote_project_pairs = []
        for _, vote in self.votes_df.iterrows():
            if pd.notna(vote['vote']):
                project_ids = [int(pid.strip()) for pid in str(vote['vote']).split(',') if pid.strip().isdigit()]
                for project_id in project_ids:
                    vote_project_pairs.append({
                        'voter_id': vote['voter_id'],
                        'project_id': project_id,
                        'age': vote['age'],
                        'age_group': vote['age_group'],
                        'sex': vote['sex'],
                        'voting_method': vote['voting_method']
                    })
        
        vote_project_df = pd.DataFrame(vote_project_pairs)
        
        # Merge with projects data
        merged_df = vote_project_df.merge(
            self.projects_df[['project_id', 'categories_expanded', 'cost', 'selected']], 
            on='project_id', how='left'
        )
        
        # Expand categories for each vote
        expanded_votes = []
        for _, row in merged_df.iterrows():
            for category in row['categories_expanded']:
                expanded_votes.append({
                    'voter_id': row['voter_id'],
                    'project_id': row['project_id'],
                    'age': row['age'],
                    'age_group': row['age_group'],
                    'sex': row['sex'],
                    'voting_method': row['voting_method'],
                    'category': category,
                    'cost': row['cost'],
                    'selected': row['selected']
                })
        
        expanded_votes_df = pd.DataFrame(expanded_votes)
        
        # Generate demographic analysis graphs
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Poland Warsaw PB - Demographic Voting Analysis', fontsize=16, fontweight='bold')
        
        # 1. Average votes per category by gender
        gender_category = expanded_votes_df.groupby(['sex', 'category']).size().unstack(fill_value=0)
        
        # Use a colormap to give each category its own color
        gender_category.plot(kind='bar', ax=axes[0, 0], colormap='tab20')
        axes[0, 0].set_title('Average Votes per Category by Gender\n(All Categories)')
        axes[0, 0].set_xlabel('Gender')
        axes[0, 0].set_ylabel('Number of Votes')
        axes[0, 0].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=0)
        
        # 2. Average votes per category by age group
        age_category = expanded_votes_df.groupby(['age_group', 'category']).size().unstack(fill_value=0)
        
        age_category.plot(kind='bar', ax=axes[0, 1], colormap='viridis')
        axes[0, 1].set_title('Average Votes per Category by Age Group\n(All Categories)')
        axes[0, 1].set_xlabel('Age Group')
        axes[0, 1].set_ylabel('Number of Votes')
        axes[0, 1].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Age distribution by category
        valid_age_votes = expanded_votes_df[expanded_votes_df['age'].notna()]
        if not valid_age_votes.empty:
            # Get all categories that have sufficient data
            category_vote_counts = valid_age_votes.groupby('category').size()
            categories_with_data = category_vote_counts[category_vote_counts >= 10].index.tolist()
            
            if categories_with_data:
                # Create age distribution box plots for all categories with sufficient data
                age_data = []
                age_labels = []
                for category in categories_with_data:
                    category_ages = valid_age_votes[valid_age_votes['category'] == category]['age'].tolist()
                    if len(category_ages) >= 10:
                        age_data.append(category_ages)
                        age_labels.append(category)
                
                if age_data:
                    box_plot = axes[1, 0].boxplot(age_data, labels=age_labels, patch_artist=True)
                    # Color the boxes
                    colors = plt.cm.Set3(np.linspace(0, 1, len(age_data)))
                    for patch, color in zip(box_plot['boxes'], colors):
                        patch.set_facecolor(color)
                    
                    axes[1, 0].set_title('Age Distribution by Category\n(All Categories with Sufficient Data)')
                    axes[1, 0].set_xlabel('Category')
                    axes[1, 0].set_ylabel('Age')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Voting method distribution
        voting_method_counts = expanded_votes_df['voting_method'].value_counts()
        
        axes[1, 1].pie(voting_method_counts.values, labels=voting_method_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Distribution of Voting Methods')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographic_voting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Demographic voting analysis graphs saved")
    
    def generate_cost_analysis(self):
        """Generate cost-related analysis graphs."""
        print("Generating cost analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Poland Warsaw PB - Cost Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cost distribution by category
        category_cost_data = []
        for category in self.all_categories:
            mask = self.projects_df[f'category_{category.replace(" ", "_").replace("&", "and")}']
            costs = self.projects_df[mask]['cost'].dropna()
            if len(costs) > 0:
                category_cost_data.append((category, costs))
        
        # Box plot of costs by category (top 10 by median cost)
        category_medians = [(cat, data.median()) for cat, data in category_cost_data]
        category_medians.sort(key=lambda x: x[1], reverse=True)
        top_categories = [cat for cat, _ in category_medians[:10]]
        
        top_cost_data = [data for cat, data in category_cost_data if cat in top_categories]
        top_cost_labels = [cat for cat, data in category_cost_data if cat in top_categories]
        
        axes[0, 0].boxplot(top_cost_data, labels=top_cost_labels)
        axes[0, 0].set_title('Cost Distribution by Category (Top 10 by Median)')
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Cost (PLN)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Cost vs Votes scatter plot
        axes[0, 1].scatter(self.projects_df['cost'], self.projects_df['votes'], alpha=0.6, color='steelblue')
        axes[0, 1].set_xlabel('Cost (PLN)')
        axes[0, 1].set_ylabel('Number of Votes')
        axes[0, 1].set_title('Cost vs Votes Relationship')
        axes[0, 1].set_xscale('log')
        
        # Add trend line
        z = np.polyfit(np.log(self.projects_df['cost'].dropna()), 
                      self.projects_df['votes'].dropna(), 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.projects_df['cost'].dropna(), 
                        p(np.log(self.projects_df['cost'].dropna())), "r--", alpha=0.8)
        
        # 3. Cost distribution histogram
        axes[1, 0].hist(self.projects_df['cost'].dropna(), bins=50, color='lightgreen', alpha=0.7)
        axes[1, 0].set_xlabel('Cost (PLN)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Project Costs')
        axes[1, 0].set_xscale('log')
        
        # 4. Average cost by success status
        success_cost = self.projects_df.groupby('selected')['cost'].mean()
        success_cost.plot(kind='bar', ax=axes[1, 1], color=['lightcoral', 'lightblue'])
        axes[1, 1].set_title('Average Cost by Project Success')
        axes[1, 1].set_xlabel('Project Selected (1=Yes, 0=No)')
        axes[1, 1].set_ylabel('Average Cost (PLN)')
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Cost analysis graphs saved")
    
    def generate_temporal_analysis(self):
        """Generate temporal analysis graphs."""
        print("Generating temporal analysis...")
        
        # Extract year from source files
        self.projects_df['year'] = self.projects_df['source_files'].apply(
            lambda x: self._extract_year_from_sources(x) if pd.notna(x) else None
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Poland Warsaw PB - Temporal Analysis', fontsize=16, fontweight='bold')
        
        # 1. Projects per year
        year_counts = self.projects_df['year'].value_counts().sort_index()
        year_counts.plot(kind='line', marker='o', ax=axes[0, 0], color='steelblue', linewidth=2)
        axes[0, 0].set_title('Number of Projects per Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Projects')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average votes per year
        year_votes = self.projects_df.groupby('year')['votes'].mean()
        year_votes.plot(kind='line', marker='s', ax=axes[0, 1], color='orange', linewidth=2)
        axes[0, 1].set_title('Average Votes per Project by Year')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Average Votes')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Average cost per year
        year_cost = self.projects_df.groupby('year')['cost'].mean()
        year_cost.plot(kind='line', marker='^', ax=axes[1, 0], color='green', linewidth=2)
        axes[1, 0].set_title('Average Project Cost by Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Average Cost (PLN)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Success rate per year
        year_success = self.projects_df.groupby('year')['selected'].mean() * 100
        year_success.plot(kind='line', marker='d', ax=axes[1, 1], color='red', linewidth=2)
        axes[1, 1].set_title('Project Success Rate by Year')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Temporal analysis graphs saved")
    
    def _extract_year_from_sources(self, source_files_str: str) -> int:
        """Extract the most common year from source files."""
        if pd.isna(source_files_str):
            return None
        
        # Extract years from source file names
        years = re.findall(r'(\d{4})', source_files_str)
        if years:
            # Return the most common year
            from collections import Counter
            year_counts = Counter(years)
            return int(year_counts.most_common(1)[0][0])
        return None
    
    def generate_summary_statistics(self):
        """Generate summary statistics and save to file."""
        print("Generating summary statistics...")
        
        summary_stats = []
        summary_stats.append("POLAND WARSAW PB - SUMMARY STATISTICS")
        summary_stats.append("=" * 50)
        summary_stats.append("")
        
        # Overall statistics
        summary_stats.append("OVERALL STATISTICS:")
        summary_stats.append(f"Total Projects: {len(self.projects_df):,}")
        summary_stats.append(f"Total Votes: {len(self.votes_df):,}")
        summary_stats.append(f"Total Unique Voters: {self.votes_df['voter_id'].nunique():,}")
        summary_stats.append(f"Total Budget: {self.projects_df['cost'].sum():,.0f} PLN")
        summary_stats.append(f"Average Project Cost: {self.projects_df['cost'].mean():,.0f} PLN")
        summary_stats.append(f"Average Votes per Project: {self.projects_df['votes'].mean():.1f}")
        summary_stats.append(f"Project Success Rate: {self.projects_df['selected'].mean() * 100:.1f}%")
        summary_stats.append("")
        
        # Category statistics
        summary_stats.append("CATEGORY STATISTICS:")
        for category in self.all_categories:
            mask = self.projects_df[f'category_{category.replace(" ", "_").replace("&", "and")}']
            if mask.sum() > 0:
                category_projects = self.projects_df[mask]
                summary_stats.append(f"\n{category}:")
                summary_stats.append(f"  Projects: {len(category_projects)}")
                summary_stats.append(f"  Average Cost: {category_projects['cost'].mean():,.0f} PLN")
                summary_stats.append(f"  Average Votes: {category_projects['votes'].mean():.1f}")
                summary_stats.append(f"  Success Rate: {category_projects['selected'].mean() * 100:.1f}%")
        
        summary_stats.append("")
        
        # Demographic statistics
        summary_stats.append("DEMOGRAPHIC STATISTICS:")
        summary_stats.append(f"Average Voter Age: {self.votes_df['age'].mean():.1f} years")
        summary_stats.append(f"Gender Distribution:")
        gender_dist = self.votes_df['sex'].value_counts()
        for gender, count in gender_dist.items():
            percentage = (count / len(self.votes_df)) * 100
            summary_stats.append(f"  {gender}: {count:,} ({percentage:.1f}%)")
        
        summary_stats.append(f"Age Group Distribution:")
        age_dist = self.votes_df['age_group'].value_counts()
        for age_group, count in age_dist.items():
            percentage = (count / len(self.votes_df)) * 100
            summary_stats.append(f"  {age_group}: {count:,} ({percentage:.1f}%)")
        
        # Save summary to file
        summary_file = self.output_dir / 'summary_statistics.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_stats))
        
        print("Summary statistics saved")
    
    def generate_all_analytics(self):
        """Generate all analytics and visualizations."""
        print("Starting comprehensive analytics generation...")
        print(f"Output directory: {self.output_dir}")
        
        try:
            # Generate all analysis types
            self.generate_category_analysis()
            self.generate_demographic_voting_analysis()
            self.generate_cost_analysis()
            self.generate_temporal_analysis()
            self.generate_summary_statistics()
            
            print("\nAll analytics generated successfully!")
            print(f"Graphs saved in: {self.output_dir}")
            print("Summary statistics saved as: summary_statistics.txt")
            
        except Exception as e:
            print(f"Error generating analytics: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the analytics."""
    print("Poland Warsaw PB Analytics Generator")
    print("=" * 50)
    
    # Initialize analytics
    analytics = PolandWarsawAnalytics()
    
    # Generate all analytics
    analytics.generate_all_analytics()

if __name__ == "__main__":
    main()
