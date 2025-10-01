#!/usr/bin/env python3
"""
Dataset Visualization Generator

This script generates visualizations for dataset distributions including:
1. Category distributions for each dataset (Poland, Worldwide, Synthetic).
2. Detailed moral value distributions (Average Scores and Violin Plots).
Each graph is saved as a separate file for maximum flexibility.
Additionally, saves a CSV with the average moral scores for each dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import MORAL_FOUNDATIONS_TO_USE

def generate_category_distribution_plot(df, dataset_name, output_dir):
    """Generates and saves a category distribution bar chart for a given dataset."""
    print(f"Generating {dataset_name} category distribution...")
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    category_counts = df['category'].value_counts()

    bars = sns.barplot(
        x=category_counts.index, 
        y=category_counts.values, 
        hue=category_counts.index,
        palette='viridis', 
        alpha=0.9,
        legend=False
    )
    
    plt.title(f'{dataset_name.title()} Dataset - Project Category Distribution', fontsize=20, fontweight='bold')
    plt.xlabel('Project Categories', fontsize=16)
    plt.ylabel('Number of Projects', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (category_counts.values.max() * 0.01),
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_categories.png'), dpi=300)
    plt.close()
    print(f"-> Saved {dataset_name} category plot.")

def generate_moral_foundation_plots(df, dataset_name, output_dir, avg_scores_dict=None):
    """
    Generates and saves two plots for moral foundations:
    1. A bar chart of the average scores.
    2. A violin plot of the score distributions.
    Optionally, stores the average scores in avg_scores_dict.
    """
    print(f"Generating {dataset_name} moral foundation visualizations...")
    moral_score_cols = [f'moral_score_{f}' for f in MORAL_FOUNDATIONS_TO_USE]
    
    moral_df = df[moral_score_cols].copy()
    moral_df.columns = MORAL_FOUNDATIONS_TO_USE

    # --- 1. Average Score Bar Chart ---
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    avg_scores = moral_df.mean().sort_values(ascending=False)
    
    if avg_scores_dict is not None:
        # Store the average scores in the provided dictionary
        avg_scores_dict[dataset_name] = moral_df.mean().to_dict()
    
    bars = sns.barplot(
        x=avg_scores.index, 
        y=avg_scores.values, 
        hue=avg_scores.index,
        palette='plasma', 
        alpha=0.9,
        legend=False
    )
    
    plt.title(f'{dataset_name.title()} Dataset - Average Moral Foundation Scores', fontsize=20, fontweight='bold')
    plt.xlabel('Moral Foundation', fontsize=16)
    plt.ylabel('Average Score (Probability)', fontsize=16)
    plt.ylim(0, max(0.5, avg_scores.max() * 1.15))

    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_moral_scores_avg.png'), dpi=300)
    plt.close()
    print(f"-> Saved {dataset_name} average score bar chart.")

def generate_combined_moral_foundation_plot(avg_scores_df, output_dir):
    """Generates a combined bar chart comparing moral foundation scores across all datasets."""
    print("Generating combined moral foundation comparison plot...")
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Prepare data for grouped bar chart
    plot_data = []
    for foundation in MORAL_FOUNDATIONS_TO_USE:
        for dataset in avg_scores_df.index:
            plot_data.append({
                'Foundation': foundation,
                'Dataset': dataset,
                'Score': avg_scores_df.loc[dataset, foundation]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped bar chart
    bars = sns.barplot(
        data=plot_df,
        x='Foundation',
        y='Score',
        hue='Dataset',
        palette='viridis',
        alpha=0.9
    )
    
    plt.title('Moral Foundation Scores Comparison Across Datasets', fontsize=20, fontweight='bold')
    plt.xlabel('Moral Foundation', fontsize=16)
    plt.ylabel('Average Score (Probability)', fontsize=16)
    plt.ylim(0, max(0.5, plot_df['Score'].max() * 1.15))
    
    # Add value labels on bars
    for bar in bars.patches:
        height = bar.get_height()
        if height > 0:  # Only add labels for non-zero values
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    
    # Customize legend
    plt.legend(title='Dataset', loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_moral_foundations_comparison.png'), dpi=300)
    plt.close()
    print("-> Saved combined moral foundation comparison plot.")

def generate_original_category_distribution_plot(dataset_name: str, output_dir: str):
    """Generates and saves a category distribution bar chart for original categories from PB files."""
    print(f"Generating {dataset_name} original category distribution...")
    
    # Load original categories data
    original_path = f"../data/{dataset_name}_original_categories.csv"
    if not os.path.exists(original_path):
        print(f"Warning: {original_path} not found, skipping original category plot for {dataset_name}")
        return
    
    df = pd.read_csv(original_path)
    
    if df.empty:
        print(f"No data found in {original_path}")
        return
    
    # For Poland dataset, create a summary with unique categories and multiple category counts
    if dataset_name == 'poland_warszawa' and 'has_multiple_categories' in df.columns:
        # Get total count of projects with multiple categories
        multiple_categories_count = df['has_multiple_categories'].sum()
        
        # Get unique categories and their counts (only single category projects)
        single_category_df = df[~df['has_multiple_categories']]
        category_counts = single_category_df['original_category'].value_counts()
        
        # Create data for plotting: multiple categories + unique categories
        plot_data = []
        plot_labels = []
        
        # Add multiple categories column first
        plot_data.append(multiple_categories_count)
        plot_labels.append('Multiple Categories')
        
        # Add unique categories
        for category, count in category_counts.items():
            plot_data.append(count)
            plot_labels.append(category)
        
        # Create bar chart
        plt.figure(figsize=(16, 10))
        sns.set_style("whitegrid")
        
        # Create bars with different colors
        colors = ['orange'] + ['skyblue'] * len(category_counts)
        bars = plt.bar(range(len(plot_data)), plot_data, color=colors, alpha=0.8)
        
        plt.title(f'{dataset_name.title()} Dataset - Original Category Distribution\n(Multiple Categories + Unique Categories)', 
                 fontsize=20, fontweight='bold')
        plt.xlabel('Categories', fontsize=16)
        plt.ylabel('Number of Projects', fontsize=16)
        plt.xticks(range(len(plot_labels)), plot_labels, rotation=45, ha='right', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, plot_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(plot_data) * 0.01),
                     f'{int(value)}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='orange', alpha=0.8, label='Multiple Categories'),
                          Patch(facecolor='skyblue', alpha=0.8, label='Single Categories')]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_original_categories.png'), dpi=300)
        plt.close()
        print(f"-> Saved {dataset_name} original category plot with multiple categories + unique categories.")
        
        # Print category summary
        print(f"  Multiple categories: {multiple_categories_count} projects")
        print(f"  Found {len(category_counts)} unique single categories:")
        for category, count in category_counts.head(10).items():
            print(f"    {category}: {count}")
        if len(category_counts) > 10:
            print(f"    ... and {len(category_counts) - 10} more categories")
        
        print(f"  Projects with multiple categories: {multiple_categories_count}")
        
    else:
        # For worldwide dataset or when no multiple categories column exists
        category_counts = df['original_category'].value_counts()
        
        plt.figure(figsize=(14, 8))
        sns.set_style("whitegrid")
        
        bars = sns.barplot(
            x=category_counts.index, 
            y=category_counts.values, 
            hue=category_counts.index,
            palette='viridis', 
            alpha=0.9,
            legend=False
        )
        
        plt.title(f'{dataset_name.title()} Dataset - Original Category Distribution', fontsize=20, fontweight='bold')
        plt.xlabel('Original Project Categories', fontsize=16)
        plt.ylabel('Number of Projects', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars.patches:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (category_counts.values.max() * 0.01),
                     f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_original_categories.png'), dpi=300)
        plt.close()
        print(f"-> Saved {dataset_name} original category plot.")
        
        # Print category summary
        print(f"  Found {len(category_counts)} unique original categories:")
        for category, count in category_counts.head(10).items():
            print(f"    {category}: {count}")
        if len(category_counts) > 10:
            print(f"    ... and {len(category_counts) - 10} more categories")
    
    print(f"  Total projects: {len(df)}")

def main():
    """Main function to generate all dataset visualizations and save average moral scores as CSV."""
    print("=" * 60)
    print("DATASET VISUALIZATION GENERATOR")
    print("=" * 60)

    output_dir = "../results/dataset_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate original category distribution plots first
    print(f"\n--- Generating Original Category Distribution Plots ---")
    generate_original_category_distribution_plot('poland_warszawa', output_dir)
    generate_original_category_distribution_plot('worldwide_mechanical', output_dir)
    
    datasets = {
        'Poland': "../data/poland_warszawa_projects_with_moral_scores.csv",
        'Worldwide': "../data/worldwide_mechanical_projects_with_moral_scores.csv",
        'Synthetic': "../data/balanced_synthetic_projects_with_moral_scores.csv"
    }

    # Dictionary to collect average scores for each dataset
    avg_scores_dict = {}

    for name, path in datasets.items():
        print(f"\n--- Processing {name} Dataset ---")
        try:
            df = pd.read_csv(path)
            print(f"Loaded {len(df)} projects.")
            
            generate_category_distribution_plot(df, name, output_dir)
            generate_moral_foundation_plots(df, name, output_dir, avg_scores_dict=avg_scores_dict)
            
        except FileNotFoundError:
            print(f"Error: Dataset not found at {path}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {name}: {e}")

    # Save the average moral scores for each dataset as a CSV
    if avg_scores_dict:
        avg_scores_df = pd.DataFrame.from_dict(avg_scores_dict, orient='index')
        avg_scores_df.index.name = 'Dataset'
        avg_scores_csv_path = os.path.join(output_dir, "average_moral_scores_by_dataset.csv")
        avg_scores_df.to_csv(avg_scores_csv_path)
        print(f"\n-> Saved average moral scores for all datasets to: {avg_scores_csv_path}")
        
        # Generate combined moral foundation comparison plot
        generate_combined_moral_foundation_plot(avg_scores_df, output_dir)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Files saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()