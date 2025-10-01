#!/usr/bin/env python3
"""
MFRC Parser for Multi-Class Moral Foundation Classification

This script:
1. Loads the MFRC dataset
2. Filters to only include the specified categories: {"Care", "Authority", "Equality", "Loyalty", "Purity", "Non-Moral"}
3. Collects annotations
4. Caps the count for each category to keep dataset manageable
5. Outputs a balanced multi-label dataset
"""

import pandas as pd
import json
from collections import Counter, defaultdict
from constants import MORAL_FOUNDATIONS


def main():
    print("Loading MFRC dataset...")
    df = pd.read_csv("hf://datasets/USC-MOLA-Lab/MFRC/final_mfrc_data.csv")
    print(f"Loaded {len(df)} total examples")

    category_to_foundation = {
        "Care": "Care",
        "Authority": "Authority", 
        "Equality": "Fairness",
        "Loyalty": "Loyalty",
        "Purity": "Sanctity",
        "Non-Moral": "Non-Moral"
    }
    
    # Lowercase mapping for filtering
    lowercase_mapping = {
        "care": "Care",
        "authority": "Authority",
        "equality": "Equality", 
        "loyalty": "Loyalty",
        "purity": "Purity",
        "non-moral": "Non-Moral"
    }
    
    # Prepare annotation data
    df['annotation_lower'] = df['annotation'].str.lower()
    df['annotation_list'] = df['annotation_lower'].str.split(',')
    
    # Filter to only include examples with our target categories
    print("Filtering to target categories...")
    
    def has_target_categories(annotation_list):
        """Check if annotation contains only our target categories"""
        if not annotation_list:
            return False
        # Clean up the annotation list
        clean_annotations = [ann.strip() for ann in annotation_list]
        # Check if all annotations are in our target set
        return all(ann in lowercase_mapping for ann in clean_annotations)
    
    # Filter the dataset
    target_mask = df['annotation_list'].apply(has_target_categories)
    filtered_df = df[target_mask].copy()
    print(f"After filtering: {len(filtered_df)} examples")
    
    # Set category limits (adjust these as needed)
    category_limits = {
        "Care": 800,
        "Authority": 800, 
        "Equality": 600,  # Fewer examples available
        "Loyalty": 600,
        "Purity": 400,    # Fewest examples available
        "Non-Moral": 800
    }
    
    # Collect examples for each category
    collected_data = []
    category_counts = defaultdict(int)
    
    print("Collecting examples by category...")
    
    # Shuffle the data for randomness
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for idx, row in filtered_df.iterrows():
        if len(collected_data) >= 4000:  # Total dataset limit
            break
            
        annotation_list = [ann.strip() for ann in row['annotation_list']]
        
        # Check if we still need examples from any of the categories in this annotation
        needed_categories = []
        for ann in annotation_list:
            if ann in lowercase_mapping:
                category = lowercase_mapping[ann]
                if category_counts[category] < category_limits[category]:
                    needed_categories.append(category)
        
        if needed_categories:
            # Create multi-hot vector
            multi_hot = [0] * len(MORAL_FOUNDATIONS)
            for ann in annotation_list:
                if ann in lowercase_mapping:
                    category = lowercase_mapping[ann]
                    foundation = category_to_foundation[category]
                    if foundation in MORAL_FOUNDATIONS:
                        multi_hot[MORAL_FOUNDATIONS.index(foundation)] = 1
            
            # Add to collected data
            text = row['text'] if 'text' in row else row['tweet_text'] if 'tweet_text' in row else row['sentence']
            collected_data.append({
                "annotation": multi_hot,
                "text": text
            })
            
            # Update counts
            for category in needed_categories:
                category_counts[category] += 1
    
    # Create final dataset
    data = {"data": collected_data}
    
    # Print statistics
    print("\nFinal dataset statistics:")
    print(f"Total examples: {len(collected_data)}")
    
    # Count active labels
    foundation_counts = Counter()
    for item in collected_data:
        active_labels = [MORAL_FOUNDATIONS[i] for i, val in enumerate(item["annotation"]) if val == 1]
        for label in active_labels:
            foundation_counts[label] += 1
    
    print("\nFoundation counts:")
    for foundation in MORAL_FOUNDATIONS:
        print(f"  {foundation}: {foundation_counts[foundation]}")
    
    # Count multi-label examples
    multi_label_count = 0
    for item in collected_data:
        if sum(item["annotation"]) > 1:
            multi_label_count += 1
    
    print(f"\nMulti-label examples: {multi_label_count} ({multi_label_count/len(collected_data)*100:.1f}%)")
    
    # Save the dataset
    output_file = "../data/mfrc_multi_class_balanced.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nDataset saved to: {output_file}")

if __name__ == "__main__":
    main()