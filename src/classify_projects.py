#!/usr/bin/env python3
"""
Script to classify all projects in the datasets using various specialized
moral value classifiers (one per foundation), replicating the MoralBERT approach.

This script adds a moral foundation score column for each of the five core foundations.
For Poland datasets, it also translates Polish project names to English before classification.
"""

import pandas as pd
import os
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from deep_translator import GoogleTranslator
import time
from constants import MORAL_FOUNDATIONS_TO_USE

def translate_polish_text(text: str) -> str:
    """Translates Polish text to English using Google Translate."""
    if not text or not isinstance(text, str) or not text.strip():
        return text
    try:
        time.sleep(0.1) # Small delay to avoid API rate limiting
        return GoogleTranslator(source='pl', target='en').translate(text)
    except Exception as e:
        print(f"Translation error for text '{text[:50]}...': {e}")
        return text

def classify_projects_in_dataset(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Classifies all projects in a dataset using a separate binary model for each moral foundation.
    
    Args:
        csv_path: Path to the CSV file containing projects.
        output_path: Optional path to save the updated CSV.
        
    Returns:
        DataFrame with added moral foundation score columns.
    """
    print(f"Loading projects from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} projects.")
        if df.empty:
            print("Warning: Dataset is empty.")
            return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()

    is_poland_dataset = 'poland' in csv_path.lower()
    if is_poland_dataset and 'name_english' not in df.columns:
        print("Detected Poland dataset - enabling translation...")
        df['name_english'] = ''

    # Initialize columns for all foundations
    for foundation in MORAL_FOUNDATIONS_TO_USE:
        df[f'moral_score_{foundation}'] = 0.0
        df[f'moral_confidence_{foundation}'] = 0.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Loop through each foundation and load its specialized model ---
    for foundation in MORAL_FOUNDATIONS_TO_USE:
        model_path = f"../models/best_roberta_model_{foundation}"
        if not os.path.exists(model_path):
            print(f"Warning: Model for '{foundation}' not found at {model_path}. Skipping this foundation.")
            continue

        print(f"\n{'='*50}")
        print(f"Loading and classifying for foundation: {foundation}")
        print(f"{'='*50}")

        try:
            model = RobertaForSequenceClassification.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model for '{foundation}': {e}. Skipping.")
            continue

        for idx, row in df.iterrows():
            if idx > 0 and idx % 100 == 0:
                print(f"  ...processed {idx}/{len(df)} projects for '{foundation}'")

            project_name_english = str(row.get('name', ''))
            if is_poland_dataset and project_name_english.strip():
                if 'name_english' in df.columns and pd.notna(row.get('name_english')) and str(row.get('name_english', '')).strip():
                    project_name_english = str(row.get('name_english', ''))
                else:
                    project_name_english = translate_polish_text(project_name_english)
                    df.at[idx, 'name_english'] = project_name_english
            
            project_text = project_name_english
            if not is_poland_dataset and pd.notna(row.get('description')) and str(row.get('description', '')).strip():
                project_text += " " + str(row.get('description', ''))
            
            if not project_text.strip():
                continue

            try:
                inputs = tokenizer(project_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    score = probabilities[0][1].item()
                
                df.at[idx, f'moral_score_{foundation}'] = score
                df.at[idx, f'moral_confidence_{foundation}'] = score
            except Exception as e:
                print(f"Error classifying project {idx} for '{foundation}': {e}")
    
    # Reorder columns for consistency
    score_cols = [f'moral_score_{f}' for f in MORAL_FOUNDATIONS_TO_USE]
    conf_cols = [f'moral_confidence_{f}' for f in MORAL_FOUNDATIONS_TO_USE]
    other_cols = [col for col in df.columns if not col.startswith('moral_score_') and not col.startswith('moral_confidence_')]
    df = df[other_cols + score_cols + conf_cols]

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nSaved updated dataset with ensemble scores to {output_path}")

    return df

def main():
    """Main function to classify all datasets with the model ensemble."""
    print("=== PROJECT MORAL CLASSIFICATION SCRIPT (MoralBERT Ensemble Approach) ===")
    datasets = [
        {'name': 'Synthetic', 'input_path': '../data/balanced_synthetic_projects.csv', 'output_path': '../data/balanced_synthetic_projects_with_moral_scores.csv'},
        {'name': 'Poland Warszawa', 'input_path': '../data/poland_warszawa_projects.csv', 'output_path': '../data/poland_warszawa_projects_with_moral_scores.csv'},
        {'name': 'Worldwide', 'input_path': '../data/worldwide_mechanical_projects.csv', 'output_path': '../data/worldwide_mechanical_projects_with_moral_scores.csv'}
    ]
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset['name']} dataset")
        print(f"{'='*60}")
        
        if os.path.exists(dataset['input_path']):
            df = classify_projects_in_dataset(dataset['input_path'], dataset['output_path'])
            if not df.empty:
                print(f"\nSummary for {dataset['name']}:")
                for foundation in MORAL_FOUNDATIONS_TO_USE:
                    score_col = f'moral_score_{foundation}'
                    if score_col in df.columns:
                        print(f"  - Avg. {foundation} Score: {df[score_col].mean():.4f}")
        else:
            print(f"Input file not found: {dataset['input_path']}")
    
    print(f"\n{'='*60}\nClassification complete!\n{'='*60}")

if __name__ == "__main__":
    main()