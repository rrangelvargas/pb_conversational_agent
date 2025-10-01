#!/usr/bin/env python3
"""
Final Model Training Script for Moral Foundation Ensemble

This script trains the final ensemble of specialized binary classifiers using the
best-performing architecture (RoBERTa-Base) and the optimal hyperparameters
identified from the grid search.

It iterates through each of the five core moral foundations, training and saving
a dedicated model for each one. Finally, it saves a summary of the evaluation
metrics for each trained model to a CSV file.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments, Trainer,
    RobertaForSequenceClassification, RobertaTokenizer,
    EarlyStoppingCallback
)
import warnings
import shutil
from constants import MORAL_FOUNDATIONS, MORAL_FOUNDATIONS_TO_USE

# --- OPTIMAL HYPERPARAMETERS (from Grid Search) ---
BEST_HYPERPARAMS = {
    'learning_rate': 2e-05,
    'weight_decay': 0.01,
    'dropout_rate': 0.1,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 32,
}

# --- DATASET CLASS ---
class MoralDatasetSingleLabel(Dataset):
    def __init__(self, texts, labels, tokenizer, target_foundation, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_foundation = target_foundation

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        current_labels = self.labels[idx]
        binary_label = 1.0 if self.target_foundation in current_labels else 0.0
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(binary_label, dtype=torch.long)
        }

# --- MODEL TRAINER ---
class FinalModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = RobertaForSequenceClassification
        self.tokenizer_class = RobertaTokenizer
        self.base_model = 'roberta-base'
        print(f"Final training configured for RoBERTa-Base on {self.device}")

    def load_data(self, data_path="../data/mfrc_multi_class_balanced.json"):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        abs_data_path = os.path.normpath(os.path.join(script_dir, data_path))
        with open(abs_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f).get('data', [])
        
        texts, labels = [], []
        for item in data:
            text, annotation_vector = item.get('text'), item.get('annotation')
            if text and isinstance(annotation_vector, list):
                current_labels = [MORAL_FOUNDATIONS[i] for i, val in enumerate(annotation_vector) if val == 1]
                if current_labels: texts.append(text); labels.append(current_labels)
        
        if not texts: raise ValueError("Data loading failed.")
        print(f"Loaded {len(texts)} total samples.")
        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        predictions = np.argmax(logits, axis=-1)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy, 'precision': p_w, 'recall': r_w, 'f1': f1_w}
    
    def train_and_save_model(self, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, target_foundation, hyperparameters):
        print(f"Training model for foundation: {target_foundation}")
        
        tokenizer = self.tokenizer_class.from_pretrained(self.base_model)
        model_config = self.model_class.from_pretrained(self.base_model).config
        
        model_config.num_labels = 2
        model_config.problem_type = "single_label_classification"
        model_config.attention_probs_dropout_prob = hyperparameters['dropout_rate']
        model_config.hidden_dropout_prob = hyperparameters['dropout_rate']
        
        model = self.model_class.from_pretrained(self.base_model, config=model_config, ignore_mismatched_sizes=True)
        
        train_dataset = MoralDatasetSingleLabel(train_texts, train_labels, tokenizer, target_foundation)
        val_dataset = MoralDatasetSingleLabel(val_texts, val_labels, tokenizer, target_foundation)
        test_dataset = MoralDatasetSingleLabel(test_texts, test_labels, tokenizer, target_foundation)

        training_args = TrainingArguments(
            output_dir=f'../models/training_output_{target_foundation}',
            num_train_epochs=5,
            per_device_train_batch_size=hyperparameters['per_device_train_batch_size'],
            per_device_eval_batch_size=hyperparameters['per_device_eval_batch_size'],
            learning_rate=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay'],
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
            report_to="none",
        )

        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, tokenizer=tokenizer, compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()

        # --- Evaluate on the final test set ---
        print(f"Evaluating final model for '{target_foundation}' on the test set...")
        eval_metrics = trainer.evaluate(eval_dataset=test_dataset)

        # Save the best model to the final destination
        final_model_path = f"../models/best_roberta_model_{target_foundation}"
        if os.path.exists(final_model_path):
            shutil.rmtree(final_model_path)
        
        trainer.save_model(final_model_path)
        print(f"Final model for '{target_foundation}' saved to {final_model_path}")

        if os.path.exists(training_args.output_dir):
            shutil.rmtree(training_args.output_dir)

        return eval_metrics

def main():
    """Main function to train and save the final model for each foundation."""
    print("===== FINAL MODEL ENSEMBLE TRAINING =====")
    print(f"Using RoBERTa-Base with optimal hyperparameters:")
    for key, val in BEST_HYPERPARAMS.items():
        print(f"  - {key}: {val}")
    
    # Setup directories and data
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    trainer_instance = FinalModelTrainer()
    train_texts, test_texts, train_labels, test_labels = trainer_instance.load_data()
    val_texts, final_test_texts, val_labels, final_test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42
    )

    # List to store metrics from each model
    all_model_metrics = []

    # Loop through each foundation and train a model
    for foundation in MORAL_FOUNDATIONS_TO_USE:
        print(f"\n{'='*70}")
        print(f"Preparing to train model for: {foundation.upper()}")
        print(f"{'='*70}")

        try:
            eval_metrics = trainer_instance.train_and_save_model(
                train_texts, train_labels, val_texts, val_labels, 
                final_test_texts, final_test_labels, # Pass test set for final evaluation
                foundation, BEST_HYPERPARAMS
            )
            
            # Store the metrics for this model
            metrics_summary = {
                'Foundation': foundation,
                'F1 Score': eval_metrics.get('eval_f1', 0),
                'Accuracy': eval_metrics.get('eval_accuracy', 0),
                'Precision': eval_metrics.get('eval_precision', 0),
                'Recall': eval_metrics.get('eval_recall', 0),
            }
            all_model_metrics.append(metrics_summary)

        except Exception as e:
            print(f"!!! FAILED to train model for '{foundation}': {e} !!!")
            import traceback
            traceback.print_exc()

    # --- SAVE AND DISPLAY FINAL METRICS ---
    print(f"\n{'='*70}")
    print("All final models have been trained and saved.")
    
    if all_model_metrics:
        metrics_df = pd.DataFrame(all_model_metrics)
        metrics_df = metrics_df.set_index('Foundation')
        
        print("\n--- Final Model Performance Summary ---")
        
        # --- FIX: Print the DataFrame directly to avoid the error ---
        print(metrics_df)
        
        csv_path = '../results/evaluation/final_ensemble_metrics.csv'
        metrics_df.to_csv(csv_path)
        print(f"\nMetrics summary saved to: {csv_path}")

    print("\nThe model ensemble is now ready for use in classify_projects.py and conversational_agent.py.")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()