#!/usr/bin/env python3
"""
Grid Search Script for Hyperparameter Optimization

This script performs a grid search to find the optimal hyperparameters for the
best-performing model architecture (RoBERTa-Base), as determined by the initial
comparison. It includes regularization parameters (dropout, weight decay) in the
search space to combat the overfitting identified in the initial training runs.
"""

import os
import json
import itertools
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
from datetime import datetime
from constants import MORAL_FOUNDATIONS


TARGET_FOUNDATION = "Care"
MODEL_NAME = "RoBERTa-Base"
MODEL_TYPE = "roberta"


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

# --- GRID SEARCH MODEL TRAINER ---
class ModelTrainerForGridSearch:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_class = RobertaForSequenceClassification
        self.tokenizer_class = RobertaTokenizer
        self.base_model = 'roberta-base'
        print(f"Grid search configured for RoBERTa-Base on {self.device}")

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
    
    def train_model(self, train_texts, train_labels, val_texts, val_labels, target_foundation, dropout_rate, training_args_override):
        tokenizer = self.tokenizer_class.from_pretrained(self.base_model)
        model_config = self.model_class.from_pretrained(self.base_model).config
        
        model_config.num_labels = 2
        model_config.problem_type = "single_label_classification"
        model_config.attention_probs_dropout_prob = dropout_rate
        model_config.hidden_dropout_prob = dropout_rate
        
        model = self.model_class.from_pretrained(self.base_model, config=model_config, ignore_mismatched_sizes=True)
        
        train_dataset = MoralDatasetSingleLabel(train_texts, train_labels, tokenizer, target_foundation)
        val_dataset = MoralDatasetSingleLabel(val_texts, val_labels, tokenizer, target_foundation)
        
        base_args = {
            'output_dir': f'../models/grid_search_runs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'num_train_epochs': 4,
            'per_device_eval_batch_size': 32,
            'logging_strategy': "epoch",
            'eval_strategy': "epoch",
            'save_strategy': "epoch",
            'load_best_model_at_end': True,
            'metric_for_best_model': "f1",
            'greater_is_better': True,
            'fp16': torch.cuda.is_available(),
            'save_total_limit': 1,
            'report_to': "none",
        }

        base_args.update(training_args_override)
        training_args = TrainingArguments(**base_args)

        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, tokenizer=tokenizer, compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        trainer.train()
        return trainer, tokenizer

def main():
    """Main function to run the grid search."""
    print(f"===== GRID SEARCH for {MODEL_NAME} on Foundation: '{TARGET_FOUNDATION}' =====")
    
    # --- HYPERPARAMETER GRID ---
    # Includes regularization parameters to combat overfitting.
    param_grid = {
        'learning_rate': [2e-5, 5e-5],
        'weight_decay': [0.01, 0.2],
        'dropout_rate': [0.1, 0.3]
    }

    # Setup directories and data
    results_dir = f"../results/grid_search_{TARGET_FOUNDATION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    trainer_instance = ModelTrainerForGridSearch()
    train_texts, test_texts, train_labels, test_labels = trainer_instance.load_data()
    val_texts, final_test_texts, val_labels, final_test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42
    )
    
    # Create all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    all_results = []
    print(f"\nStarting grid search with {len(param_combinations)} combinations...")
    
    for i, params in enumerate(param_combinations):
        run_name = "_".join([f"{k.split('_')[-1][0]}{v}" for k, v in params.items()])
        print(f"\n--- Starting Run {i+1}/{len(param_combinations)}: {run_name} ---")

        try:
            dropout_rate = params.pop('dropout_rate')
            
            trained_trainer, tokenizer = trainer_instance.train_model(
                train_texts, train_labels, val_texts, val_labels, 
                TARGET_FOUNDATION, dropout_rate, params
            )

            print("Evaluating model on the final test set...")
            test_dataset = MoralDatasetSingleLabel(final_test_texts, final_test_labels, tokenizer, TARGET_FOUNDATION)
            final_metrics = trained_trainer.evaluate(eval_dataset=test_dataset)
            
            run_summary = params.copy()
            run_summary['dropout_rate'] = dropout_rate
            run_summary['eval_f1'] = final_metrics.get('eval_f1', 0)
            run_summary['eval_accuracy'] = final_metrics.get('eval_accuracy', 0)
            run_summary['eval_loss'] = final_metrics.get('eval_loss', 0)
            all_results.append(run_summary)
            
        except Exception as e:
            print(f"!!! Run {run_name} failed with error: {e} !!!")
            import traceback
            traceback.print_exc()

    # --- SAVE AND DISPLAY RESULTS ---
    print(f"\n{'='*70}\nGRID SEARCH COMPLETE\n{'='*70}")
    
    if not all_results:
        print("No runs were successfully completed.")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by='eval_f1', ascending=False)
    
    csv_path = os.path.join(results_dir, 'grid_search_summary.csv')
    results_df.to_csv(csv_path, index=False)
    
    print("Top 5 Hyperparameter Combinations:")
    print(results_df.head().to_string())
    
    best_params = results_df.iloc[0].to_dict()
    print("\nBest Hyperparameters Found:")
    for key, value in best_params.items():
        if 'eval' not in key:
            print(f"   - {key}: {value}")

    print("\nUse these parameters to train your final models for each foundation.")
    print(f"Full results saved to: {csv_path}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()