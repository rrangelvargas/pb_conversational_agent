#!/usr/bin/env python3
"""
Training Script for Moral Value Classification Models

This script implements the single-label binary classification approach from the
MoralBERT study for a single, specified moral foundation. It trains RoBERTa,
BART, and DistilBERT to determine which base model architecture is best suited
for this task before proceeding with a full grid search.

It saves the final metrics to a CSV and generates comparison plots, including
learning curves for loss and accuracy.
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
    BartForSequenceClassification, BartTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    EarlyStoppingCallback
)
import matplotlib.pyplot as plt
import seaborn as sns
from constants import MORAL_FOUNDATIONS

TARGET_FOUNDATION = "Care"

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

class ModelTrainerSingleLabel:
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_configs = {
            'roberta': {'model_class': RobertaForSequenceClassification, 'tokenizer_class': RobertaTokenizer, 'base_model': 'roberta-base'},
            'bart': {'model_class': BartForSequenceClassification, 'tokenizer_class': BartTokenizer, 'base_model': 'facebook/bart-base'},
            'distilbert': {'model_class': DistilBertForSequenceClassification, 'tokenizer_class': DistilBertTokenizer, 'base_model': 'distilbert-base-uncased'}
        }
        print(f"Initializing {model_name} ({model_type}) on {self.device}")

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
                if current_labels:
                    texts.append(text)
                    labels.append(current_labels)
        
        if not texts: raise ValueError("Data loading failed: Zero valid samples found.")
        print(f"Loaded {len(texts)} total samples.")
        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        predictions = np.argmax(logits, axis=-1)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy, 'precision': p_w, 'recall': r_w, 'f1': f1_w}
    
    def train_model(self, train_texts, train_labels, val_texts, val_labels, target_foundation):
        config_info = self.model_configs[self.model_type]
        tokenizer = config_info['tokenizer_class'].from_pretrained(config_info['base_model'])
        model_config = config_info['model_class'].from_pretrained(config_info['base_model']).config
        model_config.num_labels = 2
        model_config.problem_type = "single_label_classification"
        model = config_info['model_class'].from_pretrained(config_info['base_model'], config=model_config, ignore_mismatched_sizes=True)
        
        train_dataset = MoralDatasetSingleLabel(train_texts, train_labels, tokenizer, target_foundation)
        val_dataset = MoralDatasetSingleLabel(val_texts, val_labels, tokenizer, target_foundation)
        
        training_args = TrainingArguments(
            output_dir=f'../models/temp_comparison_{self.model_type}_{target_foundation}',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            learning_rate=2e-5,
            weight_decay=0.01,
            save_total_limit=1,
            report_to="none",
        )
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, tokenizer=tokenizer, compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        return trainer, tokenizer, training_time

def create_comparison_plot(results_df, target_foundation):
    """Generates and saves a bar chart comparing model F1 scores."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    bars = sns.barplot(x=results_df.index, y='F1 Score', data=results_df, palette='viridis')
    plt.title(f"Model Bake-Off Comparison for '{target_foundation}' Foundation", fontsize=16, fontweight='bold')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Weighted F1 Score", fontsize=12)
    plt.ylim(0, 1)
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=11)
    plot_path = os.path.join('../results', f'model_comparisons/comparison_f1_{target_foundation}.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    plt.close()

def create_comparison_curves(all_histories, target_foundation):
    """Generates and saves learning curves for loss and accuracy."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # --- Loss Curve ---
    for model_name, history in all_histories.items():
        eval_logs = [log for log in history if 'eval_loss' in log]
        train_logs = [log for log in history if 'loss' in log and 'eval_loss' not in log]
        
        if eval_logs:
            plt.plot([log['epoch'] for log in eval_logs], [log['eval_loss'] for log in eval_logs], 
                     marker='o', linestyle='--', label=f'{model_name} Validation Loss')
        if train_logs:
            plt.plot([log['epoch'] for log in train_logs], [log['loss'] for log in train_logs], 
                     linestyle='-', label=f'{model_name} Training Loss')

    plt.title(f"Training & Validation Loss Curves for '{target_foundation}'", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    loss_path = os.path.join('../results', f'model_comparisons/comparison_loss_curves_{target_foundation}.png')
    plt.savefig(loss_path)
    print(f"Loss curves saved to {loss_path}")
    plt.close()

    # --- Accuracy Curve ---
    plt.figure(figsize=(12, 8))
    for model_name, history in all_histories.items():
        eval_logs = [log for log in history if 'eval_accuracy' in log]
        if eval_logs:
            plt.plot([log['epoch'] for log in eval_logs], [log['eval_accuracy'] for log in eval_logs], 
                     marker='o', label=f'{model_name} Validation Accuracy')
            
    plt.title(f"Validation Accuracy Curves for '{target_foundation}'", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    acc_path = os.path.join('../results', f'model_comparisons/comparison_accuracy_curves_{target_foundation}.png')
    plt.savefig(acc_path)
    print(f"Accuracy curves saved to {acc_path}")
    plt.close()

def main():
    print(f"===== MODEL COMPARISON FOR FOUNDATION: '{TARGET_FOUNDATION}' =====")
    os.makedirs('results', exist_ok=True)

    models_to_train = [
        ("RoBERTa-Base", "roberta"),
        ("BART-Base", "bart"),
        ("DistilBERT-Base", "distilbert"),
    ]
    all_results = {}
    all_training_histories = {}

    temp_loader = ModelTrainerSingleLabel("temp", "distilbert")
    train_texts, test_texts, train_labels, test_labels = temp_loader.load_data()
    val_texts, final_test_texts, val_labels, final_test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42
    )
    
    for model_name, model_type in models_to_train:
        print(f"\n{'='*60}\nTRAINING {model_name.upper()} for '{TARGET_FOUNDATION}'\n{'='*60}")
        
        trainer_instance = ModelTrainerSingleLabel(model_name, model_type)
        trained_trainer, tokenizer, _ = trainer_instance.train_model(
            train_texts, train_labels, val_texts, val_labels, TARGET_FOUNDATION
        )
        
        # Store training history
        all_training_histories[model_name] = trained_trainer.state.log_history
        
        print(f"\nEvaluating {model_name} on the final test set...")
        test_dataset = MoralDatasetSingleLabel(final_test_texts, final_test_labels, tokenizer, TARGET_FOUNDATION)
        eval_metrics = trained_trainer.evaluate(eval_dataset=test_dataset)
        
        all_results[model_name] = {
            'F1 Score': eval_metrics.get('eval_f1', 0),
            'Accuracy': eval_metrics.get('eval_accuracy', 0),
            'Precision': eval_metrics.get('eval_precision', 0),
            'Recall': eval_metrics.get('eval_recall', 0),
        }
        print(f"Evaluation complete for {model_name}: F1 Score = {all_results[model_name]['F1 Score']:.4f}")

    print(f"\n{'='*70}\nBAKE-OFF RESULTS FOR FOUNDATION: '{TARGET_FOUNDATION}'\n{'='*70}")
    results_df = pd.DataFrame.from_dict(all_results, orient='index')
    results_df = results_df.sort_values(by='F1 Score', ascending=False)
    
    print(results_df)
    
    csv_path = os.path.join('../results', f'model_comparisons/comparison_results_{TARGET_FOUNDATION}.csv')
    results_df.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")
    
    # Generate all plots
    create_comparison_plot(results_df, TARGET_FOUNDATION)
    create_comparison_curves(all_training_histories, TARGET_FOUNDATION)

if __name__ == '__main__':
    main()