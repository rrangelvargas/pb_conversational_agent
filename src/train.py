#!/usr/bin/env python3
"""
Training script with real training metrics and all 6 moral foundations including Liberty/Oppression
"""

import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
from transformers import TrainingArguments, Trainer, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import Dataset

# Add src to path
sys.path.append(os.path.dirname(__file__))

from moral_value_classifier import MoralValueClassifier

class MoralDataset(Dataset):
    """Custom dataset for moral foundation classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        unique_labels = sorted(list(set(labels)))
        self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
        self.id_to_label = {i: label for i, label in enumerate(unique_labels)}
        
        # Convert labels to IDs
        self.label_ids = [self.label_to_id[label] for label in labels]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_ids[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MetricsTracker:
    """Track training metrics for visualization."""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.per_class_accuracies = []
        self.epochs = []
    
    def log_metrics(self, epoch, train_loss=None, eval_loss=None, eval_accuracy=None, per_class_acc=None):
        """Log metrics for an epoch."""
        self.epochs.append(epoch)
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if eval_loss is not None:
            self.eval_losses.append(eval_loss)
        if eval_accuracy is not None:
            self.eval_accuracies.append(eval_accuracy)
        if per_class_acc is not None:
            self.per_class_accuracies.append(per_class_acc)

def compute_metrics(eval_pred):
    """Compute detailed metrics including per-class accuracy."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(predictions == labels)
    
    # Per-class accuracy
    unique_labels = np.unique(labels)
    per_class_acc = {}
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == labels[mask])
            per_class_acc[int(label)] = float(class_acc)  # Convert to Python types
    
    return {
        'accuracy': float(accuracy),
        'per_class_accuracy': per_class_acc
    }

def train_with_real_metrics():
    """Train model with real training metrics tracking."""
    
    print("🚀 TRAINING WITH REAL METRICS AND ALL 6 FOUNDATIONS")
    print("=" * 70)
    
    # Initialize classifier
    classifier = MoralValueClassifier("roberta-large-mnli")
    
    # Load improved balanced data (this should include Liberty/Oppression)
    texts, labels = classifier.load_extracted_data("data/mftc_improved_balanced.json")
    
    # Check if Liberty/Oppression is present
    unique_labels = sorted(list(set(labels)))
    print(f"Found labels: {unique_labels}")
    
    if 'Liberty/Oppression' not in unique_labels:
        print("⚠️  Liberty/Oppression not found! Loading original balanced data...")
        texts, labels = classifier.load_extracted_data("data/mftc_balanced.json")
        unique_labels = sorted(list(set(labels)))
        print(f"Updated labels: {unique_labels}")
    
    # Ensure balanced splits
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(train_texts)} examples")
    print(f"Test set: {len(test_texts)} examples")
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    train_dataset = MoralDataset(train_texts, train_labels, tokenizer)
    eval_dataset = MoralDataset(test_texts, test_labels, tokenizer)
    
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(unique_labels),
        problem_type="single_label_classification",
        id2label={str(i): label for i, label in enumerate(unique_labels)},
        label2id={label: i for i, label in enumerate(unique_labels)}
    )
    
    model.to(classifier.device)
    
    # Training arguments with more frequent evaluation
    training_args = TrainingArguments(
        output_dir="models/real_metrics_training",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=30,  # More frequent evaluation
        save_strategy="steps",
        save_steps=30,  # Must be multiple of eval_steps
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=1e-5,
        save_total_limit=2,
        report_to=None
    )
    
    # Create trainer with metrics computation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    # Track metrics manually
    metrics_tracker = MetricsTracker()
    
    # Custom training loop to capture metrics
    print("Starting training with real metrics tracking...")
    
    # Train the model
    trainer.train()
    
    # Get training history
    if hasattr(trainer, 'state') and trainer.state.log_history:
        log_history = trainer.state.log_history
        
        # Extract real metrics
        for log in log_history:
            if 'epoch' in log:
                epoch = log['epoch']
                
                # Extract metrics
                train_loss = log.get('loss')
                eval_loss = log.get('eval_loss')
                eval_accuracy = log.get('eval_accuracy')
                per_class_acc = log.get('eval_per_class_accuracy')
                
                metrics_tracker.log_metrics(epoch, train_loss, eval_loss, eval_accuracy, per_class_acc)
    
    # Save the model
    model_path = "models/real_metrics_training/final_model"
    trainer.save_model(model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Get final predictions
    predictions = []
    confidences = []
    
    print("Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(test_texts)} examples")
            
            inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][prediction].item()
            
            predictions.append(train_dataset.id_to_label[prediction])
            confidences.append(confidence)
    
    return metrics_tracker, test_labels, predictions, confidences, unique_labels, model_path

def plot_real_training_curves(metrics_tracker, test_labels, predictions, confidences, unique_labels, save_dir="results/training"):
    """Plot real training curves and results."""
    
    print(f"\n📈 CREATING REAL TRAINING CURVES")
    print("=" * 50)
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate final metrics
    correct = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)
    overall_accuracy = correct / len(test_labels)
    
    # Per-class accuracy
    per_class_acc = {}
    for label in unique_labels:
        true_mask = np.array(test_labels) == label
        pred_mask = np.array(predictions) == label
        true_count = np.sum(true_mask)
        correct_count = np.sum(true_mask & pred_mask)
        per_class_acc[label] = correct_count / true_count if true_count > 0 else 0
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Real Loss Curves
    ax1 = plt.subplot(2, 3, 1)
    if metrics_tracker.train_losses and metrics_tracker.eval_losses:
        epochs = metrics_tracker.epochs[:len(metrics_tracker.train_losses)]
        ax1.plot(epochs, metrics_tracker.train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
        
        eval_epochs = metrics_tracker.epochs[:len(metrics_tracker.eval_losses)]
        ax1.plot(eval_epochs, metrics_tracker.eval_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Real Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Real Accuracy Curve
    ax2 = plt.subplot(2, 3, 2)
    if metrics_tracker.eval_accuracies:
        acc_epochs = metrics_tracker.epochs[:len(metrics_tracker.eval_accuracies)]
        ax2.plot(acc_epochs, metrics_tracker.eval_accuracies, 'g-', label='Validation Accuracy', linewidth=2, marker='o')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Real Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Per-Class Accuracy Over Time (if available)
    ax3 = plt.subplot(2, 3, 3)
    if metrics_tracker.per_class_accuracies:
        epochs = metrics_tracker.epochs[:len(metrics_tracker.per_class_accuracies)]
        
        for i, label in enumerate(unique_labels):
            class_accs = []
            for epoch_data in metrics_tracker.per_class_accuracies:
                if isinstance(epoch_data, dict) and i in epoch_data:
                    class_accs.append(epoch_data[i])
                else:
                    class_accs.append(0.0)
            
            ax3.plot(epochs, class_accs, label=label, linewidth=2, marker='o', markersize=4)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Per-Class Accuracy Over Training', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Confusion Matrix
    ax4 = plt.subplot(2, 3, 4)
    cm = confusion_matrix(test_labels, predictions, labels=unique_labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels, ax=ax4)
    ax4.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    ax4.tick_params(axis='x', rotation=45)
    ax4.tick_params(axis='y', rotation=0)
    
    # Plot 5: Final Per-Class Accuracy
    ax5 = plt.subplot(2, 3, 5)
    labels_short = [label.split('/')[0] for label in unique_labels]
    acc_values = list(per_class_acc.values())
    
    bars = ax5.bar(range(len(unique_labels)), acc_values, color='skyblue', alpha=0.7)
    ax5.set_title('Final Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Moral Foundation')
    ax5.set_ylabel('Accuracy')
    ax5.set_xticks(range(len(unique_labels)))
    ax5.set_xticklabels(labels_short, rotation=45, ha='right')
    ax5.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, acc_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 6: Confidence Distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax6.set_xlabel('Confidence Score')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "real_training_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Real training analysis saved to: {save_dir}/real_training_analysis.png")
    
    # Create separate detailed plots
    
    # Plot 1: Detailed Loss Curves
    plt.figure(figsize=(12, 6))
    if metrics_tracker.train_losses and metrics_tracker.eval_losses:
        epochs = metrics_tracker.epochs[:len(metrics_tracker.train_losses)]
        plt.plot(epochs, metrics_tracker.train_losses, 'b-', label='Training Loss', linewidth=3, marker='o', markersize=8)
        
        eval_epochs = metrics_tracker.epochs[:len(metrics_tracker.eval_losses)]
        plt.plot(eval_epochs, metrics_tracker.eval_losses, 'r-', label='Validation Loss', linewidth=3, marker='s', markersize=8)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Real Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "real_loss_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Real loss curves saved to: {save_dir}/real_loss_curves.png")
    
    # Plot 2: Detailed Accuracy Curves
    plt.figure(figsize=(15, 8))
    
    # Overall accuracy
    if metrics_tracker.eval_accuracies:
        acc_epochs = metrics_tracker.epochs[:len(metrics_tracker.eval_accuracies)]
        plt.plot(acc_epochs, metrics_tracker.eval_accuracies, 'k--', label='Overall Accuracy', linewidth=4, alpha=0.8, marker='o', markersize=8)
    
    # Per-class accuracy
    if metrics_tracker.per_class_accuracies:
        epochs = metrics_tracker.epochs[:len(metrics_tracker.per_class_accuracies)]
        
        for i, label in enumerate(unique_labels):
            class_accs = []
            for epoch_data in metrics_tracker.per_class_accuracies:
                if isinstance(epoch_data, dict) and i in epoch_data:
                    class_accs.append(epoch_data[i])
                else:
                    class_accs.append(0.0)
            
            plt.plot(epochs, class_accs, label=label, linewidth=2, marker='o', markersize=6)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Real Per-Class Accuracy Curves Over Training', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "real_accuracy_curves_per_label.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Real accuracy curves per label saved to: {save_dir}/real_accuracy_curves_per_label.png")
    
    # Print final results
    print(f"\n📊 FINAL TRAINING RESULTS:")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Per-Class Accuracy:")
    for label, acc in per_class_acc.items():
        print(f"  {label}: {acc:.3f}")
    
    return overall_accuracy, per_class_acc

def main():
    """Main function to run training with real metrics and all 6 foundations."""
    
    print("🚀 TRAINING WITH REAL METRICS AND ALL 6 MORAL FOUNDATIONS")
    print("=" * 70)
    
    # Train with real metrics tracking
    metrics_tracker, test_labels, predictions, confidences, unique_labels, model_path = train_with_real_metrics()
    
    # Create real training curves
    overall_accuracy, per_class_acc = plot_real_training_curves(
        metrics_tracker, test_labels, predictions, confidences, unique_labels
    )
    
    print(f"\n✅ REAL METRICS TRAINING COMPLETE!")
    print(f"Model saved to: {model_path}")
    print(f"All visualizations saved to: results/training/")
    print(f"Final accuracy: {overall_accuracy:.3f}")
    print(f"Foundations: {unique_labels}")

if __name__ == "__main__":
    main()