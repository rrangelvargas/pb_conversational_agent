"""
Moral Value Classifier with Training and Fine-tuning Capabilities

This module provides both zero-shot classification and fine-tuning capabilities
for moral value classification using RoBERTa models.
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
warnings.filterwarnings('ignore')

class MoralValueDataset(Dataset):
    """Dataset class for moral value classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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

class MoralValueClassifier:
    """
    Moral Value Classifier using RoBERTa with zero-shot and fine-tuning capabilities.
    """
    
    def __init__(self, model_name: str = "roberta-large-mnli"):
        """
        Initialize the moral value classifier.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define moral foundations
        self.moral_foundations = [
            "Care/Harm",
            "Fairness/Cheating", 
            "Loyalty/Betrayal",
            "Authority/Subversion",
            "Sanctity/Degradation",
            "Liberty/Oppression"
        ]
        
        # Improved prompts for better zero-shot classification
        self.improved_prompts = {
            "Care/Harm": "This project directly helps vulnerable people, provides healthcare, safety, emergency services, or prevents harm to community members",
            "Fairness/Cheating": "This project addresses inequality, provides equal access to resources, ensures fair distribution of benefits, or promotes social justice",
            "Loyalty/Betrayal": "This project builds community bonds, strengthens neighborhood identity, promotes local pride, or fosters community togetherness",
            "Authority/Subversion": "This project respects established systems, follows regulations, maintains public order, or supports institutional structures",
            "Sanctity/Degradation": "This project protects sacred values, maintains cultural heritage, preserves traditions, or prevents degradation of important community assets",
            "Liberty/Oppression": "This project promotes individual freedom, autonomy, personal rights, or opposes restrictions on community members"
        }
        
        # Calibration factors (original values with improved prompts)
        self.calibration_factors = {
            "Care/Harm": 0.9,
            "Fairness/Cheating": 1.4,
            "Loyalty/Betrayal": 1.2,
            "Authority/Subversion": 1.1,
            "Sanctity/Degradation": 1.2,
            "Liberty/Oppression": 1.3
        }
        
        # Initialize zero-shot classifier
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        print(f"Device set to use {self.device}")
    
    def classify_moral_foundations(self, text: str) -> Dict:
        """
        Classify text into moral foundations using zero-shot classification.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing classification results
        """
        try:
            # Use improved prompts for classification
            candidate_labels = list(self.improved_prompts.values())
            label_names = list(self.improved_prompts.keys())
            
            # Get classification results
            result = self.classifier(text, candidate_labels)
            
            # Extract scores
            moral_scores = {}
            for i, label in enumerate(label_names):
                moral_scores[label] = result['scores'][i]
            
            # Apply calibration factors
            calibrated_scores = {}
            for foundation, score in moral_scores.items():
                calibrated_scores[foundation] = score * self.calibration_factors[foundation]
            
            # Get dominant and secondary foundations
            sorted_scores = sorted(calibrated_scores.items(), key=lambda x: x[1], reverse=True)
            dominant_foundation = sorted_scores[0]
            secondary_foundation = sorted_scores[1]
            
            # Calculate confidence gap
            confidence_gap = dominant_foundation[1] - secondary_foundation[1]
            
            # Create analysis
            analysis = f"Dominant moral foundation: {dominant_foundation[0]} (confidence: {dominant_foundation[1]:.3f})"
            
            return {
                'dominant_foundation': dominant_foundation[0],
                'confidence': dominant_foundation[1],
                'secondary_foundation': secondary_foundation[0],
                'secondary_confidence': secondary_foundation[1],
                'confidence_gap': confidence_gap,
                'all_foundation_scores': calibrated_scores,
                'analysis': analysis,
                'raw_scores': moral_scores,
                'calibrated_scores': calibrated_scores
            }
            
        except Exception as e:
            return {
                'error': f"Classification failed: {str(e)}",
                'dominant_foundation': 'Care/Harm',
                'confidence': 0.5,
                'secondary_foundation': 'Fairness/Cheating',
                'secondary_confidence': 0.3,
                'confidence_gap': 0.2,
                'all_foundation_scores': {foundation: 0.5 for foundation in self.moral_foundations},
                'analysis': f"Error in classification: {str(e)}",
                'raw_scores': {foundation: 0.5 for foundation in self.moral_foundations},
                'calibrated_scores': {foundation: 0.5 for foundation in self.moral_foundations}
            }
    
    def load_extracted_data(self, extracted_path: str) -> Tuple[List[str], List[str]]:
        """
        Load extracted tweet data with texts and annotations.
        
        Args:
            extracted_path: Path to extracted tweets file
            
        Returns:
            Tuple of (texts, labels) lists
        """
        print(f"Loading extracted tweet data from: {extracted_path}")
        
        with open(extracted_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        # Check if this is our new processed format (list of tweets with moral_foundation)
        if isinstance(data, list) and len(data) > 0 and 'moral_foundation' in data[0]:
            print(f"  Loading processed data format ({len(data)} tweets)")
            
            for tweet in data:
                tweet_text = tweet.get('tweet_text', '')
                moral_foundation = tweet.get('moral_foundation', '')
                
                if tweet_text and moral_foundation:
                    texts.append(tweet_text)
                    labels.append(moral_foundation)
        
        else:
            # Process old format (list of corpus items)
            for corpus in data:
                corpus_name = corpus.get('Corpus', 'Unknown')
                tweets = corpus.get('Tweets', [])
                
                print(f"  Processing corpus: {corpus_name} ({len(tweets)} tweets)")
                
                for tweet in tweets:
                    # Get tweet text
                    tweet_text = tweet.get('tweet_text', '')
                    
                    # Skip if no text available
                    if not tweet_text or tweet_text == 'no tweet text available':
                        continue
                    
                    # Process annotations
                    annotations = tweet.get('annotations', [])
                    if not annotations:
                        continue
                    
                    # Extract moral foundation labels from annotations
                    tweet_labels = []
                    for annotation in annotations:
                        annotation_text = annotation.get('annotation', '')
                        if annotation_text:
                            foundations = self._parse_annotation(annotation_text)
                            tweet_labels.extend(foundations)
                    
                    if tweet_labels:
                        # Use the most common label for this tweet
                        most_common_label = max(set(tweet_labels), key=tweet_labels.count)
                        texts.append(tweet_text)
                        labels.append(most_common_label)
        
        print(f"Loaded {len(texts)} tweets with moral foundations")
        return texts, labels
    
    def _parse_annotation(self, annotation_text: str) -> List[str]:
        """
        Parse annotation text to extract moral foundations.
        
        Args:
            annotation_text: Raw annotation text (e.g., "care,purity" or "non-moral")
            
        Returns:
            List of moral foundation names found in annotation
        """
        annotation_lower = annotation_text.lower().strip()
        
        # Skip non-moral annotations
        if annotation_lower == 'non-moral':
            return []
        
        # Map MFTC annotation terms to our moral foundations
        mftc_to_foundation = {
            'care': 'Care/Harm',
            'harm': 'Care/Harm',
            'fairness': 'Fairness/Cheating',
            'cheating': 'Fairness/Cheating',
            'cheat': 'Fairness/Cheating',
            'loyalty': 'Loyalty/Betrayal',
            'betrayal': 'Loyalty/Betrayal',
            'betray': 'Loyalty/Betrayal',
            'authority': 'Authority/Subversion',
            'subversion': 'Authority/Subversion',
            'subvert': 'Authority/Subversion',
            'purity': 'Sanctity/Degradation',
            'degradation': 'Sanctity/Degradation',
            'degrade': 'Sanctity/Degradation',
            'liberty': 'Liberty/Oppression',
            'oppression': 'Liberty/Oppression',
            'oppress': 'Liberty/Oppression'
        }
        
        # Split by comma and process each term
        terms = [term.strip() for term in annotation_lower.split(',')]
        found_foundations = []
        
        for term in terms:
            if term in mftc_to_foundation:
                foundation = mftc_to_foundation[term]
                if foundation not in found_foundations:
                    found_foundations.append(foundation)
        
        return found_foundations
    
    def prepare_data(self, texts: List[str], labels: List[str], test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            texts: List of input texts
            labels: List of moral foundation labels
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_dataloader, test_dataloader)
        """
        print(f"Preparing data for training...")
        
        # Create label encoder
        unique_labels = sorted(list(set(labels)))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_labels = len(unique_labels)
        
        print(f"  Found {self.num_labels} unique labels: {unique_labels}")
        
        # Convert labels to integers
        label_ids = [self.label_encoder[label] for label in labels]
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, label_ids, test_size=test_size, random_state=42, stratify=label_ids
        )
        
        print(f"  Training samples: {len(train_texts)}")
        print(f"  Test samples: {len(test_texts)}")
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Create datasets
        train_dataset = MoralValueDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = MoralValueDataset(test_texts, test_labels, self.tokenizer)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        return train_dataloader, test_dataloader
    
    def initialize_model(self, unique_labels=None):
        """Initialize the RoBERTa model for classification."""
        print(f"Initializing RoBERTa model...")
        
        # Create proper label mappings
        if unique_labels is None:
            unique_labels = sorted(list(set(self.labels)))
        
        id2label = {str(i): label for i, label in enumerate(unique_labels)}
        label2id = {label: i for i, label in enumerate(unique_labels)}
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=self.num_labels,
            problem_type="single_label_classification",
            id2label=id2label,
            label2id=label2id
        )
        
        self.model.to(self.device)
        print(f"Model initialized with {self.num_labels} labels")
        print(f"Label mappings: {id2label}")
    
    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, 
              num_epochs: int = 3, learning_rate: float = 2e-5) -> str:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            test_dataloader: Test data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Path to saved model
        """
        print(f"Starting training...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        
        # Training arguments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/moral_value_roberta_mftc_{timestamp}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            report_to=None,
            disable_tqdm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=test_dataloader.dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        print("Training started - progress will be shown below:")
        trainer.train()
        
        # Save model
        model_path = f"{output_dir}/final_model"
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print(f"Training complete! Model saved to: {model_path}")
        return model_path
    
    def evaluate(self, test_dataloader: DataLoader) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            test_dataloader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating model...")
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Get label names
        label_names = [label for label, _ in sorted(self.label_encoder.items(), key=lambda x: x[1])]
        
        # Classification report
        report = classification_report(true_labels, predictions, target_names=label_names, output_dict=True)
        
        print(f"Evaluation Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Per-class F1 scores:")
        for label, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                print(f"    {label}: {metrics['f1-score']:.3f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels,
            'label_names': label_names
        }
    
    def load_finetuned_model(self, model_path: str):
        """
        Load a fine-tuned model for inference.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        print(f"Loading fine-tuned model from: {model_path}")
        
        # Convert Windows path to forward slashes for transformers
        model_path_normalized = model_path.replace('\\', '/')
        self.finetuned_model = RobertaForSequenceClassification.from_pretrained(model_path_normalized, local_files_only=True)
        self.finetuned_tokenizer = RobertaTokenizer.from_pretrained(model_path_normalized, local_files_only=True)
        self.finetuned_model.to(self.device)
        self.finetuned_model.eval()
        
        # Get label mapping from the model config
        if hasattr(self.finetuned_model.config, 'id2label') and self.finetuned_model.config.id2label:
            # Use the actual label names from the model config
            try:
                self.finetuned_labels = [self.finetuned_model.config.id2label[str(i)] for i in range(len(self.finetuned_model.config.id2label))]
            except (KeyError, TypeError) as e:
                # Fallback to default labels
                self.finetuned_labels = ['Authority/Subversion', 'Care/Harm', 'Fairness/Cheating', 'Loyalty/Betrayal', 'Sanctity/Degradation', 'Liberty/Oppression']
        else:
            # Fallback to default labels
            self.finetuned_labels = ['Authority/Subversion', 'Care/Harm', 'Fairness/Cheating', 'Loyalty/Betrayal', 'Sanctity/Degradation', 'Liberty/Oppression']
        
        print(f"Fine-tuned model loaded successfully")
    
    def classify_with_finetuned(self, text: str) -> Dict:
        """
        Classify text using the fine-tuned model.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing classification results
        """
        if not hasattr(self, 'finetuned_model'):
            raise ValueError("Fine-tuned model not loaded. Call load_finetuned_model() first.")
        
        # Tokenize and predict
        inputs = self.finetuned_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.finetuned_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get top 2 predictions
            top2_indices = torch.topk(probabilities, 2, dim=-1).indices[0]
            top2_probs = torch.topk(probabilities, 2, dim=-1).values[0]
            
            # Ensure we have valid indices
            if len(top2_indices) == 0:
                print("Error: No top indices found!")
                return self._create_fallback_result()
            
            # Get the actual indices
            dominant_idx = top2_indices[0].item()
            secondary_idx = top2_indices[1].item() if len(top2_indices) > 1 else dominant_idx
            
            # Ensure indices are within bounds
            if dominant_idx >= len(self.finetuned_labels):
                print(f"Error: Dominant index {dominant_idx} out of bounds for labels {len(self.finetuned_labels)}")
                return self._create_fallback_result()
            
            dominant_foundation = self.finetuned_labels[dominant_idx]
            secondary_foundation = self.finetuned_labels[secondary_idx] if secondary_idx < len(self.finetuned_labels) else dominant_foundation
            
            dominant_confidence = top2_probs[0].item()
            secondary_confidence = top2_probs[1].item() if len(top2_probs) > 1 else dominant_confidence
            confidence_gap = dominant_confidence - secondary_confidence
        
        # Create all foundation scores (for compatibility)
        all_scores = {}
        # Only use the number of classes the model actually has
        num_classes = probabilities.shape[1]
        for i in range(num_classes):
            if i < len(self.finetuned_labels):
                foundation = self.finetuned_labels[i]
                all_scores[foundation] = probabilities[0][i].item()
        
        # Create analysis text
        analysis = f"Fine-tuned model predicts '{dominant_foundation}' (confidence: {dominant_confidence:.3f})"
        
        return {
            'dominant_foundation': dominant_foundation,
            'confidence': dominant_confidence,
            'secondary_foundation': secondary_foundation,
            'secondary_confidence': secondary_confidence,
            'confidence_gap': confidence_gap,
            'all_foundation_scores': all_scores,
            'analysis': analysis,
            'raw_scores': all_scores,
            'calibrated_scores': all_scores
        }
    
    def finetune_from_extracted_data(self, extracted_path: str, num_epochs: int = 3) -> str:
        """
        Fine-tune the model using extracted tweet data.
        
        Args:
            extracted_path: Path to extracted tweets file
            num_epochs: Number of training epochs
            
        Returns:
            Path to saved fine-tuned model
        """
        print("Starting fine-tuning from extracted data...")
        
        # Load extracted data
        texts, labels = self.load_extracted_data(extracted_path)
        
        if len(texts) < 50:
            print(f"Not enough training data ({len(texts)} samples). Need at least 50.")
            return None
        
        # Prepare data
        train_dataloader, test_dataloader = self.prepare_data(texts, labels)
        
        # Initialize and train model
        self.initialize_model()
        model_path = self.train(train_dataloader, test_dataloader, num_epochs)
        
        # Evaluate on test set
        results = self.evaluate(test_dataloader)
        
        return model_path
    
    def _create_fallback_result(self) -> Dict:
        """Create a fallback result when classification fails."""
        return {
            'dominant_foundation': 'Care/Harm',
            'secondary_foundation': 'Fairness/Cheating',
            'confidence': 0.5,
            'secondary_confidence': 0.3,
            'confidence_gap': 0.2,
            'all_foundation_scores': {
                "Care/Harm": 0.5, "Fairness/Cheating": 0.3, "Loyalty/Betrayal": 0.2,
                "Authority/Subversion": 0.2, "Sanctity/Degradation": 0.2, "Liberty/Oppression": 0.2
            }
        }
    
    def create_comparison_visualizations(self, zero_shot_results: Dict, finetuned_results: Dict, 
                                       test_texts: List[str], test_labels: List[str],
                                       output_dir: str = "results/comparisons/models"):
        """
        Create comprehensive comparison visualizations between zero-shot and fine-tuned models.
        
        Args:
            zero_shot_results: Results from zero-shot classification
            finetuned_results: Results from fine-tuned model
            test_texts: Test texts used for evaluation
            test_labels: True labels for test texts
            output_dir: Directory to save visualizations
        """
        print("Creating comparison visualizations...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Accuracy Comparison
        self._plot_accuracy_comparison(zero_shot_results, finetuned_results, output_dir)
        
        # 2. Per-Class Performance Comparison
        self._plot_per_class_comparison(zero_shot_results, finetuned_results, output_dir)
        
        # 3. Confusion Matrices
        self._plot_confusion_matrices(zero_shot_results, finetuned_results, test_labels, output_dir)
        
        # 4. Confidence Distribution Comparison
        self._plot_confidence_distributions(zero_shot_results, finetuned_results, output_dir)
        
        # 5. Foundation Score Comparison
        self._plot_foundation_scores(zero_shot_results, finetuned_results, output_dir)
        
        print(f"Visualizations saved to: {output_dir}")
    
    def _plot_accuracy_comparison(self, zero_shot_results: Dict, finetuned_results: Dict, output_dir: str):
        """Plot accuracy comparison between models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['Zero-Shot', 'Fine-Tuned']
        accuracies = [zero_shot_results['accuracy'], finetuned_results['accuracy']]
        
        bars = ax.bar(models, accuracies, color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_comparison(self, zero_shot_results: Dict, finetuned_results: Dict, output_dir: str):
        """Plot per-class F1 score comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get class names and F1 scores
        zero_shot_f1 = zero_shot_results['classification_report']
        finetuned_f1 = finetuned_results['classification_report']
        
        # Get all possible classes from both reports
        all_classes = set()
        for report in [zero_shot_f1, finetuned_f1]:
            for class_name in report:
                if isinstance(report[class_name], dict) and 'f1-score' in report[class_name]:
                    all_classes.add(class_name)
        
        classes = sorted(list(all_classes))
        zero_shot_scores = []
        finetuned_scores = []
        
        for class_name in classes:
            # Get F1 score from zero-shot report
            zs_score = zero_shot_f1.get(class_name, {}).get('f1-score', 0.0)
            if isinstance(zs_score, (int, float)):
                zero_shot_scores.append(zs_score)
            else:
                zero_shot_scores.append(0.0)
            
            # Get F1 score from fine-tuned report
            ft_score = finetuned_f1.get(class_name, {}).get('f1-score', 0.0)
            if isinstance(ft_score, (int, float)):
                finetuned_scores.append(ft_score)
            else:
                finetuned_scores.append(0.0)
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, zero_shot_scores, width, label='Zero-Shot', color='skyblue')
        bars2 = ax.bar(x + width/2, finetuned_scores, width, label='Fine-Tuned', color='lightcoral')
        
        ax.set_xlabel('Moral Foundations')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, zero_shot_results: Dict, finetuned_results: Dict, 
                               test_labels: List[str], output_dir: str):
        """Plot confusion matrices for both models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get unique labels for consistent ordering
        unique_labels = sorted(list(set(test_labels)))
        
        # Zero-shot confusion matrix
        cm_zero_shot = confusion_matrix(test_labels, zero_shot_results['predictions'], labels=unique_labels)
        sns.heatmap(cm_zero_shot, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=unique_labels, yticklabels=unique_labels)
        ax1.set_title('Zero-Shot Model Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Fine-tuned confusion matrix
        cm_finetuned = confusion_matrix(test_labels, finetuned_results['predictions'], labels=unique_labels)
        sns.heatmap(cm_finetuned, annot=True, fmt='d', cmap='Reds', ax=ax2,
                   xticklabels=unique_labels, yticklabels=unique_labels)
        ax2.set_title('Fine-Tuned Model Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distributions(self, zero_shot_results: Dict, finetuned_results: Dict, output_dir: str):
        """Plot confidence score distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Zero-shot confidence distribution
        zero_shot_confidences = zero_shot_results.get('confidences', [])
        if zero_shot_confidences:
            ax1.hist(zero_shot_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Zero-Shot Model Confidence Distribution')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.axvline(np.mean(zero_shot_confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(zero_shot_confidences):.3f}')
            ax1.legend()
        
        # Fine-tuned confidence distribution
        finetuned_confidences = finetuned_results.get('confidences', [])
        if finetuned_confidences:
            ax2.hist(finetuned_confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title('Fine-Tuned Model Confidence Distribution')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(finetuned_confidences), color='red', linestyle='--',
                      label=f'Mean: {np.mean(finetuned_confidences):.3f}')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_foundation_scores(self, zero_shot_results: Dict, finetuned_results: Dict, output_dir: str):
        """Plot average foundation scores comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get foundation names
        foundations = list(self.moral_foundations)
        
        # Calculate average scores for each foundation
        zero_shot_avg_scores = []
        finetuned_avg_scores = []
        
        for foundation in foundations:
            # Zero-shot average score
            zs_scores = zero_shot_results.get('foundation_scores', {}).get(foundation, [])
            zs_avg = np.mean(zs_scores) if zs_scores else 0
            zero_shot_avg_scores.append(zs_avg)
            
            # Fine-tuned average score
            ft_scores = finetuned_results.get('foundation_scores', {}).get(foundation, [])
            ft_avg = np.mean(ft_scores) if ft_scores else 0
            finetuned_avg_scores.append(ft_avg)
        
        x = np.arange(len(foundations))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, zero_shot_avg_scores, width, label='Zero-Shot', color='skyblue')
        bars2 = ax.bar(x + width/2, finetuned_avg_scores, width, label='Fine-Tuned', color='lightcoral')
        
        ax.set_xlabel('Moral Foundations')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Foundation Scores Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(foundations, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/foundation_scores_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_complete_model(self, data_file: str = "data/mftc_extracted_with_liberty.json", 
                           num_epochs: int = 3) -> str:
        """
        Train a complete model with all 6 moral foundations including Liberty/Oppression.
        
        Args:
            data_file: Path to the combined data file
            num_epochs: Number of training epochs
            
        Returns:
            Path to the trained model
        """
        print("🚀 TRAINING COMPLETE MODEL WITH ALL 6 MORAL FOUNDATIONS")
        print("=" * 60)
        
        # Load data
        texts, labels = self.load_extracted_data(data_file)
        
        if len(texts) < 50:
            print(f"Not enough training data ({len(texts)} samples). Need at least 50.")
            return None
        
        print(f"Loaded {len(texts)} training examples")
        
        # Show label distribution
        label_counts = Counter(labels)
        print("Label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(labels)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Prepare data
        train_dataloader, test_dataloader = self.prepare_data(texts, labels)
        
        # Get unique labels for model initialization
        unique_labels = sorted(list(set(labels)))
        
        # Initialize and train model
        self.initialize_model(unique_labels)
        model_path = self.train(train_dataloader, test_dataloader, num_epochs)
        
        # Evaluate on test set
        results = self.evaluate(test_dataloader)
        
        return model_path