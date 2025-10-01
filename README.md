# Large Language Models in Participatory Budgeting: AI-Assisted Public Engagement

A conversational AI system that uses Moral Foundations Theory to recommend participatory budgeting projects aligned with citizens' values and preferences.

## Overview

This project develops a value-aligned conversational agent for participatory budgeting (PB) that bridges the gap between citizen values and complex policy proposals. The system employs an ensemble of specialized RoBERTa-based moral classifiers and a three-component scoring algorithm to provide personalized project recommendations.

## Key Features

- **Moral Classification Ensemble**: Five specialized binary classifiers for Care, Fairness, Loyalty, Authority, and Sanctity foundations
- **Three-Component Scoring**: Combines category matching, keyword relevance, and moral alignment for recommendations
- **Multi-Dataset Support**: Works with synthetic, Poland Warszawa, and worldwide Mechanical Turk datasets
- **Comprehensive Evaluation**: NDCG@5 and F1@k metrics with systematic weight optimization
- **Transparent Recommendations**: Shows scoring breakdown and preserves citizen agency

## Architecture

```
src/
├── conversational_agent.py      # Main ProjectRecommender class
├── classify_projects.py         # Moral classification for all projects
├── recommendation_evaluator.py  # Evaluation framework and metrics
├── train_final_models.py        # Model training pipeline
├── train_single_class.py        # Individual foundation training
├── grid_search.py              # Hyperparameter optimization
├── constants.py                # Configuration and keywords
├── pb_parser.py                # Participatory budgeting data parser
├── mfrc_parser.py              # Moral Foundations Reddit Corpus parser
├── generate_dataset_visualizations.py  # Data analysis and plots
└── utils.py                    # Utility functions
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pb_conversational_agent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Train Moral Classification Models

```bash
# Train individual foundation models
python src/train_single_class.py

# Train final ensemble
python src/train_final_models.py
```

### 2. Classify Projects with Moral Scores

```bash
# Generate moral scores for all datasets
python src/classify_projects.py
```

### 3. Run the Conversational Agent

```python
from src.conversational_agent import ProjectRecommender

# Initialize with synthetic dataset
agent = ProjectRecommender(dataset_type="synthetic")

# Generate recommendations
results = agent.generate_recommendations(
    "I want to support education and equal opportunities for everyone"
)

# Display results
for i, project in enumerate(results['recommendations'], 1):
    print(f"{i}. {project['name']} (Score: {project['final_score']:.2f})")
```

### 4. Evaluate System Performance

```bash
# Run comprehensive evaluation
python src/recommendation_evaluator.py
```

## Usage Examples

### Basic Recommendation

```python
agent = ProjectRecommender(dataset_type="poland")
results = agent.generate_recommendations(
    "Environmental protection and sustainability are important to me"
)

# Results include:
# - Project recommendations with scores
# - Moral foundation analysis
# - Category and keyword matching details
```

### Interactive Chat Interface

```python
agent = ProjectRecommender(dataset_type="worldwide")
agent.chat_interface()  # Interactive command-line interface
```

### Weight Optimization

```python
from src.recommendation_evaluator import RecommendationEvaluator

evaluator = RecommendationEvaluator()
evaluator.optimize_weights()  # Find optimal weight combinations
```

## Dataset Structure

### Input Format
Projects should be in CSV format with columns:
- `project_id`: Unique identifier
- `name`: Project name
- `description`: Project description
- `category`: Project category
- `cost`: Project cost
- `latitude`, `longitude`: Geographic coordinates
- `votes`: Number of votes received
- `selected`: Whether project was selected (0/1)

### Output Format
After moral classification, additional columns are added:
- `moral_score_Care`, `moral_score_Fairness`, etc.: Probability scores for each foundation
- `moral_confidence_*`: Confidence scores (same as moral scores)

## Three-Component Scoring System

The recommendation algorithm combines three components:

1. **Category Match Score (C)**: Binary score (0 or 1) based on project category alignment
2. **Normalized Keyword Score (K)**: Ratio of matched keywords to total user keywords
3. **Moral Alignment Score (M)**: Dot product of project and user moral foundation scores

**Final Score**: `(C × Wc) + (K × Wk) + (M × Wm)`

Where `Wc`, `Wk`, `Wm` are configurable weights (default: 5.0, 3.0, 2.0).

## Evaluation Results

### Model Performance
- **RoBERTa Ensemble**: F1-scores up to 0.879 across moral foundations
- **Best Performance**: Fairness (0.879), Loyalty (0.865), Sanctity (0.865)
- **Challenging Foundation**: Authority (0.755) due to abstract nature

### Recommendation Quality
- **Poland Dataset**: NDCG@5 = 0.900 (highest performance)
- **Synthetic Dataset**: NDCG@5 = 0.859
- **Worldwide Dataset**: NDCG@5 = 0.849
- **F1@5 Scores**: 0.632-0.669 across datasets

### Optimal Weight Configurations
- **Category weights**: 8.0-5.0 (dominant signal)
- **Keyword/Moral weights**: 1.0-3.0 (balanced complementary signals)

## Configuration

### Key Parameters (`src/constants.py`)

```python
# Moral foundations to classify
MORAL_FOUNDATIONS_TO_USE = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Sanctity']

# Project categories
PROJECT_CATEGORIES = ['Education', 'Environment, Public heath and Safety', 
                     'Culture and Community', 'Transportation', 'Recreation', 'Other']

# Category-specific keywords
CATEGORY_KEYWORDS = {
    'Education': ['school', 'education', 'learning', 'student', 'teacher', ...],
    'Environment, Public heath and Safety': ['health', 'safety', 'environment', ...],
    # ... other categories
}
```

### Weight Configuration

```python
# In conversational_agent.py
agent = ProjectRecommender(
    dataset_type="synthetic",
    category_weight=5.0,    # Category matching importance
    keyword_weight=3.0,     # Keyword matching importance  
    moral_weight=2.0        # Moral alignment importance
)
```

## Data Processing Pipeline

### 1. Data Parsing
```bash
# Parse participatory budgeting data
python src/pb_parser.py
```

### 2. Moral Classification
```bash
# Classify all projects with moral scores
python src/classify_projects.py
```

### 3. Visualization
```bash
# Generate dataset analysis plots
python src/generate_dataset_visualizations.py
```

## File Structure

```
pb_conversational_agent/
├── src/                          # Source code
├── data/                         # Datasets
│   ├── balanced_synthetic_projects_with_moral_scores.csv
│   ├── poland_warszawa_projects_with_moral_scores.csv
│   ├── worldwide_mechanical_projects_with_moral_scores.csv
│   └── ground_truth.json
├── models/                       # Trained models
│   ├── best_roberta_model_Care/
│   ├── best_roberta_model_Fairness/
│   └── ...
├── results/                      # Evaluation results
│   ├── evaluation/
│   ├── weight_optimization/
│   └── dataset_analysis/
├── requirements.txt
├── src_codebase.txt             # Complete source code
└── README.md
```

## Performance Optimization

### GPU Usage
- **CUDA recommended** for model training and inference
- **Memory requirements**: ~2GB GPU memory for ensemble
- **CPU fallback**: Automatic detection and fallback

### Batch Processing
- **Project classification**: Processes all projects in batches
- **Translation**: Rate-limited for Poland dataset (Google Translate)
- **Evaluation**: Parallel processing for weight optimization

## Research Methodology

This project implements a comprehensive evaluation framework:

1. **Model Architecture Comparison**: RoBERTa vs BART vs DistilBERT
2. **Hyperparameter Optimization**: Grid search for optimal configurations
3. **Weight Optimization**: Systematic testing of scoring component weights
4. **Cross-Dataset Evaluation**: Performance across synthetic and real-world data
5. **Ground Truth Validation**: Manual annotation of perfect/good/poor matches

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- Pandas, NumPy, Matplotlib, Seaborn
- Deep Translator (for Poland dataset)

## License

This project is part of academic research. Please cite appropriately if used in research.

## Citation

```
Rodrigo Rangel Vargas dos Santos. "Large Language Models in Participatory Budgeting: AI-Assisted Public Engagement." 
MSc Data Science Project, City St George's, University of London, 2025.
```