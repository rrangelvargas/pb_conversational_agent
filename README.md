# Moral Value Extractor

A sophisticated system for identifying moral values in text using AI models and moral foundations theory.

## Overview

The Moral Value Extractor combines a pre-trained RoBERTa model fine-tuned on moral stories with comprehensive keyword analysis to identify moral values, ethical principles, and moral reasoning patterns in text. It's based on Moral Foundations Theory and provides detailed analysis of moral dilemmas and value conflicts.

## Features

- **AI-Powered Analysis**: Uses state-of-the-art RoBERTa models for semantic understanding
- **Moral Foundations Theory**: Implements the six core moral foundations
- **Value Detection**: Identifies 10+ general moral values
- **Dilemma Analysis**: Detects moral conflicts and provides recommendations
- **Batch Processing**: Efficiently processes multiple texts
- **Modular Design**: Clean, maintainable code structure

## Architecture

```
moral_value_extractor/
├── constants.py          # Configuration constants and mappings
├── moral_value_extractor.py  # Main model class with all functionality
├── demo.py              # Example usage and demonstrations
└── README.md            # This documentation
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:
   ```bash
   pip install torch transformers numpy pandas
   ```
3. Activate your virtual environment (if using one)

## Quick Start

```python
from moral_value_extractor import MoralValueExtractor

# Initialize the model
extractor = MoralValueExtractor()

# Extract values from text
text = "The judge showed great compassion while maintaining justice."
values = extractor.extract_values(text)

# Analyze moral dilemmas
dilemma = "A doctor must choose between saving one patient or helping many."
analysis = extractor.analyze_moral_dilemma(dilemma)
```

## Usage Examples

### Basic Value Extraction
```python
extractor = MoralValueExtractor()
values = extractor.extract_values("Be honest and kind to others.")
# Returns: {'Honesty': 0.8, 'Compassion': 0.8}
```

### Batch Processing
```python
texts = ["Text 1", "Text 2", "Text 3"]
results = extractor.batch_extract_values(texts)
```

### Moral Dilemma Analysis
```python
analysis = extractor.analyze_moral_dilemma(
    "Complex moral situation description..."
)
print(f"Primary values: {analysis['primary_values']}")
print(f"Conflicts: {analysis['potential_conflicts']}")
print(f"Recommendation: {analysis['recommendation']}")
```

## Moral Foundations

The system recognizes these core moral foundations:

1. **Care/Harm** - Promoting well-being, avoiding harm
2. **Fairness/Cheating** - Justice, equality, rights
3. **Loyalty/Betrayal** - Trust, commitment, allegiance
4. **Authority/Subversion** - Leadership, hierarchy, respect
5. **Sanctity/Degradation** - Purity, sacredness, spirituality
6. **Liberty/Oppression** - Freedom, autonomy, independence

## General Values

Additional moral values detected:

- Justice, Compassion, Honesty, Respect, Responsibility
- Courage, Integrity, Generosity, Forgiveness, Humility

## Configuration

Key constants can be modified in `constants.py`:

- `DEFAULT_MODEL_NAME`: HuggingFace model to use
- `DEFAULT_THRESHOLD`: Confidence threshold for value detection
- `DEFAULT_TOP_K`: Number of top values to return
- `MODEL_CONFIG`: Model parameters (max_length, truncation, padding)

## Running the Demo

```bash
python demo.py
```

This will:
1. Load the moral reasoning model
2. Test value extraction on sample texts
3. Demonstrate moral dilemma analysis
4. Show batch processing capabilities

## Customization

### Adding New Values
Edit `constants.py` to add new value categories and keywords:

```python
GENERAL_VALUE_KEYWORDS["NewValue"] = ["keyword1", "keyword2"]
MORAL_RECOMMENDATIONS["NewValue"] = "Your recommendation here."
```

### Modifying Thresholds
Adjust confidence thresholds in `constants.py`:

```python
ANALYSIS_THRESHOLDS = {
    "high_confidence": 0.8,    # More strict
    "medium_confidence": 0.5,  # Adjusted
    "low_confidence": 0.2      # More sensitive
}
```

## Performance

- **Model Loading**: ~1-2 minutes on first run (downloads model)
- **Inference**: ~100-500ms per text depending on length
- **Batch Processing**: Efficient parallel processing
- **Memory**: ~1.5GB GPU memory usage (CUDA recommended)

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- NumPy
- Pandas

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Support

For questions or issues, please check the documentation or create an issue in the repository.
