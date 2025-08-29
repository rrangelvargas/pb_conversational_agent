"""
Utility functions for the conversational agent project.
Common functions used across multiple scripts.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def add_src_to_path():
    """Add src directory to Python path for imports."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.append(src_dir)

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data with error handling.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise pd.errors.EmptyDataError(f"File is empty: {file_path}")
        return df
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {e}")

def save_csv_data(df: pd.DataFrame, file_path: str, index: bool = False) -> None:
    """
    Save DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to save
        file_path: Path to save file
        index: Whether to save index
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=index)
        print(f"Data saved to: {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def check_dependencies(required_packages: List[str]) -> bool:
    """
    Check if all required packages are installed.
    
    Args:
        required_packages: List of package names to check
        
    Returns:
        True if all packages are available, False otherwise
    """
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}")
        except ImportError:
            missing_packages.append(package)
            print(f"{package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    
            print("All required dependencies are available!")
    return True

def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path: str, device: Optional[torch.device] = None) -> Tuple[Optional[AutoModelForSequenceClassification], Optional[AutoTokenizer]]:
    """
    Load a pre-trained model and tokenizer.
    
    Args:
        model_path: Path to the model directory
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading fails
    """
    if device is None:
        device = get_device()
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return None, None
    
    try:
        print(f"📥 Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

def print_separator(title: str = "", length: int = 60) -> None:
    """Print a formatted separator line."""
    if title:
        print(f"\n{'='*length}")
        print(f" {title}")
        print(f"{'='*length}")
    else:
        print(f"\n{'-'*length}")

def format_currency(amount: float) -> str:
    """Format currency amount with commas and dollar sign."""
    return f"${amount:,.0f}"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def create_directory_if_not_exists(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory_path, exist_ok=True)

def get_project_root() -> str:
    """Get the project root directory (parent of src)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_directory() -> str:
    """Get the data directory path."""
    return os.path.join(get_project_root(), "data")

def get_models_directory() -> str:
    """Get the models directory path."""
    return os.path.join(get_project_root(), "models")

def get_logs_directory() -> str:
    """Get the logs directory path."""
    return os.path.join(get_project_root(), "logs")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{message}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{message}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{message}")

def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️ {message}")

def print_progress(current: int, total: int, description: str = "Progress") -> None:
    """Print a progress bar."""
    percentage = (current / total) * 100
    bar_length = 30
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"\r{description}: |{bar}| {percentage:.1f}% ({current}/{total})", end='')
    if current == total:
        print()
