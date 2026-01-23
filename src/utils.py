"""
utils.py - Utility functions for data processing.

This module contains helper functions.
"""
import os
import random
import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    log_loss,
    accuracy_score
)

# Config folders
TMP_FOLDER = 'tmp'
MODELS_FOLDER = "models"

# Configurations
MODELS = ['NeuralNetwork', 'GeneticProgramming']
TASK_TYPES = ['regression', 'classification']
LOSS_FUNCTIONS_REGRESSION = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'r2': r2_score
}
LOSS_FUNCTIONS_CLASSIFICATION = {
    'accuracy': accuracy_score,
    'log_loss': log_loss
}


def load_config(config_path: str) -> dict:
    """Loads configuration.

    Raises:
        FileNotFoundError: If filepath is not leading to any file.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        dict: Loaded configuration.
    """

    # Validate configuration file path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config file: {config_path}")

    # Loads yaml
    with open(config_path, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

def set_reproducibility(seed=42):
    """Set global seeds for reproducilibility"""
    # 1. Set python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. Set Python built-in random seed
    random.seed(seed)
    # 3. Set NumPy seed
    np.random.seed(seed)
    # 4. Set TensorFlow seed
    tf.random.set_seed(seed)
