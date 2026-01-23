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
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"  # ensure TF ops are deterministic
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
