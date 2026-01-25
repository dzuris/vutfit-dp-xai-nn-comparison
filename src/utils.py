"""Utility functions and constants for model training and explainability.

Provides configuration loading, reproducibility setup, warning suppression, and
constants for paths, model types, task types, and loss functions. These utilities
are used across the training, XAI, and explainability pipelines.

Constants:
    TMP_FOLDER (str): Temporary directory for artifacts ("tmp").
    MODELS_FOLDER (str): Directory where trained models are saved ("models").
    EXPLANATIONS_STORE_FOLDER (str): Directory for XAI artifacts ("tmp/exp").
    MODELS (list[str]): Supported model types ("NeuralNetwork", "GeneticProgramming").
    TASK_TYPES (list[str]): Supported task types ("regression", "classification").
    LOSS_FUNCTIONS_REGRESSION (dict): Loss functions for regression tasks
        (mae, mse, r2).
    LOSS_FUNCTIONS_CLASSIFICATION (dict): Loss functions for classification tasks
        (accuracy, log_loss).

See Also:
    src.base_model: Uses constants and loss functions from this module
    src.logging_handler: May reference folder constants
"""
import warnings
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
EXPLANATIONS_STORE_FOLDER = os.path.join(TMP_FOLDER, "exp")

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
    """Set global random seeds for reproducible results.

    Configures random number generation across Python, NumPy, and TensorFlow
    to ensure deterministic behavior. Also disables non-deterministic TensorFlow
    operations and suppresses TensorFlow logging.

    Args:
        seed (int): Random seed value (default: 42).

    Side Effects:
        - Sets `PYTHONHASHSEED` environment variable
        - Sets `TF_DETERMINISTIC_OPS` to ensure TensorFlow determinism
        - Suppresses TensorFlow INFO, WARNING, and ERROR logs
        - Seeds `random`, `numpy.random`, and `tensorflow.random`

    Notes:
        - Call this function early in the pipeline (e.g., at the start of `main()`)
        - Ensures consistent model initialization and training behavior
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"  # ensure TF ops are deterministic
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ignore_warnings():
    """Suppress non-critical warnings from dependencies.

    Filters warnings from Keras, SHAP, and FutureWarning categories that
    may clutter the output without providing actionable information.

    Side Effects:
        - Suppresses Keras UserWarnings
        - Suppresses SHAP UserWarnings
        - Suppresses all FutureWarnings

    Notes:
        - Call this at the start of main pipelines (e.g., XAI, training)
        - Only suppresses non-critical warnings; critical errors are still raised
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="keras")
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")
    warnings.filterwarnings("ignore", category=FutureWarning)
