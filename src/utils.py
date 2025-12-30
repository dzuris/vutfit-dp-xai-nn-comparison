"""
utils.py - Utility functions for data processing.

This module contains helper functions.
"""
import os
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss, accuracy_score
from exceptions import (
    UnsupportedLossException,
    UnsupportedTaskTypeException)

# Config folders
TMP_FOLDER = 'tmp'

# Configurations
MODELS_FOLDER = "models"
MODELS = ['NeuralNetwork', 'GeneticProgramming']
TASK_TYPES = ['regression', 'classification']
LOSS_FUNCTIONS_REGRESSION = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error
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
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Cannot find config file: {config_path}")

    with open(config_path, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)


def calculate_loss(y_test: dict, y_pred: dict, loss_func: str, task_type: str) -> float:
    """Calculates the aggregated loss function.

    The function controls if provided loss function is supported for selected task_type,
    and calculates averaged loss function across all the targets.

    Raises:
        UnsupportedLossException: If an unsupported loss function is provided.
        UnsupportedTaskTypeException: If an unsupported task type is provided.
    
    Args:
        y_test (dict): Real Values. Dictionary of target columns as keys and predictions
            for column as values.
        y_pred (dict): Predicted Values. Dictionary of target columns as keys and
            predictions for column as values.
        loss_func (str): Selected loss function:
            - Regression: 'mae' (Mean Absolute Error) or 'mse' (Mean Squared Error).
            - Classification: 'accuracy' or 'log_loss'.
        task_type (str): Type of task ('regression' or 'classification').

    Returns:
        float: Computed loss value accross all target columns.
    """

    if task_type == TASK_TYPES[0]:
        if loss_func not in LOSS_FUNCTIONS_REGRESSION:
            raise UnsupportedLossException(
                f'Unsupported regression loss function: {loss_func}.')
        loss_func_map = LOSS_FUNCTIONS_REGRESSION
    elif task_type == TASK_TYPES[1]:
        if loss_func not in LOSS_FUNCTIONS_CLASSIFICATION:
            raise UnsupportedLossException(
                f'Unsupported classification loss function: {loss_func}.')
        loss_func_map = LOSS_FUNCTIONS_CLASSIFICATION
    else:
        raise UnsupportedTaskTypeException(
            f"Unsupported task type: {task_type}. Use '{TASK_TYPES[0]}' or '{TASK_TYPES[1]}'.")

    total_loss = 0.0

    for target in y_test.keys():
        y_test_values = np.array(y_test[target])
        y_pred_values = np.array(y_pred[target])

        if task_type == TASK_TYPES[1]:
            if loss_func == 'log_loss':
                # Ensure y_pred contains probabilities for log_loss
                y_pred_values = np.clip(y_pred_values, 1e-15, 1 - 1e-15) # Avoid log(0)
            else:
                # Convert probability predictions to class labels if needed
                y_pred_values = np.round(y_pred_values) # Works for binary classification

        # Compute loss
        loss = loss_func_map[loss_func](y_test_values, y_pred_values)
        total_loss += loss

    return total_loss / len(y_test)
