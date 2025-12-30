"""Training model.

This module trains and saves the model.
"""
import time
import pandas as pd
from i_model import get_model, BaseModel
from logging_handler import LoggerHandler


def train_model(
        X: pd.DataFrame,
        y: pd.DataFrame,
        config: dict,
        logger: LoggerHandler) -> BaseModel:
    """
    Function for training the NN or GP model.

    The functions obtains model selection, and trains model on provided datasets. Moreover
    the elapsed time for training is logged.

    Args:
        X (pd.DataFrame): Dataset Features.
        y (pd.DataFrame): Dataset Targets.
        config (dict): Configuration for training and initializing the model.
        logger (LoggerHandler): Handler for recording logs.

    Returns:
        BaseModel: Trained model.
    """
    # Load model selection
    selected_model = config['selected_model']
    logger.add_log(f'Trained model: {selected_model}')

    # Measure training time
    start_time = time.time()

    # Create and train the model
    model = get_model(selected_model, X, y, config, logger)
    model.create_and_train_model()

    # Save model into a file
    model.save_model()

    # Stop measuring training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.add_log(f'Time in seconds taken for training the model: {elapsed_time}')

    return model
