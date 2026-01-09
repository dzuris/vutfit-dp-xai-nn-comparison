"""Training model.

This module trains and saves the model.
"""
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.i_model import get_model, BaseModel
from src.utils import MODELS
from src.logging_handler import LoggerHandler
from src.exceptions import UnsupportedModelException


def train_model( # pylint: disable=too-many-arguments, too-many-positional-arguments
        X: pd.DataFrame,
        y: pd.DataFrame,
        y_encoder: LabelEncoder,
        target_column: str,
        config: dict,
        logger: LoggerHandler) -> BaseModel:
    """
    Function for training the NN and GP models.

    The functions obtains model selection, and trains model on provided dataset. Moreover
    the elapsed time for training is logged.

    Args:
        X (pd.DataFrame): Dataset Features.
        y (pd.DataFrame): Dataset Targets.
        y_encoder (LabelEncoder): Encoder for target column.
        target_column (str): Target column name.
        config (dict): Configuration for training and initializing the model.
        logger (LoggerHandler): Handler for recording logs.

    Returns:
        BaseModel: Trained model.
    """
    # Load model selection
    selected_model = config['selected_model']
    logger.add_log(f'Trained model: {selected_model}')

    if selected_model == MODELS[0]: # NN
        model_filename = f"model_nn_{target_column}.keras"
    elif selected_model == MODELS[1]: # GP
        model_filename = f"model_gp_{target_column}.pickle"
    else:
        raise UnsupportedModelException(
            f"Unsupported model provided: {selected_model},"
            f" valid options: '{MODELS[0]}' and '{MODELS[1]}'."
        )

    # Create and train the model
    model = get_model(
        selected_model=selected_model,
        X=X,
        y=y,
        y_encoder=y_encoder,
        config=config,
        logger=logger,
        model_filename=model_filename
    )

    # Measure training time
    start_time = time.time()

    # Training the model
    print(f"\nTraining model for target: {target_column}")
    model.create_and_train_model()

    # Stop measuring training time
    end_time = time.time()

    # Save model into a file
    model.save_model()

    elapsed_time = end_time - start_time
    logger.add_log(f'Time in seconds taken for training the model: {elapsed_time}')

    return model
