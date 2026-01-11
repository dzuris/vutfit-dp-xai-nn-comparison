"""Interface Model module.

This module contains implementation of Neural Network Model that is subclass of BaseModel.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils import MODELS
from src.base_model import BaseModel
from src.nn_model import NeuralNetworkModel
from src.gp_model import GeneticProgrammingModel
from src.logging_handler import LoggerHandler
from src.exceptions import UnsupportedModelException

def get_model(selected_model: str, # pylint: disable=too-many-positional-arguments, too-many-arguments
              X: pd.DataFrame,
              y: pd.DataFrame,
              y_encoder: LabelEncoder,
              config: dict,
              logger: LoggerHandler,
              model_filename: str) -> BaseModel:
    """Obtain selected model.

    The function initializes Model object with all necessary arguments.

    Args:
        selected_model (str): Model selection ('NeuralNetwork' or 'GeneticProgramming').
        X (pd.DataFrame): Dataset Features.
        y (pd.DataFrame): Dataset Targets.
        y_encoder: LabelEncoder: Label encoder for target column.
        config (dict): Configuration settings.
        logger (LoggerHandler): Handler for logging messages.
        model_filename (str): File name of the model.

    Raises:
        UnsupportedModelException: If unsupported model selection si provided.

    Returns:
        BaseModel: Model object.
    """
    models = {
        MODELS[0]: NeuralNetworkModel,
        MODELS[1]: GeneticProgrammingModel
    }

    if selected_model not in models:
        raise UnsupportedModelException(
            f"Unsupported model selection: {selected_model}."
            f"Possible values: '{MODELS[0]}' or '{MODELS[1]}'.")

    return models[selected_model](X, y, y_encoder, config, logger, model_filename)
