"""Interface Model module.

This module contains implementation of Neural Network Model that is subclass of BaseModel.
"""
import pandas as pd
from utils import MODELS
from base_model import BaseModel
from neural_network_model import NeuralNetworkModel
from genetic_programming_model import GeneticProgrammingModel
from logging_handler import LoggerHandler
from exceptions import UnsupportedModelException

def get_model(selected_model: str, X: pd.DataFrame, y: pd.DataFrame, # pylint: disable=too-many-positional-arguments, too-many-arguments
              class_names: list[str], config: dict, logger: LoggerHandler) -> BaseModel:
    """Obtain selected model.

    The function initializes Model object with all necessary arguments.

    Args:
        selected_model (str): Model selection ('NeuralNetwork' or 'GeneticProgramming').
        X (pd.DataFrame): Dataset Features.
        y (pd.DataFrame): Dataset Targets.
        class_names (list[str]): List of unique target column values.
        config (dict): Configuration settings.
        logger (LoggerHandler): Handler for logging messages.

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

    return models[selected_model](X, y, class_names, config, logger)
