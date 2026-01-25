"""Model factory for instantiating concrete model implementations.

Provides a factory function `get_model()` that returns the appropriate model
instance (NeuralNetworkModel or GeneticProgrammingModel) based on configuration.
This decouples client code from concrete model classes and enables polymorphic
usage via the `BaseModel` interface.

Functions:
    get_model(selected_model, X, y, y_encoder, config, logger, model_filename) -> BaseModel:
        Factory function that instantiates and returns the selected model type.

See Also:
    src.base_model.BaseModel: Abstract base interface
    src.nn_model.NeuralNetworkModel: Neural network implementation
    src.gp_model.GeneticProgrammingModel: Genetic programming implementation
    src.model.main: Uses get_model() to obtain model instances during training
    src.xai.main: Uses get_model() to load models for explanation
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
    """Instantiate and return the selected model type.

    Factory function that creates a concrete model instance based on the
    `selected_model` parameter. Maps model names to their implementation classes
    and initializes them with the provided data and configuration.

    Args:
        selected_model (str): Model identifier. Must be one of:
            - 'NeuralNetwork': TensorFlow/Keras neural network
            - 'GeneticProgramming': Symbolic regression via genetic programming
        X (pd.DataFrame): Feature matrix (training data).
        y (pd.DataFrame | pd.Series): Target values.
        y_encoder (LabelEncoder): Encoder for categorical targets (or None for regression).
        config (dict): Configuration dictionary with data, task, loss, and training settings.
        logger (LoggerHandler): Logger instance for recording workflow messages.
        model_filename (str): Filename for saving/loading the model.

    Returns:
        BaseModel: Concrete model instance (NeuralNetworkModel or GeneticProgrammingModel).

    Raises:
        UnsupportedModelException: If `selected_model` is not 'NeuralNetwork' or
            'GeneticProgramming'.

    Notes:
        - The returned model implements the `BaseModel` interface
        - Client code can use polymorphism (e.g., `model.predict()`, `model.explain_shap()`)
        - Model type is case-sensitive; use exact strings from `src.utils.MODELS`
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
