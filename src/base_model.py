"""BaseModel module.

This module contains abstract class for models.
"""
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.logging_handler import LoggerHandler
from src.utils import (
    TASK_TYPES,
    MODELS_FOLDER,
    LOSS_FUNCTIONS_REGRESSION,
    LOSS_FUNCTIONS_CLASSIFICATION
)
from src.exceptions import (
    UnsupportedLossException,
    UnsupportedTaskTypeException
)


class BaseModel(ABC): # pylint: disable=too-many-instance-attributes
    """Abstract model class.

    This abstract class provides methods that needs
    to be implemented in subclasses.

    Attributes:
        y_encoder (LabelEncoder): Encoder for target column.
        target_column (str): Target column name.
        class_names (list[str]): List of unique values in the target column.
        file_path (str): Path to the model file.
        task_type (str): Type of the task the model is dealing
            with ('regression' or 'classification').
        selected_loss (str): Name of the selected loss function ('mse', 'mae' for regression;
            'accuracy', 'log_loss' for classification)
        logger (LoggerHandler): Logging handler.
        feature_names (list[str]): List of feature names (X.columns).
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.DataFrame): Training targets.
        y_test (pd.DataFrame): Testing targets.

    Methods:
        save_model():
            Saves the model to the file.
        load_model():
            Loads the model from the file.
        create_and_train_model():
            Creates and train the model on provided data, with configuration settings.
        predict() -> np.ndarray:
            Predict test data.
        get_model_loss() -> float:
            Calculates loss value on test data.
        get_model_summary():
            Summarize the trained model's attributes.
        visualize_model():
            Visualize the model.
        explain_shap():
            Explain the model using SHAP values.
        explain_lime():
            Explain the model using LIME algorithm and returns explanation for each instance.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        y_encoder: LabelEncoder,
        config: dict,
        logger: LoggerHandler,
        model_filename: str,
        folder_path: str = MODELS_FOLDER
    ):
        """Initialize a base model instance.

        This method sets y_encoder, target_column, class_names as model attributes. Obtain
        file_path, sets task_type and check if is valid. Furthermore sets other attributes such
        us selected_loss, logger and feature_names. Finally it splits X and y into training
        and testing sets.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset targets.
            y_encoder: LabelEncoder: Encoder for target column.
            config (dict): Configuration.
            logger (LoggerHandler): Logger.
            model_filename (str): Filename of model's file for saving/loading.
            folder_path (str): Path to folder where model should be saved/loaded from.
        """
        # Set y encoder
        self.y_encoder = y_encoder

        # Set target column and unique class names
        if isinstance(y, pd.Series):
            self.target_column = y.name
        elif isinstance(y, pd.DataFrame):
            if len(y.columns) == 1:
                self.target_column = y.columns[0]
        if self.y_encoder:
            self.class_names = self.y_encoder.classes_
        else:
            self.class_names = None

        # Set file where to store/load from model
        self.file_path = os.path.join(folder_path, f"{model_filename}")
        os.makedirs(folder_path, exist_ok=True) # Ensure that 'folder_path' folder exists

        # Set task type
        self.task_type = config['data']['type']
        if self.task_type not in TASK_TYPES:
            raise ValueError(
                f"Unsupported task type: {self.task_type}."
                "Task type could be 'regression' or 'classification'.")

        # Obtain other values from config
        self.selected_loss = config['loss_function']

        # Set logger
        self.logger = logger

        # Set feature names
        self.feature_names = X.columns.tolist()

        # Set datasets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )

    @abstractmethod
    def save_model(self):
        """Saves the model to a file.
        
        This method must be implemented by subclass to define how the model
        should be persisted to disk.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'save_model' must be implemented in a subclass.")

    @abstractmethod
    def load_model(self):
        """Loads the model from a file.
        
        This method must be implemented by subclass to define how the model
        should be loaded from a disk.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'load_model' must be implemented in a subclass.")

    @abstractmethod
    def create_and_train_model(self, training_config: dict):
        """Abstract method for creating and training the model.

        This method must be implemented by subclass to define how to create
        and train the model.

        Args:
            training_config (dict): Model's training configuration.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "The method 'create_and_train_model' must be implemented in a subclass.")

    @abstractmethod
    def predict(self) -> np.ndarray:
        """Predict data using model.
        
        This method must be implemented by subclass to define how the model
        should make predictions.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'predict' must be implemented in a subclass.")

    def get_model_loss(self) -> float:
        """Method calculates model's loss value.

        Returns:
            float: Calculated loss value.
        """
        y_pred = self.predict()
        y_true = self.y_test

        return calculate_loss(
            y_true=y_true,
            y_pred=y_pred,
            loss_name=self.selected_loss,
            task_type=self.task_type
        )

    @abstractmethod
    def get_model_summary(self):
        """Summarize model's attributes.

        This method must be implemented by subclass to define how to summarize the model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'summarize' must be implemented in a subclass.")

    @abstractmethod
    def visualize_model(self):
        """Visualize the model.

        This method must be implemented by subclass to define how to visualize the model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'visualize_model' must be implemented in a subclass.")

    @abstractmethod
    def explain_shap(self):
        """Explain model using SHAP values.

        This method must be implemented by subclass to define
        how to explain the model using SHAP method.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'explain_shap' must be implemented in a subclass.")

    @abstractmethod
    def explain_lime(self, instances):
        """Explain model using LIME method.
        
        This method must be implemented by subclass to define
        how to explain the model using LIME method.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'explain_lime' must be implemented in a subclass.")


def obtain_loss_function(task_type: str, loss_name: str):
    """Obtains loss function according to selection and task type.

    The function validates if selected loss function is possible to use for
    selected task type.

    Args:
        task_type (str): Task type, 'regression' or 'classification'.
        loss_name (str): Selected loss function name.

    Raises:
        UnsupportedLossException: If an unsupported loss function is provided.
        UnsupportedTaskTypeException: If an unsupported task type is provided.

    Returns:
        callable: Loss function corresponding to the selected task type and configuration.
    """
    if task_type == TASK_TYPES[0]: # regression
        if loss_name not in LOSS_FUNCTIONS_REGRESSION:
            raise UnsupportedLossException(
                f'Unsupported regression loss: {loss_name}.')
        return LOSS_FUNCTIONS_REGRESSION[loss_name]

    if task_type == TASK_TYPES[1]: # classification
        if loss_name not in LOSS_FUNCTIONS_CLASSIFICATION:
            raise UnsupportedLossException(
                f'Unsupported classification loss: {loss_name}.')
        return LOSS_FUNCTIONS_CLASSIFICATION[loss_name]

    raise UnsupportedTaskTypeException(
        f"Unsupported task type: {task_type}. Use '{TASK_TYPES[0]}' or '{TASK_TYPES[1]}'.")


def calculate_loss(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        loss_name: str,
        task_type: str) -> float:
    """Calculates the aggregated loss function.

    The function calculates loss value on y_true and y_pred data
    for selected task type and loss function.
    
    Args:
        y_test (np.ndarray): Real Values.
        y_pred (np.ndarray): Predicted Values.
        loss_func (str): Selected loss function:
            - Regression: 'mae' (Mean Absolute Error) or 'mse' (Mean Squared Error).
            - Classification: 'accuracy' or 'log_loss'.
        task_type (str): Type of task ('regression' or 'classification').

    Returns:
        float: Computed loss function value.
    """

    # Obtain loss function and checks for its validity
    loss_fn = obtain_loss_function(task_type, loss_name)

    # Process predictions for classification tasks
    if task_type == TASK_TYPES[1]:

        # log_loss expects probabilities
        if loss_name == 'log_loss':
            # Ensure predictions are clipped to avoid log(0)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # Avoid log(0)
        else:
            # accuracy_score expects class labels
            if y_pred.dtype.kind == "f":
                y_pred = np.round(y_pred).astype(int)

    # Compute loss
    return float(loss_fn(y_true, y_pred))
