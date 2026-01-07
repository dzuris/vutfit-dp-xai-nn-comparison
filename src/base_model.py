"""BaseModel module.

This module contains abstract class for models.
"""
from abc import ABC, abstractmethod
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from logging_handler import LoggerHandler
from utils import calculate_loss, TASK_TYPES, MODELS_FOLDER


class BaseModel(ABC): # pylint: disable=too-many-instance-attributes
    """Abstract model class.

    This abstract class provides methods that needs
    to be implemented in subclasses.

    Attributes:
        class_names: list[str]: List of target column unique class names.
        file_path (str): Path to the model file.
        task_type (str): Type of task the model is dealing with (regression or classification).
        target_columns (list[str]): List of target columns.
        selected_loss (str): Selected loss function (mse, mae for regression;
            accuracy, log_loss for classification)
        target_unique_classes (ndarray): Array containing the
            unique class labels for classification.
        logger (LoggerHandler): Logging handler.
        feature_names (list[str]): List of feature names.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.DataFrame): Training targets.
        y_test (pd.DataFrame): Testing features.


    Methods:
        save_model():
            Saves model to the file.
        load_model():
            Loads model from the file.
        create_and_train_model():
            Creates and train model on provided data, with config settings.
        predict() -> dict:
            Predict test data.
        get_model_loss() -> float:
            Calculates loss function value on test data.
        get_model_summary():
            Summarize the trained model's attributes.
        visualize_model():
            Visualize the model.
        explain_shap():
            Explain the model using SHAP values.
        explain_lime() -> list(explanations):
            Explain the model using LIME algorithm and returns explanation for each instance.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            class_names: list[str],
            config: dict,
            logger: LoggerHandler,
            model_filename: str,
            folder_path: str = MODELS_FOLDER):
        """Model object constructor

        The method creates a file_path to file where model should be stored/loaded from. If the
        path to the file does not exist, the methods creates directories to the file. Moreover
        the function checks if obtained task type is valid and set the task_type attribute. Set
        the target_columns and selected_loss attributes. Checks for number of target columns
        validity and sets target_unique_classes attribute for classification task. Attach
        logger attribute and split data into X_train, X_test, y_train and y_test attributes.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset targets.
            class_names: list[str]: Unique class names for target column.
            config (dict): Configuration.
            logger (LoggerHandler): Logger.
            model_filename (str): Filename where model should be saved/loaded from.
            folder_path (str): Path to folder to where model should be saved/loaded from.
        """
        # Set unique class names
        self.class_names = class_names

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
        self.target_columns = config['data']['target_columns']
        self.selected_loss = config['loss_function']

        # For classification allow only one target column, and obtain number of classes for target
        if self.task_type == TASK_TYPES[1]:
            if y.shape[1] > 1:
                raise ValueError("Only one target column is allowed for classification.")
            # Obtain number of classification classes
            self.target_unique_classes = y.iloc[:, 0].unique()

        # Set logger
        self.logger = logger

        # Set feature names
        self.feature_names = X.columns.tolist()

        # Set datasets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state'])

    @abstractmethod
    def save_model(self):
        """Saves model to a file.
        
        This method must be implemented by subclass to define how the model
        should be persisted to disk.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'save_model' must be implemented in a subclass.")

    @abstractmethod
    def load_model(self):
        """Loads model from a file.
        
        This method must be implemented by subclass to define how the model
        should be loaded from a disk.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'load_model' must be implemented in a subclass.")

    @abstractmethod
    def create_and_train_model(self):
        """Abstract method for creating and training the model.

        This method must be implemented by subclass to define how to create
        and train the model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "The method 'create_and_train_model' must be implemented in a subclass.")

    @abstractmethod
    def predict(self) -> dict:
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
        predictions = self.predict()
        y_test_dict = self.y_test.to_dict(orient="list")

        print(f"Loss function: {self.selected_loss}")
        return calculate_loss(y_test_dict, predictions, self.selected_loss, self.task_type)

    @abstractmethod
    def get_model_summary(self):
        """Summarize model's attributes."""
        raise NotImplementedError("The method 'summarize' must be implemented in a subclass.")

    @abstractmethod
    def visualize_model(self):
        """Visualize the model."""
        raise NotImplementedError("The method 'visualize_model' must be implemented in a subclass.")

    @abstractmethod
    def explain_shap(self):
        """Explain model using SHAP values.

        This method must be implemented by subclass to define how to explain the model.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("The method 'explain_shap' must be implemented in a subclass.")

    @abstractmethod
    def explain_lime(self, instances):
        """Explain model using LIME method."""
        raise NotImplementedError("The method 'explain_lime' must be implemented in a subclass.")
