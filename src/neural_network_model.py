"""NeuralNetworkModel module.

This module contains implementation of Neural Network Model that is subclass of BaseModel.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # pylint: disable=no-name-in-module  # type: ignore
from base_model import BaseModel
from logging_handler import LoggerHandler
from utils import MODELS_FOLDER, TASK_TYPES, TMP_FOLDER

class NeuralNetworkModel(BaseModel):
    """Neural Network Model.

    This class implements Neural Network Model methods.

    Attributes:
        training_config (dict): Configuration for training Neural Network.
        logger (LoggerHandler): Logging Handler.
        model (Sequential): The Neural Network model.

    Methods:
        save_model():
            Saves the Neural Network model to a file.
        load_model():
            Loads the Neural Network model from a file.
        create_and_train_model():
            Creates and train Neural Network model on provided data, with config settings.
        predict():
            Predicts test data.
        visualize_model():
            Visualize the model.
        explain_shap():
            Explains NN model using SHAP method.
        explain_lime() -> list(lime.explanation.Explanation):
            Explain the model using LIME algorithm and returns explanations for each instance.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            config: dict,
            logger: LoggerHandler,
            model_filename: str = "nn_trained_model.keras",
            folder_path: str = MODELS_FOLDER):
        """
        Neural Network model constructor.

        Sets training_config, logger and model attributes.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset targets.
            config (dict): Configuration.
            logger (LoggerHandler): Handler for logging.
            model_filename (str): Neural Network model filename.
            folder_path (str): Path to folder where NN model files should be stored/loaded from.
        """
        super().__init__(X, y, config, logger, model_filename, folder_path)
        self.training_config = config['model_training']['nn']
        self.logger.add_log(f"NN Training configuration: {self.training_config}")
        self.model = None

    def save_model(self):
        """Saves model into a file."""
        self.model.save(self.file_path)

    def load_model(self):
        """Loads the model from the file into model attribute.
        
        Raises:
            FileNotFoundError: If filepath is leading to no file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f'Cannot find provided file: {self.file_path}.')

        self.model = tf.keras.models.load_model(self.file_path) # pylint: disable=no-member

    def create_and_train_model(self): # pylint: disable=too-many-arguments, too-many-positional-arguments, arguments-differ
        """
        The function creates and train NN model.

        Firstly initializes model's architecture, creates output layer according to task type,
        sets optimizer, compiles the model and trains the model.

        Raises:
            ValueError: If unsupported optimizer is provided in configuration for nn
                optimizer ('adam' or 'sgd' are valid values). Or unsupported type is
                provided ('regression' or 'classification' are valid values).
        """
        self.model = Sequential()

        # Explicit input definition
        self.model.add(tf.keras.Input(shape=(self.X_train.shape[1],))) # pylint: disable=no-member

        # Hidden Layers
        for _ in range(self.training_config['hidden_layers']):
            self.model.add(
                Dense(self.training_config['hidden_units'],
                      activation=self.training_config['activation_function']))
            self.model.add(Dropout(self.training_config['dropout_rate']))

        # Output Layer
        if self.task_type == TASK_TYPES[0]:
            # No activation for regression
            number_of_outputs = len(self.target_columns)
            self.model.add(Dense(number_of_outputs, activation='linear'))
            loss_function = "mean_squared_error"
            metrics = ['mae', 'mse']
        elif self.task_type == TASK_TYPES[1]:
            # Classification
            classification_classes_count = len(self.target_unique_classes)
            if classification_classes_count == 1:
                # Binary classification
                self.model.add(Dense(1, activation='sigmoid'))
                loss_function = 'binary_crossentropy'
            else:
                # Multi-class classification
                self.model.add(Dense(classification_classes_count, activation='softmax'))
                loss_function = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Set optimizer
        if self.training_config['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=self.training_config['learning_rate'])
        elif self.training_config['optimizer'] == 'sgd':
            optimizer = SGD(learning_rate=self.training_config['learning_rate'])
        else:
            raise ValueError(
                f'Unsupported Neural Network optimizer: {self.training_config['optimizer']}')

        # Compile model
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

        # Train the model
        log_dir = f"{TMP_FOLDER}/fit"
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir) # pylint: disable=no-member
        self.model.fit(self.X_train, self.y_train,
                       epochs=self.training_config['epochs'],
                       batch_size=self.training_config['batch_size'],
                       validation_data=(self.X_test, self.y_test),
                       verbose=self.training_config['print_train_logs'],
                       callbacks=[tensorboard_cb])

    def predict(self) -> dict:
        """Predicts test data using the model.

        Returns:
            (dict): Predictions for each target column where column name is the key and
                list of predictions is value.
        """
        # List of predictions
        predictions_nn = self.model.predict(self.X_test)

        # Initialize the dictionary
        result_dict = {target: [] for target in self.target_columns}

        if self.task_type == TASK_TYPES[0]:
            # For regression, return raw values
            for value in predictions_nn:
                for idx, target in enumerate(self.target_columns):
                    result_dict[target].append(value[idx])
        elif self.task_type == TASK_TYPES[1]:
            # Classification
            target_column = self.target_columns[0]
            if self.model.output_shape[-1] == 1:
                # Binary classification: Convert probabilities to class labels (0 or 1)
                predictions_nn = (predictions_nn >= 0.5).astype(int).flatten()
                result_dict[target_column] = predictions_nn.tolist()
            else:
                # Multi-class classification: Get the class index with highest probability
                predictions_nn = np.argmax(predictions_nn, axis=1)
                result_dict[target_column] = predictions_nn.tolist()

        return result_dict

    def visualize_model(self):
        """
        Visualize the weights and activations of the Dense layers in the neural network.

        This function visualizes:
        1. The weights of the first layer in the model.
        2. The activations of all Dense layers in the model for a single input sample.

        Raises:
            ValueError: If the model does not contain any Dense layers.
        """
        weights, _ = self.model.layers[0].get_weights()

        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Layer 1 Weights")
        plt.show()

        _ = self.model(self.X_test[:1])

        # Select Dense layers only
        dense_layers = [
            layer for layer in self.model.layers if isinstance(layer, tf.keras.layers.Dense) # pylint: disable=no-member
            ]

        activation_model = Model(
            inputs=self.model.inputs,
            outputs=[layer.output for layer in dense_layers]
        )

        x_sample = self.X_train[:1]

        activations = activation_model.predict(x_sample)

        for i, act in enumerate(activations[:-1]): # exclude output layer
            plt.figure(figsize=(8, 2.5))
            plt.imshow(act, aspect='auto', cmap='viridis')
            plt.colorbar(label="Activation value")
            plt.title(f"Hidden Layer {i+1} Activations")
            plt.yticks([]) # one sample
            plt.xlabel("Neuron index")
            plt.show()

    def explain_shap(self):
        """Explain model using SHAP explainer on train data."""
        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(self.model, self.X_train.sample(100, random_state=42))

        # Calculate SHAP values
        shap_values = explainer.shap_values(self.X_train)

        # Handle single and multi-output NN shap values
        # For loop iterates through every output, for each output depicts summary
        # graph which feature contributes to the result
        for output_idx, col in enumerate(self.target_columns):
            if shap_values.ndim == 2:   # Regression or single-probability binary
                shap_to_plot = shap_values
            else:
                shap_to_plot = shap_values[:, :, output_idx]

            shap.summary_plot(
                shap_to_plot,
                features=self.X_train,
                feature_names=self.X_train.columns.tolist(),
                show=False
            )
            plt.title(f"SHAP Summary Plot — Target: {col}\n")
            plt.tight_layout()
            plt.show()

    def explain_lime(self, instances):
        """
        Explain the Neural Network model using LIME for the given instances.

        Args:
            instances (pd.DataFrame or np.ndarray): Instances to explain.

        Raises:
            ValueError: If not trained model exists for any target variable.

        Returns:
            list: A list of LIME explanation objects for each instance.
        """
        # Determine mode, class names and labels based on task type
        if self.task_type == TASK_TYPES[1]: # Classification
            mode = "classification"
            class_names = [str(cls) for cls in self.y_train.iloc[:, 0].unique()]
            labels = list(range(len(class_names)))
        elif self.task_type == TASK_TYPES[0]: # Regression
            mode = "regression"
            class_names = None
            labels = None
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Initialize the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=class_names,
            mode=mode,
            random_state=42
        )

        # Ensure the output directory exists
        os.makedirs(TMP_FOLDER, exist_ok=True)

        # Explain each instance and save the explanation to a file
        explanations = []
        for index in range(len(instances)):
            # Generate explanation for the given instance
            data_row = self.X_train.iloc[index].values
            explanation = explainer.explain_instance(
                data_row=data_row,
                predict_fn=self.model.predict,
                labels=labels,
                num_features=self.X_train.shape[1]
            )
            explanations.append(explanation)

            # Save the explanation to a file
            explanation_file = os.path.join(TMP_FOLDER, f"lime_explanation_nn_{index}.html")
            explanation.save_to_file(explanation_file)
            print(f"Explanation for instance {index} saved to {explanation_file}")

        return explanations
