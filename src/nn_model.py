"""NeuralNetworkModel module.

This module contains implementation of Neural Network Model that is a subclass of BaseModel.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # pylint: disable=no-name-in-module  # type: ignore
from src.base_model import BaseModel
from src.logging_handler import LoggerHandler
from src.utils import MODELS_FOLDER, TASK_TYPES, TMP_FOLDER

class NeuralNetworkModel(BaseModel):
    """Neural Network Model.

    This class implements Neural Network Model methods.

    Attributes:
        model (Sequential): The Neural Network model.

    Methods:
        save_model():
            Saves the Neural Network model into a file.
        load_model():
            Loads the Neural Network model from a file.
        create_and_train_model():
            Creates and train Neural Network model on provided data, with config settings.
        predict() -> np.ndarray:
            Predicts test data.
        get_model_summary():
            Summarize the trained model's attributes.
        visualize_model():
            Visualize the NN model.
        explain_shap():
            Explains the model using SHAP method.
        explain_lime():
            Explains the model using LIME method.
    """
    def __init__( # pylint: disable=too-many-positional-arguments, too-many-arguments
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            y_encoder: LabelEncoder,
            config: dict,
            logger: LoggerHandler,
            model_filename: str,
            folder_path: str = MODELS_FOLDER):
        """
        Neural Network model constructor.

        Initializes the model.

        Args:
            X (pd.DataFrame): Dataset features.
            y (pd.DataFrame): Dataset target.
            y_encoder (LabelEncoder): Encoder for target column.
            config (dict): Configuration.
            logger (LoggerHandler): Handler for logging.
            model_filename (str): Neural Network model filename.
            folder_path (str): Path to folder where NN model files should be stored/loaded from.
        """
        super().__init__(
            X=X,
            y=y,
            y_encoder=y_encoder,
            config=config,
            logger=logger,
            model_filename=model_filename,
            folder_path=folder_path)
        self.model = None

    def save_model(self):
        """Saves the model into a file."""
        self.model.save(self.file_path)

    def load_model(self):
        """Loads the model from the file into model attribute.
        
        Raises:
            FileNotFoundError: If filepath is leading to no file.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f'Cannot find provided file: {self.file_path}.')

        self.model = tf.keras.models.load_model(self.file_path) # pylint: disable=no-member

    def create_and_train_model(self, training_config: dict): # pylint: disable=too-many-arguments, too-many-positional-arguments, arguments-differ
        """
        Create and train a single-output neural network model.

        Args:
            training_config: Model's training configuration.

        Raises:
            ValueError: If unsupported optimizer is provided in configuration for nn
                optimizer ('adam' or 'sgd' are valid values). Or unsupported type is
                provided ('regression' or 'classification' are valid values).
        """

        # Log the training configuration
        self.logger.add_log(f"- NN Training configuration: {training_config}")

        self.model = Sequential()

        # Input layer
        self.model.add(tf.keras.Input(shape=(self.X_train.shape[1],))) # pylint: disable=no-member

        # Hidden Layers
        for _ in range(training_config['hidden_layers']):
            self.model.add(
                Dense(training_config['hidden_units'],
                      activation=training_config['activation_function']))
            self.model.add(Dropout(training_config['dropout_rate']))

        # Output Layer
        if self.task_type == TASK_TYPES[0]: # regression
            # Single continuous output
            self.model.add(Dense(1, activation='linear'))
            loss_function = "mean_squared_error"
            metrics = ['mae', 'mse']

        elif self.task_type == TASK_TYPES[1]: # classification
            num_classes = len(self.class_names)

            if num_classes == 2:
                # Binary classification -> 1 output neuron
                self.model.add(Dense(1, activation="sigmoid"))
                loss_function = "binary_crossentropy"
                metrics = ["accuracy"]

            else:
                # Multi-class classification -> N output neurons
                self.model.add(Dense(num_classes, activation="softmax"))
                loss_function = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Set optimizer
        opt_name = training_config["optimizer"]
        lr = training_config["learning_rate"]
        if opt_name == 'adam':
            optimizer = Adam(learning_rate=lr)
        elif opt_name == 'sgd':
            optimizer = SGD(learning_rate=lr)
        else:
            raise ValueError(
                f'Unsupported Neural Network optimizer: {opt_name}')

        # Compile model
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

        # Train the model
        log_dir = f"{TMP_FOLDER}/fit"
        os.makedirs(log_dir, exist_ok=True) # Ensure the output directory exists
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir) # pylint: disable=no-member

        self.model.fit(self.X_train,
                       self.y_train,
                       epochs=training_config['epochs'],
                       batch_size=training_config['batch_size'],
                       validation_data=(self.X_test, self.y_test),
                       verbose=training_config['print_train_logs'],
                       callbacks=[tensorboard_cb])

    def predict(self, X=None) -> np.ndarray:
        """Predicts values using the trained neural network model.

        Args:
            X (pd.DataFrame, optional): Input features.
                If None, uses self.X_test.

        Returns:
            np.ndarray: Predicted values for the single target column.
        """

        # 1. Choose input data
        if X is None:
            X = self.X_test

        # 2. Raw model predictions
        preds = self.model.predict(X)

        # 3. Regression
        if self.task_type == TASK_TYPES[0]: # regression
            return preds.flatten()

        # 4. Classification
        num_outputs = self.model.output_shape[-1]

        if num_outputs == 1:
            # Binary classification -> sigmoid output
            # Convert probabilities to class labels (0 or 1)
            return (preds >= 0.5).astype(int).flatten()

        # Multi-class classification -> softmax output
        # Return class indices
        return np.argmax(preds, axis=1)

    # --- EXPLAINING FUNCTIONS ---
    def get_model_summary(self):
        """
        Saves a summary of the neural network model for interpretability into a json file.

        Includes:
        - Model architecture (layers, neurons, activation functions).
        - Total number of parameters.
        - Training information (loss function, optimizer, learning rate).
        - Computed loss value using selected loss function in configuration.
        """

        # Sets output file
        output_file = f"{TMP_FOLDER}/nn_summarize_{self.target_column}.json"

        # Create a dictionary to store the summary
        nn_summary = {
            "nn_architecture": {
                "num_of_layers": len(self.model.layers),
                "layers": []
            },
            "network_parameters": {
                "total_trainable_parameters": self.model.count_params(),
                "loss_function": self.model.loss,
                "optimizer": self.model.optimizer.name,
                "lr": float(self.model.optimizer.learning_rate.numpy())
            },
            "loss_value": {
                "selected_loss_name": self.selected_loss,
                "loss_value": self.get_model_loss()
            }
        }

        # Add layer details
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                "layer": i + 1,
                "name": layer.name,
                "type": type(layer).__name__
            }
            if hasattr(layer, 'units'):
                layer_info["neurons"] = layer.units
            if hasattr(layer, 'activation'):
                layer_info["activation"] = layer.activation.__name__
            if hasattr(layer, 'rate'):
                layer_info["dropout_rate"] = layer.rate
            nn_summary["nn_architecture"]["layers"].append(layer_info)

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(nn_summary, f, indent=4)

        print(f"Neural network summary saved into {output_file}.")

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
        """Explains the model using SHAP explainer."""

        os.makedirs(TMP_FOLDER, exist_ok=True)

        # Define prediction function
        def predict_fn(X):
            X = np.array(X)
            preds = self.model.predict(X)

            # Regression or Binary classification -> (n,) shape
            if self.task_type == "regression" or preds.shape[1] == 1:
                return preds.reshape(-1)

            # Multi-class -> (n, C)
            return preds

        # Calculate shap values
        shap_values = self.calculate_shap_values(
            X=self.X_train,
            predict_fn=predict_fn,
            task_type=self.task_type,
            model_type="nn",
            target_column=self.target_column,
            class_names=self.target_column
        )

        # --- Regression or Binary Classification ---
        if self.task_type == TASK_TYPES[0] or (
            self.task_type == TASK_TYPES[1] and self.model.output_shape[1] == 1
        ):
            figure_file = f"{TMP_FOLDER}/nn_shap_figure_{self.target_column}.png"
            shap.summary_plot(
                shap_values,
                features=self.X_train,
                feature_names=self.X_train.columns.tolist(),
                show=False
            )
            plt.title(f"SHAP Summary Plot — Target: {self.target_column}\n")
            plt.tight_layout()
            plt.savefig(figure_file)
            plt.show()
            print(f"SHAP figure saved to: {figure_file}")
            return

        # --- Multi-class classification ---
        if self.task_type == TASK_TYPES[1] and self.model.output_shape[1] > 1:
            shap_values = np.transpose(shap_values, (2, 0, 1))
            for i, class_name in enumerate(self.class_names):
                figure_file = f"{TMP_FOLDER}/nn_shap_figure_{self.target_column}_{class_name}.png"
                shap.summary_plot(
                    shap_values[i],
                    features=self.X_train.values,
                    feature_names=self.X_train.columns.tolist(),
                    show=False
                )
                plt.title(f"SHAP Summary Plot - Class: {class_name}")
                plt.tight_layout()
                plt.savefig(figure_file)
                plt.show()
                print(f"SHAP figure saved to: {figure_file}")

    def explain_lime(self, instances):
        """
        Explains the Neural Network model using LIME for the given instances.

        Args:
            instances (pd.DataFrame or np.ndarray): Instances to explain.
        """

        # Initialize the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=self.class_names if self.task_type == TASK_TYPES[1] else None,
            mode="regression" if self.task_type == TASK_TYPES[0] else "classification",
            random_state=42
        )

        # Ensure the output directory exists
        os.makedirs(TMP_FOLDER, exist_ok=True)

        # Explain each instance for each target column and save the explanation to a file
        for inst in instances:

            # Generate explanation for the given instance
            data_row = self.X_train.iloc[inst].values

            explanation = explainer.explain_instance(
                data_row=data_row,
                predict_fn=self.model.predict,
                labels=(
                    list(range(len(self.class_names)))
                    if self.task_type == TASK_TYPES[1]
                    else None
                ),
                num_features=min(10, self.X_train.shape[1])
            )

            # Save the explanation to a file
            explanation_file = os.path.join(
                TMP_FOLDER, f"nn_lime_{self.target_column}_{inst}.html"
            )
            explanation.save_to_file(explanation_file)
            print(f"Instance: {inst}")
            print(f"- Saved to: {explanation_file}")
            print(f"\n- Data:\n{self.X_train.iloc[inst]}")
            print(f"\n- Target:\n{self.y_train.iloc[inst]}")

            # Convert explanation to a dictionary
            explanation_dict = {
                "instance": inst,
                "target_column": self.target_column,
                "explanation": explanation.as_list(),
                "class_names": self.class_names,
                "data_row": self.X_train.iloc[inst].to_dict(),
                "target": self.y_train.iloc[inst]
            }

            # Save the explanation as a JSON file
            json_file = os.path.join(TMP_FOLDER, f"nn_lime_{self.target_column}_{inst}.json")
            with open(json_file, "w", encoding='utf-8') as f:
                json.dump(explanation_dict, f, indent=4)

            print(f"- Saved JSON explanation to: {json_file}")
            print('-----------------------------------')
