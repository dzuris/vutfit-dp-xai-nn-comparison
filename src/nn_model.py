"""Neural Network model implementation for regression and classification.

Provides a concrete implementation of `BaseModel` using TensorFlow/Keras to
build, train, and explain neural networks. Supports single-target regression
and both binary and multi-class classification tasks. Includes training
callbacks (early stopping, learning rate reduction), XAI methods (SHAP, LIME, summarization),
and visualization of weights and layer activations.

Features:
    - Customizable architecture (hidden layers, units, activations)
    - Batch normalization and dropout for regularization
    - Multiple optimizers (Adam, SGD) and learning rate scheduling
    - Early stopping and learning rate reduction callbacks
    - TensorBoard integration for training monitoring
    - SHAP explanations for global feature importance
    - LIME explanations for local instance-level insights
    - Visualization of layer weights and activations

See Also:
    src.base_model.BaseModel: Abstract base class
    src.nn_model.NeuralNetworkModel: Concrete implementation
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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.optimizers import Adam, SGD # pylint: disable=no-name-in-module  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # pylint: disable=no-name-in-module  # type: ignore
from src.base_model import BaseModel
from src.logging_handler import LoggerHandler
from src.utils import MODELS_FOLDER, TASK_TYPES, TMP_FOLDER, EXPLANATIONS_STORE_FOLDER

class NeuralNetworkModel(BaseModel):
    """Keras-based neural network for supervised learning.

    Implements a fully-connected feedforward neural network for regression and
    classification tasks. Extends `BaseModel` with TensorFlow/Keras integration,
    training callbacks, and built-in XAI (SHAP and LIME) explanations.

    Attributes:
        model (Sequential): TensorFlow/Keras Sequential model instance.
        (Inherits from BaseModel: X_train, X_test, y_train, y_test, target_column,
         task_type, selected_loss, feature_names, logger, file_path, y_encoder, class_names)

    Methods:
        save_model():
            Save trained Keras model to disk.
        load_model():
            Load Keras model from disk.
        create_and_train_model(training_config: dict):
            Build architecture and train on data.
        predict(X=None) -> np.ndarray:
            Generate predictions for regression or classification.
        get_model_summary():
            Serialize model architecture and hyperparameters to JSON.
        visualize_model():
            Visualize layer 1 weights and hidden layer activations.
        explain_shap():
            Generate SHAP values and summary plots.
        explain_lime(instances):
            Generate LIME explanations for specified instances.
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
        """Initialize a neural network model.

        Calls parent `BaseModel.__init__()` to set up data splitting, logging,
        and metadata, then initializes the model attribute to `None`.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.DataFrame): Training targets.
            y_encoder (LabelEncoder): Encoder for categorical targets (optional).
            config (dict): Configuration dict with data, task, loss, and training settings.
            logger (LoggerHandler): Logger instance.
            model_filename (str): Filename for saving/loading the Keras model.
            folder_path (str): Directory path (default: `MODELS_FOLDER`).
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
        """Save the trained Keras model to disk.

        Uses `model.save()` to persist the model architecture, weights, and
        optimizer state to the file path specified in `self.file_path`.
        """
        self.model.save(self.file_path)

    def load_model(self):
        """Load a Keras model from disk.

        Restores the model from `self.file_path` using `tf.keras.models.load_model()`.

        Raises:
            FileNotFoundError: If the model file does not exist at `self.file_path`.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f'Cannot find provided file: {self.file_path}.')

        self.model = tf.keras.models.load_model(self.file_path) # pylint: disable=no-member

    def create_and_train_model(self, training_config: dict): # pylint: disable=too-many-branches
        """Build and train the neural network model.

        Constructs a Sequential model with input layer, hidden layers (with optional
        batch normalization and dropout), and task-specific output layer. Then trains
        using `model.fit()` with optional callbacks (early stopping, learning rate
        reduction, TensorBoard).

        Args:
            training_config (dict): Training hyperparameters including:
                - hidden_layers (int): Number of hidden layers
                - hidden_units (int): Neurons per hidden layer
                - activation_function (str): Activation (e.g., 'relu')
                - dropout_rate (float): Dropout probability (0.0-1.0)
                - batch_normalization (bool): Apply batch norm after each hidden layer
                - optimizer (str): 'adam' or 'sgd'
                - learning_rate (float): Initial learning rate
                - epochs (int): Training epochs
                - batch_size (int): Batch size
                - print_train_logs (bool): Verbosity level (0-2)
                - early_stopping (bool): Enable early stopping
                - reduce_lr (bool): Enable learning rate reduction
                - tensorboard_cb (bool): Enable TensorBoard logging

        Raises:
            ValueError: If optimizer not in ['adam', 'sgd'] or task_type unsupported.

        Notes:
            - Regression: single linear output (MSE loss)
            - Binary classification: single sigmoid output (binary crossentropy loss)
            - Multi-class: softmax output with sparse categorical crossentropy loss
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
            if training_config['batch_normalization']:
                self.model.add(BatchNormalization())
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

        callbacks = []

        # Early stopping
        if training_config.get('early_stopping', False):
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True
            )
            callbacks.append(early_stop)

        # ReduceLROnPlateau
        if training_config.get('reduce_lr', False):
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',      # Watch the validation loss
                factor=0.2,              # Reduce LR by 80% (new LR = LR * 0.2)
                patience=20,             # Wait 20 epochs before cutting LR
                min_lr=1e-6              # Don't let it drop below this value
            )
            callbacks.append(reduce_lr)

        # Tensorboard generating
        if training_config.get('tensorboard_cb', False):
            log_dir = f"{TMP_FOLDER}/fit"
            os.makedirs(log_dir, exist_ok=True) # Ensure the output directory exists
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir) # pylint: disable=no-member
            callbacks.append(tensorboard_cb)

        # Train the model
        self.model.fit(self.X_train,
                       self.y_train,
                       epochs=training_config['epochs'],
                       batch_size=training_config['batch_size'],
                       validation_data=(self.X_test, self.y_test),
                       verbose=training_config['print_train_logs'],
                       callbacks=callbacks)

    def predict(self, X=None) -> np.ndarray:
        """Generate predictions on input data.

        For regression: returns continuous values.
        For binary classification: converts sigmoid output (>0.5) to class labels.
        For multi-class: returns class indices from softmax output.

        Args:
            X (pd.DataFrame, optional): Input features. If None, uses `self.X_test`.

        Returns:
            np.ndarray: Predicted values (regression) or class labels (classification).
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
        """Serialize model architecture and training info to JSON.

        Saves a summary containing:
        - Layer-by-layer details (type, name, units, activation, dropout)
        - Total trainable parameters
        - Optimizer and learning rate
        - Selected loss function and computed loss value

        Output: `{EXPLANATIONS_STORE_FOLDER}/nn_summarize_{target_column}.json`
        """

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

        # Sets output file
        output_file = f"{EXPLANATIONS_STORE_FOLDER}/nn_summarize_{self.target_column}.json"

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(nn_summary, f, indent=4)

        print(f"Neural network summary saved into {output_file}.")

    def visualize_model(self):
        """Visualize layer 1 weights and hidden layer activations.

        Generates two types of visualizations:
        1. Heatmap of input->layer1 weights
        2. Heatmap of activations for each hidden layer (using first training sample)

        Outputs:
        - `nn_visualize_layer1_weights_{target}.png`
        - `nn_visualize_hidden_layer_{i}_activations_{target}.png` (one per hidden layer)
        """
        weights, _ = self.model.layers[0].get_weights()

        figure_file = (
            f"{EXPLANATIONS_STORE_FOLDER}/nn_visualize_layer1_weights_{self.target_column}.png"
        )
        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Layer 1 Weights")
        plt.savefig(figure_file)
        plt.close()

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
            figure_file = (
                f"{EXPLANATIONS_STORE_FOLDER}/"
                f"nn_visualize_hidden_layer_{i+1}_activations_{self.target_column}.png"
            )
            plt.figure(figsize=(8, 2.5))
            plt.imshow(act, aspect='auto', cmap='viridis')
            plt.colorbar(label="Activation value")
            plt.title(f"Hidden Layer {i+1} Activations")
            plt.yticks([]) # one sample
            plt.xlabel("Neuron index")
            plt.savefig(figure_file)
            plt.close()

    def explain_shap(self):
        """Generate SHAP feature importance explanations.

        Computes SHAP values using `GradientExplainer` and saves:
        - SHAP values array (`.npz`)
        - Feature metadata and base values (`.json`)
        - Summary plot (`.png`)

        For regression and binary classification: single summary plot.
        For multi-class: one summary plot per class.

        Outputs:
        - `nn_shap_values_{target}.npz`
        - `nn_shap_metadata_{target}.json`
        - `nn_shap_figure_{target}.png` (or per-class variants)
        """
        # Calculate shap values
        shap_values = self.calculate_shap_values(
            X_train=self.X_train,
            X_test=self.X_test,
            predict_fn_model=self.model,
            task_type=self.task_type,
            model_type="nn",
            target_column=self.target_column,
            class_names=self.class_names
        )

        # --- Regression or Binary Classification ---
        if self.task_type == TASK_TYPES[0] or (
            self.task_type == TASK_TYPES[1] and self.model.output_shape[1] == 1
        ):
            figure_file = f"{EXPLANATIONS_STORE_FOLDER}/nn_shap_figure_{self.target_column}.png"

            # If GradientExplainer returns a list for single output, take the first element
            vals_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

            shap.summary_plot(
                vals_to_plot,
                features=self.X_test,
                feature_names=self.X_test.columns.tolist(),
                show=False
            )
            plt.title(f"SHAP Summary Plot — Target: {self.target_column}\n")
            plt.tight_layout()
            plt.savefig(figure_file)
            plt.close()
            print(f"SHAP figure saved to: {figure_file}")
            return

        # --- Multi-class classification ---
        if self.task_type == TASK_TYPES[1] and self.model.output_shape[1] > 1:
            shap_values = np.transpose(shap_values, (2, 0, 1))
            for i, class_name in enumerate(self.class_names):
                figure_file = (
                    f"{EXPLANATIONS_STORE_FOLDER}/"
                    f"nn_shap_figure_{self.target_column}_{class_name}.png"
                )
                shap.summary_plot(
                    shap_values[i],
                    features=self.X_test,
                    feature_names=self.X_test.columns.tolist(),
                    show=False
                )
                plt.title(f"SHAP Summary Plot - Class: {class_name}")
                plt.tight_layout()
                plt.savefig(figure_file)
                plt.close()
                print(f"SHAP figure saved to: {figure_file}")

    def explain_lime(self, instances):
        """Generate LIME local explanations for specific instances.

        Creates an interpretable model for each instance to explain predictions
        locally. Saves both HTML and JSON formats.

        Args:
            instances (list[int]): Indices of training instances to explain.

        Outputs:
        - `nn_lime_{target}_{index}.html` (interactive plot)
        - `nn_lime_{target}_{index}.json` (structured explanation)

        Notes:
            - Uses top 10 most important features per explanation
            - For multi-class: explains all classes
        """

        # Initialize the LIME explainer
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=self.class_names if self.task_type == TASK_TYPES[1] else None,
            mode="regression" if self.task_type == TASK_TYPES[0] else "classification",
            random_state=42
        )

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
                EXPLANATIONS_STORE_FOLDER, f"nn_lime_{self.target_column}_{inst}.html"
            )
            explanation.save_to_file(explanation_file)
            print(f"Instance: {inst}")
            print(f"- Saved to: {explanation_file}")
            print(f"\n- Data:\n{self.X_train.iloc[inst]}")
            print(f"\n- Target:\n{self.y_train.iloc[inst]}")

            # Convert explanation to a dictionary
            explanation_dict = {
                "instance": int(inst),
                "target_column": self.target_column,
                "explanation": explanation.as_list(),
                "class_names": self.class_names.tolist() if self.class_names is not None else None,
                "data_row": self.X_train.iloc[inst].to_dict(),
                "target": int(self.y_train.iloc[inst])
            }

            # Save the explanation as a JSON file
            json_file = os.path.join(
                EXPLANATIONS_STORE_FOLDER, f"nn_lime_{self.target_column}_{inst}.json"
            )
            with open(json_file, "w", encoding='utf-8') as f:
                json.dump(explanation_dict, f, indent=4)

            print(f"- Saved JSON explanation to: {json_file}")
            print('-----------------------------------')
