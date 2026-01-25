"""Program module for initializing the model."""
import argparse
import sys
from pathlib import Path
from src.preprocess import load_and_preprocess_data
from src.model.train import train_model
from src.logging_handler import LoggerHandler
from src.utils import (
    LOSS_FUNCTIONS_REGRESSION,
    LOSS_FUNCTIONS_CLASSIFICATION,
    TASK_TYPES, MODELS,
    load_config,
    set_reproducibility
)
from src.exceptions import (
    UnsupportedLossException,
    UnsupportedTaskTypeException,
    UnsupportedModelException
)


def validate_configuration(config: dict):
    """Config validation.

    The function validates configuration settings before training and raises exceptions
    if config is invalid.

    Args:
        config (dict): Configuration to validate.

    Raises:
        UnsupportedModelException: If provided model is not supported.
        UnsupportedTaskTypeException: If provided task type is not supported.
        UnsupportedLossException: If provided Loss is not supported for task type.
        ValueError: If for classification more targets are provided.
    """

    # Validate Selected Model
    selected_model = config['selected_model']
    if selected_model not in MODELS:
        raise UnsupportedModelException(
            f"Unsupported model: {selected_model}. Use '{MODELS[0]}' or '{MODELS[1]}'."
        )

    # Validate Task Type
    task_type = config['data']['type']
    if task_type not in TASK_TYPES:
        raise UnsupportedTaskTypeException(
            f"Unsupported task type: {task_type}. Use '{TASK_TYPES[0]}' or '{TASK_TYPES[1]}'."
        )

    # Validate Loss function selection
    loss_func = config['loss_function']
    if task_type == TASK_TYPES[0]:
        # Regression
        if loss_func not in LOSS_FUNCTIONS_REGRESSION:
            raise UnsupportedLossException(
                f'Unsupported regression loss function: {loss_func}.')
    elif task_type == TASK_TYPES[1]:
        # Classification
        if loss_func not in LOSS_FUNCTIONS_CLASSIFICATION:
            raise UnsupportedLossException(
                f'Unsupported classification loss function: {loss_func}.')

    # Validate target columns
    if task_type == TASK_TYPES[1]:
        num_targets = len(config['data']['target_columns'])
        if num_targets != 1:
            raise ValueError("For classification only one target column is allowed!")


def main(): # pylint: disable=too-many-locals
    """Model training pipeline entry point.

    This module provides a command-line interface for training machine learning models
    (Neural Networks or Genetic Programming) on regression or classification tasks.
    It handles configuration validation, data preprocessing, model training, and
    loss evaluation.

    Usage:
        python -m src.model.main --config config_training.yaml

    Examples:
        # Use default configuration
        python -m src.model.main

        # Use custom configuration file
        python -m src.model.main --config my_config.yaml

        # Via Makefile
        make model

    Configuration:
        The YAML config file should specify:
        - selected_model: "nn" (neural network) or "gp" (genetic programming)
        - data: Dataset path, target columns, task type, preprocessing options
        - loss_function: Metric for evaluating model performance
        - logging: Log file settings
        - model_training: model-specific training hyperparameters

    See Also:
        src.model.train.train_model: Model training orchestrator
        src.preprocess.load_and_preprocess_data: Data loading and preprocessing
        src.nn_model.NNModel: Neural network implementation
        src.gp_model.GPModel: Genetic programming implementation
    """
    # Set seed for reproducibility
    set_reproducibility()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Program for getting explanations from the model.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration file'
    )

    # Parse arguments
    args = parser.parse_args()

    # -----------------------------------------
    # Load the configuration
    # -----------------------------------------
    config = load_config(args.config)

    # Validate Configuration
    try:
        validate_configuration(config)
    except (UnsupportedLossException,
            UnsupportedTaskTypeException,
            UnsupportedModelException) as e:
        print("ERROR - ", e)
        sys.exit(1)

    # -----------------------------------------
    # Create Logger
    # -----------------------------------------
    dataset_file_name = Path(config['data']['dataset_path']).name
    logger = LoggerHandler(
        config=config['logging'],
        logger_name="MainLogger",
        file=dataset_file_name,
        model_type=config['selected_model'],
        program_type="Training")

    # -----------------------------------------
    # Load dataset and split it into train and test sets
    # -----------------------------------------
    print("\nPreprocessing...")
    X, y, y_encoders = load_and_preprocess_data(config['data'])

    # -----------------------------------------
    # Train the model
    # -----------------------------------------
    print("\nTraining...")
    models = {}
    target_columns = config['data']['target_columns']
    for target_column in target_columns:
        logger.add_log(f"Target column: '{target_column}'")
        model = train_model(
            X=X,
            y=y[target_column],
            y_encoder=y_encoders[target_column] if y_encoders else None,
            target_column=target_column,
            config=config,
            logger=logger
            )
        models[target_column] = model

    # -----------------------------------------
    # Make predictions
    # -----------------------------------------
    print("\nPredicting...")
    selected_loss = config['loss_function']
    log_loss_func = f"Loss function: '{selected_loss}'"
    log_loss_values_title = 'Loss values:'
    logger.add_log(log_loss_func)
    logger.add_log(log_loss_values_title)
    print(log_loss_func)
    print(log_loss_values_title)
    for target_column in target_columns:
        model = models[target_column]
        loss = model.get_model_loss()

        # Log results
        log_target_column_loss = f'- {target_column}: {loss}'
        logger.add_log(log_target_column_loss)
        print(log_target_column_loss)

    print("\n--- SUCCESSFUL RUN ---")


if __name__ == "__main__":
    main()
