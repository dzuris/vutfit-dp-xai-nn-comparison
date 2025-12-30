"""Program module."""
import argparse
import sys
from pathlib import Path
from preprocess import load_and_preprocess_data
from train import train_model
from logging_handler import LoggerHandler
from utils import (
    LOSS_FUNCTIONS_REGRESSION,
    LOSS_FUNCTIONS_CLASSIFICATION,
    TASK_TYPES, MODELS,
    load_config)
from exceptions import (
    UnsupportedLossException,
    UnsupportedTaskTypeException,
    UnsupportedModelException
)


def validate_configuration(config: dict):
    """Config validation.

    The function validates configuration settings before training and raises exceptions
    if config is not valid.

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


def main():
    """Main program function for generating the model.
    
    Runs loading config, validating config, preprocessing, training and obtaining loss values.
    """
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Program for getting explanations from the model.')
    parser.add_argument(
        '--config', type=str, default='config.yaml', help='Path to the configuration file'
    )

    # Parse arguments
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Validate Configuration
    try:
        validate_configuration(config)
    except (UnsupportedLossException,
            UnsupportedTaskTypeException,
            UnsupportedModelException) as e:
        print("ERROR - ", e)
        sys.exit(1)

    # Create Logger
    logger = LoggerHandler(config['logging'], "MainLogger")

    # Log current running file
    logger.add_log(f'File: {Path(config['data']['dataset_path']).name}')

    print("\nPreprocessing...")
    # Load dataset and split it into train and test sets
    X, y = load_and_preprocess_data(config['data'])

    print("\nTraining...")
    # Train the model
    model = train_model(X, y, config, logger)

    # Make predictions
    print("\nPredicting...")
    loss = model.get_model_loss()

    # Log results
    logger.add_log(f'Loss function: {config['loss_function']}')
    logger.add_log(f'Loss on test data: {loss}')
    print(f'Loss on test data: {loss}')

    print("\n--- SUCCESSFUL RUN ---")


if __name__ == "__main__":
    main()
