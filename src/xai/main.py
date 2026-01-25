"""XAI (Explainable AI) pipeline entry point.

This module provides a command-line interface for explaining a trained model
using multiple XAI methods: SHAP (global importance), LIME (local explanations),
model visualization, and model summarization. It handles configuration loading,
data preprocessing, model loading, method execution, and logging.

Usage:
    python -m src.xai.main --config src/xai/config_xai.yaml

Examples:
    # Use default configuration
    python -m src.xai.main

    # Use custom configuration file
    python -m src.xai.main --config my_xai_config.yaml

    # Via Makefile
    make xai

Configuration:
    The YAML config file should specify:
    - selected_model: "nn" or "gp"
    - data: Dataset path, target columns, preprocessing options
    - xai.method: One of ["shap", "lime", "visualize", "summarize"]
    - xai.model_target_column_name: Target column to explain (must exist in data)
    - xai.model_filename_to_explain: Saved model filename to load
    - xai.lime.index_instance_to_explain: Indices of instances for LIME
    - logging: Logger configuration
    - output directories: Where to store generated artifacts

See Also:
    src.i_model.get_model: Factory to obtain model implementation
    src.preprocess.load_and_preprocess_data: Data loading and preprocessing
    src.utils.EXPLANATIONS_STORE_FOLDER: Output directory for XAI artifacts
    src.exceptions.UnsupportedXaiMethodException: Error type for unsupported methods
"""

import os
import time
import argparse
from pathlib import Path
from src.i_model import get_model
from src.utils import (
    load_config,
    set_reproducibility,
    ignore_warnings,
    EXPLANATIONS_STORE_FOLDER
)
from src.preprocess import load_and_preprocess_data
from src.logging_handler import LoggerHandler
from src.exceptions import UnsupportedXaiMethodException

XAI_METHODS = ['shap', 'lime', 'visualize', 'summarize']

def main():
    """Execute the XAI pipeline for a trained model.

    Workflow:
        1. Ignore warnings and set reproducibility seeds
        2. Parse command-line arguments for config file path
        3. Load configuration and initialize logger
        4. Load and preprocess dataset
        5. Load model checkpoint specified in config
        6. Execute selected XAI method:
           - "shap": Global feature importance via SHAP
           - "lime": Local explanations for specified instances
           - "visualize": Model architecture visualization
           - "summarize": Human-friendly model summary
        7. Log execution time and print completion message

    Command-line Arguments:
        --config (str): Path to YAML configuration file.
                        Default: 'config.yaml'

    Raises:
        ValueError: If `xai.model_target_column_name` is not present in the dataset.
        UnsupportedXaiMethodException: If selected XAI method is unknown or unsupported.
        FileNotFoundError: If model file or dataset path in config do not exist.

    Side Effects:
        - Creates output directory `EXPLANATIONS_STORE_FOLDER` if missing
        - Writes XAI artifacts (e.g., SHAP npz/json, LIME json, summaries) to disk
        - Logs progress and timing into configured logger

    Example:
        $ python -m src.xai.main --config src/xai/config_xai.yaml

    Notes:
        - Ensure the model checkpoint exists at `xai.model_filename_to_explain`
        - `xai.model_target_column_name` must match one of the dataset's target columns
        - For LIME, supply indices via `xai.lime.index_instance_to_explain`
    """
    ignore_warnings()
    # Set seed for reproducibility
    set_reproducibility()

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Explaining Model's behavior script")
    parser.add_argument('--config',
                        type=str,
                        default='config.yaml',
                        help="Path to the configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print('Config loaded...')

    # Create Logger
    dataset_file_name = Path(config['data']['dataset_path']).name
    logger = LoggerHandler(
        config=config['logging'],
        logger_name="XAILogger",
        file=dataset_file_name,
        model_type=config['selected_model'],
        program_type='XAI'
    )
    print('Logger created...')

    # Obtain XAI method for logging
    selected_method_xai = config['xai']['method']
    logger.add_log(f"Selected XAI method: '{selected_method_xai}'")

    # -------------------------
    # Load and preprocess the data
    # -------------------------
    print("\nLoading the data...")
    X, y, y_encoders = load_and_preprocess_data(config['data'])
    print('Data preprocessed.')

    # -------------------------
    # Load the model
    # -------------------------
    print("\nLoading the model...")
    target_column = config['xai']['model_target_column_name']

    # Check if the target_column exists in y
    if target_column not in y:
        raise ValueError(
            f"The taregt column '{target_column}' does not exist in the provided data, "
            f"possible target column options: {y.columns.values}."
        )
    logger.add_log(f"Target column: '{target_column}'")

    model = get_model(
        selected_model=config['selected_model'],
        X=X,
        y=y[target_column],
        y_encoder=y_encoders[target_column] if y_encoders else None,
        config=config,
        logger=logger,
        model_filename=config['xai']['model_filename_to_explain']
    )
    model.load_model()
    print('Model loaded.')

    # -------------------------
    # XAI
    # -------------------------
    # Obtain selected XAI method (e.g. SHAP, LIME, etc)
    print("\nExplaining the model...")

    # Create the folder to store explain files
    os.makedirs(EXPLANATIONS_STORE_FOLDER, exist_ok=True)

    # Measure explaining time
    start_time = time.time()

    try:
        if selected_method_xai == XAI_METHODS[0]: # shap
            model.explain_shap()
        elif selected_method_xai == XAI_METHODS[1]: # lime
            instances_to_explain = config['xai']['lime']['index_instance_to_explain']
            model.explain_lime(instances=instances_to_explain)
        elif selected_method_xai == XAI_METHODS[2]: # visualize
            model.visualize_model()
        elif selected_method_xai == XAI_METHODS[3]: # summarize
            model.get_model_summary()
        else:
            raise UnsupportedXaiMethodException(
                f"Unknown XAI method selection: {selected_method_xai}."
            )
    except KeyError as exc:
        raise UnsupportedXaiMethodException(
            f"Unsupported XAI method selection: {selected_method_xai}."
        ) from exc
    except NotImplementedError as exc:
        raise UnsupportedXaiMethodException(
            f"Unsupported XAI method: '{selected_method_xai}', "
            f"for task type: '{config['selected_model']}'."
            ) from exc

    # Stop measuring explaining time
    end_time = time.time()
    logger.add_log(f"Explaining time: {end_time - start_time}")

    print('MODEL SUCCESSFULLY EXPLAINED!')

if __name__ == "__main__":
    main()
