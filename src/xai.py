"""xai.py - Explaining the model.

This module loads the model and explain it.
"""
import time
import argparse
from pathlib import Path
from src.i_model import get_model
from src.utils import load_config
from src.preprocess import load_and_preprocess_data
from src.logging_handler import LoggerHandler
from src.exceptions import UnsupportedXaiMethodException

XAI_METHODS = ['shap', 'lime', 'visualize', 'summarize']

def main():
    """Main program function.

    The program explains selected model with selected xai method.
    """
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
    X, y, y_encoders = load_and_preprocess_data(config['data'])
    print('Data preprocessed...')

    # -------------------------
    # Load the model
    # -------------------------
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
    print('Model loaded...')

    # -------------------------
    # XAI
    # -------------------------
    # Obtain selected XAI method (e.g. SHAP, LIME, etc)
    print("Explaining the model...")
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
