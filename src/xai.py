"""Explaining model.

This module loads the model and explain it.
"""
import argparse
from i_model import get_model
from utils import load_config
from preprocess import load_and_preprocess_data
from logging_handler import LoggerHandler
from exceptions import UnsupportedXaiMethodException

XAI_METHODS = ['shap', 'lime', 'visualize']

def main():
    """Main program function.
    
    Load config, load model, and explain the model using selected method.
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
    logger = LoggerHandler(config['logging'], "XAILogger")
    print('Logger created...')

    # Load and preprocess the data
    X, y = load_and_preprocess_data(config['data'])
    print('Data preprocessed...')

    # Load the model
    selected_model = config['selected_model']
    model = get_model(selected_model, X, y, config, logger)
    model.load_model()
    print('Model loaded...')

    # Obtain selected XAI method (e.g. SHAP, LIME, etc)
    selected_method_xai = config['xai']['method']

    try:
        if selected_method_xai == XAI_METHODS[0]: # shap
            model.explain_shap()
        elif selected_method_xai == XAI_METHODS[1]: # lime
            _ = model.explain_lime(config['xai']['lime']['index_instance_to_explain'])
        elif selected_method_xai == XAI_METHODS[2]: # visualize
            model.visualize_model()
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

    print('MODEL SUCCESSFULLY EXPLAINED!')

if __name__ == "__main__":
    main()
