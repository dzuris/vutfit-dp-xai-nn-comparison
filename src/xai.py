"""Explaining model.

This module loads the model and explain it.
"""
import argparse
from utils import load_config
from preprocess import load_and_preprocess_data
from models import get_model
from logging_handler import LoggerHandler

def main():
    """Main program function.
    
    Load config, load model, and explain the model.
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

    # Explain model using SHAP
    model.explain_shap()
    print('MODEL SUCCESSFULLY EXPLAINED!')

if __name__ == "__main__":
    main()
