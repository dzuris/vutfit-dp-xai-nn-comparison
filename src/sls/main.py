"""Script for explaining XAI techniques."""
import argparse
from src.utils import load_config
from src.sls.sls_model import SlsExplainer


def main():
    """Script for generating text explanation for already explained data.

    The program uses techniques SHAP, LIME and summarization for generating
    text explanation about the model targeting non-experts.
    """

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Program for getting explanations from the model.'
    )
    parser.add_argument(
        '--config', type=str, default='config_txtexl.yaml', help='Path to the configuration file'
    )

    # Parse arguments
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Initialize explanation
    _ = SlsExplainer(config)

if __name__ == "__main__":
    main()
