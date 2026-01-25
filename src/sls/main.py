"""SLS (SHAP-LIME-Summarization) explanation generator.

This module provides a command-line interface for generating human-readable
explanations from trained models using SHAP, LIME, and model summarization.
The explanations are designed for non-expert users and combine global feature
importance, local predictions, and model architecture descriptions.

Usage:
    python -m src.sls.main --config config_sls.yaml

Examples:
    # Use default configuration
    python -m src.sls.main

    # Use custom configuration file
    python -m src.sls.main --config my_custom_config.yaml

    # Via Makefile
    make sls

Configuration:
    The YAML config file should specify:
    - model_type: "nn" (neural network) or "gp" (genetic programming)
    - shap_file: Path to SHAP values file (.npz format)
    - shap_metadata_file: Path to SHAP metadata file (.json format)
    - lime_file: Path to LIME explanation file (.json format)
    - model_summary_file: Path to model summary file (.json format)
    - output_file: Path where the generated explanation report will be saved

See Also:
    src.sls.sls_model.SlsExplainer: Main explanation orchestrator class
    src.sls.explainer.ExplanationGenerator: Template-based report generator
    src.xai.main: XAI pipeline that generates SHAP/LIME artifacts
    src.model.main: Model training pipeline
"""
import argparse
from src.utils import load_config
from src.sls.sls_model import SlsExplainer


def basic_explanation():
    """Generate a human-readable explanation report from XAI artifacts.

    Workflow:
        1. Parse command-line arguments for the config path
        2. Load YAML configuration
        3. Initialize `SlsExplainer` with the config
        4. Generate the explanation report and print it
        5. Print the output file path

    Command-line Arguments:
        --config (str): Path to YAML configuration file.
                        Default: 'config_txtexl.yaml'

    Raises:
        FileNotFoundError: If the config file or referenced artifact files are missing.
        json.JSONDecodeError: If any JSON artifact file is malformed.
        KeyError: If required configuration keys are missing.

    Example:
        $ python -m src.sls.main --config src/sls/config_sls.yaml

    Notes:
        - The `SlsExplainer` expects pre-generated SHAP, LIME, and summary files
          in `folder_with_explanations`.
        - The report is saved to the path composed from `output_folder` and `output_filename`.
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
    explainer = SlsExplainer(config)
    explanation = explainer.get_explanation()

    print(explanation)
    print("\n✓ Explanation saved to:", explainer.get_output_file())

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)

if __name__ == "__main__":
    # Run explanation
    basic_explanation()
