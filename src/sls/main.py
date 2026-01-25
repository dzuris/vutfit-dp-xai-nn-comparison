"""Script for explaining XAI techniques."""
import argparse
from src.utils import load_config
from src.sls.sls_model import SlsExplainer


def basic_explanation():
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
