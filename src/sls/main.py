"""Script for explaining XAI techniques."""
import argparse
from src.utils import load_config
from src.sls.sls_model import SlsExplainer


def basic_explanation():
    """Generate basic explanation using default configuration."""

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

if __name__ == "__main__":
    # Run explanation
    basic_explanation()

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)
