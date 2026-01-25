"""SLS (SHAP-LIME-Summarization) explainer smoke test.

Validates that the SLS pipeline can load SHAP values, SHAP metadata, LIME
explanations, and the model summary, and that it generates non-empty
explanations for architecture, SHAP, and LIME sections.

Usage:
    # Run directly
    python -m src.sls.test_sls

    # Run via Makefile
    make test-sls

Requirements:
    - Config file at 'src/sls/config_sls.yaml'
    - Pre-generated artifacts in the directory referenced by the config:
      * {model_type}_shap_values_{target}.npz
      * {model_type}_shap_metadata_{target}.json
      * {model_type}_lime_{target}_{index}.json
      * {model_type}_summarize_{target}.json

See Also:
    src.sls.sls_model.SlsExplainer: Orchestrates loading and report generation
    src.sls.explainer.ExplanationGenerator: Produces the explanation text
"""

import yaml
from src.sls.sls_model import SlsExplainer

def test_sls():
    """Run a basic SLS pipeline verification.

    Loads the SLS config, constructs `SlsExplainer`, asserts that SHAP/LIME/summary
    artifacts are loaded, and checks that each explanation section is non-empty.

    Raises:
        AssertionError: If any artifact is missing or if an explanation
            section is empty.
    """
    # Load config
    with open('src/sls/config_sls.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create explainer
    explainer = SlsExplainer(config)
    generator = explainer.generator

    # Test data loading
    assert generator.shap_data is not None, "SHAP data not loaded"
    assert generator.shap_metadata is not None, "SHAP metadata not loaded"
    assert generator.lime_explanation is not None, "LIME explanations not loaded"
    assert generator.model_summary is not None, "Model summary not loaded"

    # Test explanation generation
    arch_text = generator.explain_model_architecture()
    assert len(arch_text) > 0, "Architecture explanation is empty"

    shap_text = generator.explain_shap_values()
    assert len(shap_text) > 0, "SHAP explanation is empty"

    lime_text = generator.explain_lime_instance()
    assert len(lime_text) > 0, "LIME explanation is empty"

    print("✓ All tests passed!")
    print(f"  - SHAP values loaded: {generator.shap_data['values'].shape}")
    print(f"  - Features: {len(generator._get_feature_names())}") # pylint: disable=protected-access
    print(f"  - LIME instance index: {generator.lime_explanation['instance']}")
    print(f"  - Architecture explanation: {len(arch_text)} chars")
    print(f"  - SHAP explanation: {len(shap_text)} chars")
    print(f"  - LIME explanation: {len(lime_text)} chars")
    print(f"\n✓ Explanation saved to: {explainer.get_output_file()}")

if __name__ == '__main__':
    test_sls()
