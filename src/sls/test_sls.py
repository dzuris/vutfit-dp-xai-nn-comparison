#!/usr/bin/env python3
"""Quick test of the SLS explanation system."""

import yaml
from src.sls.sls_model import SlsExplainer

def test_sls():
    """Test the SLS system."""
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
    print(f"  - LIME instances: {len(generator.lime_explanation)}")
    print(f"  - Architecture explanation: {len(arch_text)} chars")
    print(f"  - SHAP explanation: {len(shap_text)} chars")
    print(f"  - LIME explanation: {len(lime_text)} chars")
    print("\n✓ Explanation saved to: tmp/txtexpl.txt")

if __name__ == '__main__':
    test_sls()
