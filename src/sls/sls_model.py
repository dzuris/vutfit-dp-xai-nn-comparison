"Module containing SHAP-LIME-Summarization explainer implementation."
import os
import json
import numpy as np

# Default value for output filename
DEFAULT_OUTPUT_FILENAME = "sls_explanation.txt"

class SlsExplainer(): # pylint: disable=too-few-public-methods
    """SHAP-LIME-Summarization Explainer."""
    def __init__(self, config: dict):
        # Register path to output file
        filename = config.get("output_filename", DEFAULT_OUTPUT_FILENAME)
        self.output_file = os.path.join(config['tmp'], filename)

        # SHAP
        shap_file = config['shap']['file']
        shap_metadata = config['shap']['metadata_file']
        if shap_file and shap_metadata:
            self.shap_file = shap_file
            self.shap_metadata = shap_metadata
            print("SHAP ready")

        # LIME
        lime_files = config['lime']['files']
        if lime_files:
            print("LIME ready")

        # Summarization
        summarization_file = config['summarization']['file']
        if summarization_file:
            print("Summarization ready")

    def explain_shap(self) -> str:
        """Explain shap somehow."""
        data = np.load(self.shap_file, allow_pickle=True)
        shap_values = data["shap_values"]
        # background = data["background"]
        # X_train = data["X_train"]

        with open(self.shap_metadata, encoding='utf-8') as f:
            metadata = json.load(f)

        print(metadata["feature_names"])
        print(shap_values.shape)
