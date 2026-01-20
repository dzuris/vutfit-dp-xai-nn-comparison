"Module containing SHAP-LIME-Summarization explainer implementation."
import os
from src.sls.explainer import ExplanationGenerator

# Default value for output filename
DEFAULT_OUTPUT_FILENAME = "sls_explanation.txt"
DEFAULT_OUTPUT_FOLDER = "."

class SlsExplainer(): # pylint: disable=too-few-public-methods
    """SHAP-LIME-Summarization Explainer."""
    def __init__(self, config: dict):
        # Register path to output file
        filename = config.get("output_filename", DEFAULT_OUTPUT_FILENAME)
        output_folder = config.get("output_folder", DEFAULT_OUTPUT_FOLDER)
        self.output_file = os.path.join(output_folder, filename)

        # Initialize the explanation generator
        self.generator = ExplanationGenerator(model_type=config.get("model_type", "nn"))

        # SHAP
        shap_file = config['shap']['file']
        shap_metadata = config['shap']['metadata_file']
        if shap_file and shap_metadata:
            self.generator.load_shap_data(shap_file, shap_metadata)

        # LIME
        lime_file = config['lime']['file']
        if lime_file:
            self.generator.load_lime_explanation(lime_file)

        # Summarization
        summarization_file = config['summarization']['file']
        if summarization_file:
            self.generator.load_model_summary(summarization_file)

        # Generate and save the full explanation
        self.explanation_text = self.generator.generate_full_explanation(
            output_file=self.output_file
        )
        print(f"Explanation generated and saved to: {self.output_file}")

    def get_explanation(self) -> str:
        """Get the generated explanation text.
        
        Returns:
            The full explanation text
        """
        return self.explanation_text
