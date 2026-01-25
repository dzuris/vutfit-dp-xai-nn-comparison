"Module containing SHAP-LIME-Summarization explainer implementation."
import os
from src.sls.explainer import ExplanationGenerator

# Default value for output filename
DEFAULT_OUTPUT_FILENAME = "sls_explanation.txt"
DEFAULT_OUTPUT_FOLDER = "."

class SlsExplainer():
    """SLS (SHAP-LIME-Summarization) Explainer.

    This class orchestrates the generation of human-readable model explanations by
    combining SHAP feature importance analysis, LIME local explanations, and model
    architecture summaries into a unified report for non-expert users.

    Attributes:
        output_file (str): Path where the generated explanation report is saved.
        generator (ExplanationGenerator): Template-based explanation report generator.
        explanation_text (str): The complete generated explanation text.

    Methods:
        get_explanation() -> str:
            Retrieves the generated explanation report text combining model
            architecture, SHAP analysis, LIME analysis, and method comparison.
        get_output_file() -> str:
            Retrieves the output file path where the explanation report was saved.
    """
    def __init__(self, config: dict):
        """Initialize explainer and generate explanation report.

        Loads configuration, initializes ExplanationGenerator, loads all XAI
        artifacts (SHAP, LIME, model summary) based on config paths, and
        generates the combined explanation report.

        Args:
            config (dict): Configuration with model_type, target_column,
                          lime_instance_index, folder_with_explanations, etc.
        """
        # Register path to output file
        filename = config.get("output_filename", DEFAULT_OUTPUT_FILENAME)
        output_folder = config.get("output_folder", DEFAULT_OUTPUT_FOLDER)
        self.output_file = os.path.join(output_folder, filename)

        # Initialize the explanation generator
        model_type = config.get("model_type", "nn")
        self.generator = ExplanationGenerator(model_type=model_type)

        # Obtain configuration properties for building the file names
        explanations_folder = config.get('folder_with_explanations', "tmp/exp")
        target_column = config['target_column']

        # SHAP
        shap_file = os.path.join(
            explanations_folder, f"{model_type}_shap_values_{target_column}.npz"
        )
        shap_metadata = os.path.join(
            explanations_folder, f"{model_type}_shap_metadata_{target_column}.json"
        )
        if shap_file and shap_metadata:
            self.generator.load_shap_data(shap_file, shap_metadata)

        # LIME
        lime_instance_index = config['lime_instance_index']
        lime_file = os.path.join(
            explanations_folder, f"{model_type}_lime_{target_column}_{lime_instance_index}.json"
        )
        if lime_file:
            self.generator.load_lime_explanation(lime_file)

        # Summarization
        summarization_file = os.path.join(
            explanations_folder, f"{model_type}_summarize_{target_column}.json"
        )
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

    def get_output_file(self) -> str:
        """Get the output filename.
        
        Returns:
            The output file name
        """
        return self.output_file
