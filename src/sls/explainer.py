"""Template-based explainer for SHAP, LIME and Model Summarization.

This module provides human-readable text explanations combining SHAP values,
LIME explanations, and model summarization for non-expert users.
"""
import json
from typing import Dict, List
import numpy as np


class ExplanationGenerator:
    """Generates human-readable explanations from SHAP, LIME, and model data."""

    def __init__(self, model_type: str = "nn"):
        """Initialize the explanation generator."""
        self.model_type = model_type
        self.shap_data = None
        self.lime_explanation = None
        self.model_summary = None
        self.shap_metadata = None

    def load_shap_data(self, shap_file: str, metadata_file: str) -> None:
        """Load SHAP values and metadata.
        
        Args:
            shap_file: Path to .npz file containing SHAP values
            metadata_file: Path to .json file with feature names
        """
        data = np.load(shap_file, allow_pickle=True)
        self.shap_data = {
            'values': data['shap_values'],
            'background': data['background'],
            'X_train': data['X_train']
        }

        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.shap_metadata = json.load(f)

    def load_lime_explanation(self, lime_file: str) -> None:
        """Load LIME explanations from JSON files.
        
        Args:
            lime_files: List of paths to LIME explanation JSON files
        """
        with open(lime_file, 'r', encoding='utf-8') as f:
            self.lime_explanation = json.load(f)

    def load_model_summary(self, summary_file: str) -> None:
        """Load model summarization.
        
        Args:
            summary_file: Path to model summary JSON file
        """
        with open(summary_file, 'r', encoding='utf-8') as f:
            self.model_summary = json.load(f)

    def _get_task_type(self) -> str:
        """Get task type from metadata."""
        if self.shap_metadata:
            return self.shap_metadata.get('task_type', '')
        return ''

    def _get_feature_names(self) -> List[str]:
        """Get feature names from metadata."""
        if self.shap_metadata:
            return self.shap_metadata.get('feature_names', [])
        return []

    def _get_feature_attribs(self) -> Dict:
        """Get feature attributes from metadata."""
        if self.shap_metadata:
            return self.shap_metadata.get('feature_attribs', {})
        return {}

    def _get_target_column(self) -> str:
        """Get target column from metadata."""
        if self.shap_metadata:
            return self.shap_metadata.get('target_column', 'target')
        return 'target'

    def explain_model_architecture(self) -> str: # pylint: disable=too-many-statements, too-many-locals
        """Explain model architecture in non-expert terms.
        
        Returns:
            String explanation of model architecture
        """
        if not self.model_summary:
            return ""

        text = []
        text.append("=" * 70)
        text.append("HOW THE MODEL WORKS - MODEL ARCHITECTURE")
        text.append("=" * 70)
        text.append("")

        model_type = (self.model_type or "").lower()

        # --- Neural Network summary ---
        if model_type == "nn" and "nn_architecture" in self.model_summary:
            arch = self.model_summary.get('nn_architecture', {})
            layers = arch.get('layers', [])
            params = self.model_summary.get('network_parameters', {})

            text.append("WHAT IS THIS MODEL?")
            text.append("-" * 70)
            text.append("This is an artificial neural network - think of it as a series of")
            text.append("interconnected layers that learn patterns from data.")
            text.append("")

            text.append("MODEL STRUCTURE:")
            text.append("-" * 70)

            dense_layers = [l for l in layers if l.get('type') == 'Dense']
            dropout_layers = [l for l in layers if l.get('type') == 'Dropout']

            text.append(f"• Total layers: {len(layers)}")
            text.append(f"• Processing layers (Dense): {len(dense_layers)}")
            text.append(f"• Dropout layers (for regularization): {len(dropout_layers)}")
            text.append("")

            text.append("LAYER-BY-LAYER BREAKDOWN:")
            text.append("-" * 70)
            for layer in layers:
                layer_type = layer.get('type', 'Unknown')
                layer_name = layer.get('name', 'unnamed')

                if layer_type == 'Dense':
                    neurons = layer.get('neurons')
                    activation = layer.get('activation')
                    text.append(f"• Layer: {layer_name}")
                    text.append(f"  - Type: Processing layer with {neurons} neurons")
                    text.append(f"  - Activation: {activation} (how neurons process information)")
                    text.append("")
                elif layer_type == 'Dropout':
                    dropout_rate = layer.get('dropout_rate', 0) * 100
                    text.append(f"• Layer: {layer_name}")
                    text.append(f"  - Type: Regularization ({dropout_rate:.0f}% dropout)")
                    text.append("  - Purpose: Prevents overfitting by randomly disabling neurons")
                    text.append("")

            text.append("TRAINING CONFIGURATION:")
            text.append("-" * 70)
            text.append(
                f"• Total trainable parameters: {params.get('total_trainable_parameters', 'N/A')}"
            )
            text.append(f"• Loss function: {params.get('loss_function', 'N/A')}")
            text.append("  (Measures how wrong the model's predictions are)")
            text.append(f"• Optimizer: {params.get('optimizer', 'N/A')}")
            text.append("  (Algorithm that adjusts the model to improve)")
            text.append(f"• Learning rate: {params.get('lr', 'N/A')}")
            text.append("  (How big each adjustment step is)")
            text.append("")

            loss_info = self.model_summary.get('loss_value', {})
            text.append("CURRENT PERFORMANCE:")
            text.append("-" * 70)
            text.append(f"• Loss metric: {loss_info.get('selected_loss_name', 'N/A')}")
            text.append(f"• Loss value: {loss_info.get('loss_value', 'N/A'):.4f}")
            text.append("  (Lower loss = better performance)")
            text.append("")

            return "\n".join(text)

        # --- Genetic Programming summary ---
        if model_type == "gp":
            text.append("WHAT IS THIS MODEL?")
            text.append("-" * 70)
            text.append("This is a genetic programming model - it evolves mathematical")
            text.append("expressions (trees) to fit the data.")
            text.append("")

            # Core attributes from GP summary
            tree_str = self.model_summary.get("best_tree", "N/A")
            num_nodes = self.model_summary.get("number_of_nodes", "N/A")
            depth = self.model_summary.get("depth", "N/A")
            avg_branching = self.model_summary.get("avg_branching_factor", "N/A")
            loss_info = self.model_summary.get("loss_func", {})
            rules = self.model_summary.get("rules", [])

            text.append("MODEL STRUCTURE:")
            text.append("-" * 70)
            text.append("• Representation: Evolved expression tree")
            text.append(f"• Depth (how many nested steps): {depth}")
            text.append(f"• Number of nodes (complexity): {num_nodes}")
            text.append(f"• Average branching (children per node): {avg_branching}")
            text.append("")

            text.append("KEY RULE (HUMAN-READABLE FORM):")
            text.append("-" * 70)
            if rules:
                text.append(f"• Rule example: {rules[0]}")
            else:
                text.append("• Rules not provided in summary.")
            text.append("")

            text.append("TRAINING/QUALITY METRIC:")
            text.append("-" * 70)
            text.append(f"• Metric: {loss_info.get('name', 'N/A')}")
            text.append(f"• Value: {loss_info.get('value', 'N/A')}")
            text.append("  (Lower loss = better performance for regression)")
            text.append("")

            text.append("MODEL STRING (for auditors):")
            text.append("-" * 70)
            text.append(f"• {tree_str}")
            text.append("")

            return "\n".join(text)

        # Fallback if model_type not recognized
        text.append("Model type not recognized; please check the configuration and summary.")
        return "\n".join(text)

    def explain_shap_values(self) -> str:
        """Explain SHAP values for a specific instance.
            
        Returns:
            String explanation of SHAP values
        """
        if self.shap_data is None or self.shap_metadata is None:
            return ""

        text = []
        text.append("=" * 70)
        text.append("FEATURE IMPORTANCE - SHAP ANALYSIS")
        text.append("=" * 70)
        text.append("")

        feature_names = self._get_feature_names()
        shap_values = self.shap_data['values']

        # Calculate mean absolute SHAP values for overall importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Sort features by importance
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        text.append("WHAT IS SHAP?")
        text.append("-" * 70)
        text.append("SHAP (SHapley Additive exPlanations) tells us how much each feature")
        text.append("(input variable) contributes to the model's prediction. Positive values")
        text.append("push the prediction up, negative values push it down.")
        text.append("")

        text.append("OVERALL FEATURE IMPORTANCE (based on all instances):")
        text.append("-" * 70)

        for rank, idx in enumerate(sorted_indices[:5], 1):
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                importance = mean_abs_shap[idx]
                text.append(f"{rank}. {feature_name}: {importance:.4f}")
        text.append("")

        # Other attributes
        text.append("RANGE AND DIRECTION")
        text.append("-" * 70)
        for rank, idx in enumerate(sorted_indices[:5], 1):
            feature_name = feature_names[idx]
            feature_attribs = self._get_feature_attribs()[feature_name]
            impact_min = feature_attribs["range_min"]
            impact_max = feature_attribs["range_max"]
            direction = feature_attribs["direction"]
            text.append(
                f"{rank}. {feature_name} range: {impact_min:.2f} to {impact_max:.2f},"
                f" direction: {direction}."
            )

        text.append("")
        text.append("What does it mean when direction is positive?")
        text.append("- Positive direction means that higher values of feature")
        text.append("  push model predicted value higher.")
        text.append("- Lower feature values push model prediction to lower predicted value.")
        text.append("What does it mean when direction is inverse?")
        text.append("- Inverse direction means that higher values of feature")
        text.append("  push model predicted value lower.")
        text.append("- Lower feature values push model prediction to higher predicted value.")

        return "\n".join(text)

    def explain_lime_instance(self) -> str:
        """Explain LIME prediction for a specific instance.
        
        Args:
            instance_idx: Index of the LIME explanation to use
            
        Returns:
            String explanation of LIME
        """
        if not self.lime_explanation:
            return ""

        text = []
        text.append("=" * 70)
        text.append("LOCAL EXPLANATION - LIME ANALYSIS")
        text.append("=" * 70)
        text.append("")

        lime = self.lime_explanation

        text.append("WHAT IS LIME?")
        text.append("-" * 70)
        text.append("LIME (Local Interpretable Model-agnostic Explanations) explains a")
        text.append("single prediction by finding simple rules that work for similar inputs.")
        text.append("It shows which ranges of feature values matter most for THIS prediction.")
        text.append("")

        target = lime.get('target', 'N/A')
        text.append(f"PREDICTED VALUE: {target}")
        text.append("")

        # Get input values
        data_row = lime.get('data_row', {})
        text.append("INPUT VALUES FOR THIS PREDICTION:")
        text.append("-" * 70)
        for feature, value in sorted(data_row.items()):
            text.append(f"• {feature}: {value}")
        text.append("")

        # Get explanations
        explanation = lime.get('explanation', [])
        text.append("FEATURE CONTRIBUTIONS (What made the model predict this value?):")
        text.append("-" * 70)

        for feature_rule, contribution in explanation:
            direction = "increases" if contribution > 0 else "decreases"
            text.append(f"• Rule: {feature_rule}")
            text.append(f"  Impact: {direction} prediction by {abs(contribution):.4f}")
            text.append("")

        return "\n".join(text)

    def explain_prediction_comparison(self) -> str:
        """Generate comparison explanation between SHAP and LIME.

        Returns:
            String explanation comparing both methods
        """
        text = []
        text.append("=" * 70)
        text.append("COMPARING EXPLANATIONS: SHAP vs LIME")
        text.append("=" * 70)
        text.append("")

        text.append("WHY TWO METHODS?")
        text.append("-" * 70)
        text.append("SHAP shows which features are generally important across ALL")
        text.append("predictions (global view).")
        text.append("")
        text.append("LIME shows which specific feature ranges matter for THIS particular")
        text.append("prediction (local view).")
        text.append("")
        text.append("Together, they give a complete understanding: what matters in general,")
        text.append("and what was specifically important for this prediction.")
        text.append("")

        return "\n".join(text)

    def generate_full_explanation(self, output_file: str = None) -> str:
        """Generate complete explanation combining all methods.

        Args:
            output_file: Optional path to save the explanation

        Returns:
            Complete explanation text
        """
        sections = []

        # Add header
        header = [
            "=" * 70,
            "MODEL EXPLAINABILITY REPORT",
            "Non-Expert User Guide",
            "=" * 70,
            "",
            "This report explains how the model works and why it made",
            "specific predictions using three complementary methods:",
            "1. Model Architecture - How the model is structured",
            "2. SHAP - Which features generally matter most",
            "3. LIME - Which specific values mattered for this prediction",
            "",
        ]
        sections.append("\n".join(header))

        # Add model architecture
        arch_explanation = self.explain_model_architecture()
        if arch_explanation:
            sections.append(arch_explanation)

        # Add SHAP explanation
        shap_explanation = self.explain_shap_values()
        if shap_explanation:
            sections.append(shap_explanation)

        # Add LIME explanation
        lime_explanation = self.explain_lime_instance()
        if lime_explanation:
            sections.append(lime_explanation)

        # Add comparison
        comparison = self.explain_prediction_comparison()
        if comparison:
            sections.append(comparison)

        # Add footer
        footer = [
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
            "",
            "For more information, consult the documentation or contact the",
            "model developer.",
        ]
        sections.append("\n".join(footer))

        full_text = "\n\n".join(sections)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)

        return full_text
