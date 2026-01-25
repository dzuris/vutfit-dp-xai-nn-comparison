# Summarization - LIME - SHAP Explanation

The SLS algorithm is explaining multiple obtained explanation methods into one text explanation.

## Running the file

1. Set `config_sls.yaml`.

```txt
# output_filename: "txtexpl.txt"    - OPTIONAL: output filename, default to 'sls_explanation.txt'
# output_folder: "tmp"              - OPTIONAL: output folder, default to '.'
folder_with_explanations: "tmp/exp" - folder containing the explanations
model_type: "nn" # Options: 'nn' and 'gp'   - type of the model
target_column: "Y1"         - target column name
lime_instance_index: 0      - lime instance index to explain
```

The files are expecting to be in specific format all in the selected folder:

Shap files:
- Shap values: "{model_type} _ shap _ values _ {target_column}.npz"
- Shap metadata: "{model_type} _ shap _ metadata _ {target_column}.json"

Lime files:
- Lime: "{model_type} _ lime _ {target_column}.json"

Summarize file:
- Summarization: "{model_type} _ summarize _ {target_column}.json"

2. Run `make sls` from project root directory.
