# Summarization - LIME - SHAP Explanation

The SLS algorithm is explaining multiple obtained explanation methods into one text explanation.

## Running the file

1. Set `config_sls.yaml`.

```txt
output_filename: "txtexpl.txt"  - name of the output file
output_folder: "tmp"    - folder where to save the output file
model_type: "nn" # Options: 'nn' and 'gp'   - if we are explaining neural network or genetic programming model
shap:
  file: "tmp/nn_shap_values_Y1.npz"                 - shap values file
  metadata_file: "tmp/nn_shap_metadata_Y1.json"     - shap metadata file
lime:
  file: "tmp/nn_lime_Y1_0.json"                     - lime instance to explain file (only 1 always)
summarization:
  file: "tmp/nn_summarize_Y1.json"                  - summarization to explain
```

2. Run `make sls` from project root directory.
