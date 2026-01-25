# XAI explanations generator

## Configuration

```yaml
data:
  dataset_path: "PATH/TO/DATASET.csv"    
  type: TYPE # Options: regression, classification
  # If columns are not valid Python identifiers, the program coonverts them,
  # and here you need to specify valid column
  # names (e.g. "Concrete compressive strength(MPa, megapascals)" -> "Concrete_compressive_strength_MPa_megapascals").
  target_columns:
    - TARGET1
    - TARGET2
  missing_value_strategy: 'STRATEGY'  # Options: 'mean', 'median', 'mode', 'drop'
  test_size: TEST_SIZE # e.g. 0.2
  random_state: RANDOM_STATE # e.g. 42

selected_model: MODEL # Options: 'NeuralNetwork', 'GeneticProgramming'
loss_function: LOSS_FUNCTION_NAME # Options: regression: 'mae', 'mse', 'r2', classification: 'accuracy', 'log_loss'

xai:
  # model filename must match dataset in data config
  model_filename_to_explain: MODEL_FILEPATH
  model_target_column_name: TARGET_COLUMN
  # Method options for both nn and gp: 'shap', 'lime', 'visualize', 'summarize'
  method: XAI_METHOD
  lime:
    index_instance_to_explain: [LIME_INSTANCES_INDICES] # List of indices in data to explain

logging:
  format: LOG_MSG_FORMAT
  level: LOG_LEVEL
  file: LOG_FILEPATH
```

## Running the XAI script

Run `make xai`, or `python -m src.xai.main --config "path/to/config.yaml` from the project root folder.

## Authors and acknowledgement

Adam Dzurilla <xdzuri00@stud.fit.vutbr.cz>
