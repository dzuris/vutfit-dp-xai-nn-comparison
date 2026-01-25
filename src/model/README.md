# Model training

This script trains the Genetic Programming or Neural network model on provided dataset.

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

model_training:
  nn:
    # Neural Network Model
    hidden_units: UNITS_PER_LAYER
    hidden_layers: LAYERS
    activation_function: ACTIVATION_FUNCTION_NAME # Must be valid for tensorflow networks
    batch_normalization: BOOL # If should add batch normalization layer after Dense layer
    learning_rate: LEARNING_RATE
    optimizer: OPTIMIZER_NAME   # Options: 'adam', 'sgd'
    epochs: NUMBER_OF_EPOCHS
    batch_size: BATCH_SIZE
    dropout_rate: DROPOUT_RATE
    early_stopping: BOOL    # If training should stop when performance stop increasing
    reduce_lr: BOOL   # If learning rate should 
    tensorboard_cb: BOOL    # If tensorboard should be generated during training
    print_train_logs: PRINT_TRAINING_LOGS # 0 - not printing, 1 - printing
  gp:
    # Genetic Programming Model
    population_size: POPULATION_SIZE
    generations: GENERATIONS
    crossover_pb: CROSSOVER_PROBABILITY
    mutation_pb: MUTATION_PROBABILITY
    penalty_bloat: BLOAT_PENALTY
    elitism: ELITISM BOOL

logging:
  format: LOGGING_FORMAT
  level: LOG_LEVEL
  file: LOG_FILE_PATH
```

## Run the program

Run the program using `Makefile`, or run `python -m src.model.main --config "path/to/config.yaml"` from the root folder.

## Authors and acknowledgement

Adam Dzurilla <xdzuri00@stud.fit.vutbr.cz>
