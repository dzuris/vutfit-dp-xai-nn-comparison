"""Data loading and preprocessing pipeline.

Handles the complete data preparation workflow: file format detection and loading,
column name normalization, missing value imputation, categorical encoding, and
splitting into features and target dataframes. Supports CSV, JSON, and Excel formats.

Functions:
    clean_variable_name(name: str) -> str:
        Convert column names to valid Python identifiers.
    load_dataset(filepath: str) -> pd.DataFrame:
        Load dataset from CSV, JSON, or Excel file.
    load_and_preprocess_data(config: dict) -> tuple:
        Complete preprocessing pipeline: load, impute, encode, and split data.

See Also:
    src.base_model: Uses preprocessed data for model training
    src.model.main: Calls load_and_preprocess_data during training
"""
import os
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_variable_name(name: str) -> str:
    """Convert dataset column names to valid Python identifiers.

    Replaces non-alphanumeric characters with underscores, removes consecutive
    underscores, and strips leading/trailing underscores to produce a valid
    Python variable name suitable for DataFrame column names.

    Args:
        name (str): Original column name from dataset.

    Returns:
        str: Cleaned column name (valid Python identifier).
    """
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace invalid characters with _
    name = re.sub(r'_+', '_', name)  # Remove consecutive underscores
    return name.strip('_')  # Remove leading/trailing underscores


def load_dataset(filepath : str):
    """Load dataset from file into a pandas DataFrame.

    Automatically detects file format by extension (CSV, JSON, Excel) and loads
    the data. Normalizes column names to valid Python identifiers using
    `clean_variable_name()`.

    Args:
        filepath (str): Path to the dataset file.

    Raises:
        FileNotFoundError: If the file at `filepath` does not exist.
        ValueError: If file extension is not .csv, .json, .xlsx, or .xls.

    Supported Formats:
        - CSV: Comma-separated values
        - JSON: JavaScript Object Notation (expects records format)
        - Excel: .xlsx or .xls spreadsheet files

    Returns:
        pd.DataFrame: Loaded dataset with cleaned column names.
    """
    if filepath.endswith('.json'):
        # Read JSON file
        df = pd.read_json(filepath)
    elif filepath.endswith('.csv'):
        # Read CSV file
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        # Read Excel file
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Supported formats: .json, .csv, .xlsx, .xls")

    df.columns = [clean_variable_name(col) for col in df.columns]
    return df


def load_and_preprocess_data(config : dict):
    """Execute complete data loading and preprocessing pipeline.

    Loads dataset, handles missing values using the specified strategy, encodes
    categorical features and targets, and splits data into features (X) and
    targets (y) dataframes. Returns label encoders for target columns for later
    decoding of predictions.

    Args:
        config (dict): Configuration dictionary with keys:
            - dataset_path (str): Path to dataset file
            - target_columns (list[str]): Names of target columns
            - missing_value_strategy (str): Strategy for handling missing values
              ('mean', 'median', 'mode', 'drop'). Default: 'mean'

    Returns:
        tuple: (X_encoded, y_encoded, y_encoders)
            - X_encoded (pd.DataFrame): Features with categorical columns encoded
            - y_encoded (pd.DataFrame): Targets with categorical columns encoded
            - y_encoders (dict[str, LabelEncoder]): Encoders for each target column
              (None for numeric targets; set only for categorical targets)

    Workflow:
        1. Load dataset from file
        2. Handle missing values using specified strategy
        3. Split into features and targets by column names
        4. Label-encode categorical feature columns
        5. Label-encode categorical target columns and store encoders
        6. Return encoded dataframes and encoder dictionary

    Missing Value Strategies:
        - 'mean': Fill with column mean (numeric columns only)
        - 'median': Fill with column median (numeric columns only)
        - 'mode': Fill with column mode
        - 'drop': Remove rows with missing values

    Notes:
        - All categorical columns are label-encoded (0, 1, 2, ...)
        - Store `y_encoders` to decode predictions back to original labels
        - Missing value strategies are applied after loading but before encoding
    """

    # Obtain dataset filepath
    dataset_filepath = config['dataset_path']
    print(f"Dataset: {os.path.basename(dataset_filepath)}")

    # Load the dataset
    df = load_dataset(dataset_filepath)
    print(f'Columns in dataset:\n{df.columns}')

    # Handle missing values
    strategy = config.get('missing_value_strategy', 'mean')
    if strategy == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif strategy == 'median':
        df.fillna(df.median(), inplace=True)
    elif strategy == 'mode':
        df.fillna(df.mode().iloc[0], inplace=True)
    elif strategy == 'drop':
        df.dropna(inplace=True)

    # Split data to features and targets
    target_columns = config['target_columns']
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    print(f'Target columns: {target_columns}')

    # Label-encode categorical columns in X
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object" or X_encoded[col].dtype.name == "category":
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])

    # Encode categorical target columns if needed
    y_encoded = y.copy()
    y_encoders = {}
    for col in y_encoded.columns:
        if not pd.api.types.is_numeric_dtype(y_encoded[col]):
            le = LabelEncoder()
            y_encoded[col] = le.fit_transform(y_encoded[col])
            y_encoders[col] = le

    return X_encoded, y_encoded, y_encoders
