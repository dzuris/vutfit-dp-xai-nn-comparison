"""
Module for loading and preprocessing data.

This module obtains configuration and loads dataset from provided file. Furthermore
missing data are preprocessed using missing_value_strategy, target columns preprocessed if
needed and splitted into features and targets data frames.
"""
import os
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_variable_name(name: str) -> str:
    """Convert dataset column names into valid Python identifiers.

    Args:
        name (str): Original name.

    Returns:
        str: Processed name.
    """
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace invalid characters with _
    name = re.sub(r'_+', '_', name)  # Remove consecutive underscores
    return name.strip('_')  # Remove leading/trailing underscores


def load_dataset(filepath : str):
    """Loads dataset from file into DataFrame.

    This method loads data from provided file into DataFrame according to extension of filepath,
    and converts data frame columns names into valid Python identifiers.

    Raises:
        ValueError: If provided datafile extension is not .json, .csv or .xlsx.

    Args:
        filepath (str): File path to dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
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
    """Loads and preprocesses data.

    This function preprocesses loaded dataset, handles missing values, encodes columns in target
    if needed and split data into features and targets data frames.

    Args:
        config: Configuration.

    Returns:
        (pd.DataFrame, pd.DataFrame): Features and targets data frames.
    """
    dataset_filepath = config['dataset_path']

    # Print dataset filename
    print(f"Dataset: {os.path.basename(dataset_filepath)}")

    # Load dataset
    df = load_dataset(dataset_filepath)

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

    print(f'Columns in dataset:\n{df.columns}')
    target_columns = config['target_columns']
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    # Encode categorical target columns if needed
    y_encoded = y.copy()
    for col in y_encoded.columns:
        if not pd.api.types.is_numeric_dtype(y_encoded[col]):
            label_encoder = LabelEncoder()
            y_encoded[col] = label_encoder.fit_transform(y_encoded[col])

    return X, y_encoded
