"""Custom exception types for configuration and explainability pipelines.

Defines domain-specific exceptions used across the training and XAI pipelines
to clearly signal invalid configurations or unsupported operations.
"""


class UnsupportedLossException(Exception):
    """Invalid Loss Function Selection."""


class UnsupportedTaskTypeException(Exception):
    """Invalid Task Type Selection."""


class UnsupportedModelException(Exception):
    """Invalid Model Selection."""


class UnsupportedXaiMethodException(Exception):
    """Invalid XAI Method Selection."""
