"""Module containing custom exceptions."""
class UnsupportedLossException(Exception):
    """Invalid Loss Function Selection."""

class UnsupportedTaskTypeException(Exception):
    """Invalid Task Type Selection."""

class UnsupportedModelException(Exception):
    """Invalid Model Selection."""

class UnsupportedXaiMethodException(Exception):
    """Invalid XAI Method Selection."""
