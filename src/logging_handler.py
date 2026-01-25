"""Logging handler for model training and XAI pipelines.

Provides a simplified interface to Python's logging system, creating file-based
loggers with unique names for different pipeline stages (training, XAI, explanation).
Automatically handles log directory creation and appends context metadata (timestamp,
dataset filename, model type, program type) to each log file.

Usage:
    from src.logging_handler import LoggerHandler
    
    config = {
        'level': 'INFO',
        'file': 'logs/training_log.txt',
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }
    logger = LoggerHandler(
        config=config,
        logger_name="TrainingLogger",
        file="iris.csv",
        model_type="NeuralNetwork",
        program_type="Training"
    )
    logger.add_log("Starting training...")
    logger.add_log("Error occurred", level="ERROR")

See Also:
    src.model.main: Uses LoggerHandler for training logs
    src.xai.main: Uses LoggerHandler for XAI logs
"""
import os
import logging
import datetime


class LoggerHandler(): # pylint: disable=too-few-public-methods
    """File-based logging handler with automatic context metadata.

    Wraps Python's `logging` module to simplify log file creation and writing.
    Initializes a logger with a unique name, creates the log directory if needed,
    and appends header information (timestamp, dataset, model type, program type)
    to the log file.

    Attributes:
        logger (logging.Logger): Underlying Python logger instance.

    Methods:
        add_log(message: str, level: str = "INFO"):
            Write a log message at the specified level (INFO, WARNING, ERROR, etc.).
    """
    def __init__( # pylint: disable=too-many-arguments, too-many-positional-arguments
            self,
            config: dict,
            logger_name: str,
            file: str,
            model_type: str,
            program_type: str
        ):
        """Initialize the logger and write header metadata.

        Creates or retrieves a logger with the specified name, configures it with
        a file handler, and writes initial context information to the log file.

        Args:
            config (dict): Logger configuration with keys:
                - level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
                - file (str): Path to the log file (directory created if missing)
                - format (str): Log message format string (e.g., '%(asctime)s - %(message)s')
            logger_name (str): Unique identifier for this logger instance (must be distinct
                              across concurrent loggers).
            file (str): Dataset filename (for logging context, not the log file path).
            model_type (str): Model identifier ('NeuralNetwork', 'GeneticProgramming').
            program_type (str): Pipeline stage ('Training', 'XAI', 'SLS').

        Side Effects:
            - Creates log directory (from config['file']) if it doesn't exist
            - Clears any existing handlers for the specified logger_name
            - Writes header lines to the log file (timestamp, file, model, program type)
        """
        level = config['level']
        filename = config['file']
        directory_path = os.path.dirname(filename)
        os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists

        # Create and configure the logger
        self.logger = logging.getLogger(logger_name) # Unique name for the logger
        self.logger.setLevel(
            getattr(logging, level)) # Convert level string to logging constant

        # Clear any existing handlers for this logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create file handler
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(logging.Formatter(config['format']))

        # Add the handler to the logger
        self.logger.addHandler(file_handler)

        # Initialize info
        self.logger.info('')
        self.logger.info('--------- START ---------')
        self.logger.info("Timestamp: %s", datetime.datetime.now())
        self.logger.info("File: %s", file)
        self.logger.info("Model: %s", model_type)
        self.logger.info("Program type: %s", program_type)

    def add_log(self, message: str, level: str = "INFO"):
        """Write a log message at the specified severity level.

        Routes the message to the appropriate logging method (info, warning, error, etc.)
        based on the `level` parameter. If the level is invalid, defaults to INFO.

        Args:
            message (str): Log message content.
            level (str, optional): Severity level. Valid values:
                'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
                Default: 'INFO'.
        
        Notes:
            - Case-insensitive (e.g., 'info', 'Info', 'INFO' all work)
            - Messages are appended to the log file specified in config
            - Formatting is determined by the format string in config
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
