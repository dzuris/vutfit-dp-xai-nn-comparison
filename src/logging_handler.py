"""Module provides Logging Handler.

This module contains wrap for Logging, working as defying the handler with unique name and then
logging into a file using one add_log command.
"""
import os
import logging
import datetime


class LoggerHandler(): # pylint: disable=too-few-public-methods
    """Logger Handler prints logs into file obtained from config.

    Attributes:
        logger: Logger object.

    Methods:
        add_log (message: str, level: str):
            Prints log message into the log file.
    """
    def __init__( # pylint: disable=too-many-arguments, too-many-positional-arguments
            self,
            config: dict,
            logger_name: str,
            file: str,
            model_type: str,
            program_type: str
        ):
        """Logger Handler constructor

        This method initializes the logger and prints some basic info about the task.

        Args:
            config (dict): Logger configuration settings.
            logger_name (str): Name of the logger, needs to be distinct for each logger.
            file (str): Dataset's filename.
            model_type (str): Model type (NN or GP).
            program_type (str): Program's type (e.g. Training, XAI...)
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
        """Prints log message to the logger file.

        Args:
            message (str): Log message.
            level (str, optional): Level of the message. Defaults to 'INFO'.
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
