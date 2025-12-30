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
    def __init__(self, config: dict, logger_name: str):
        """Logger Handler constructor

        This methods sets the logger name, level, clear handlers if neccessarry,
        sets file_handler and sets formatter.

        Args:
            config (dict): Configuration settings.
            logger_name (str): Name of the logger, needs to be distinct for each logger.
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
        self.logger.info('--- START ---')
        self.logger.info("Timestamp: %s", datetime.datetime.now())

    def add_log(self, message: str, level: str = "INFO"):
        """Prints log message to the logger file.

        Args:
            message (str): Log message.
            level (str, optional): Level of the message. Defaults to 'INFO'.
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
