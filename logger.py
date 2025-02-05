import logging
from pathlib import Path

class Logger:
    """Handles logging for all modules, writing output to a specified log file."""

    def __init__(self, log_file: str) -> None:
        """Initializes the logger and ensures the log directory exists."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the Output directory exists

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # File handler to log messages to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter for log messages
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler to logger (avoid duplicate handlers)
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """Logs an INFO message."""
        self.logger.info(message)

    def error(self, message: str) -> None:
        """Logs an ERROR message."""
        self.logger.error(message)

    def warning(self, message: str) -> None:
        """Logs a WARNING message."""
        self.logger.warning(message)
