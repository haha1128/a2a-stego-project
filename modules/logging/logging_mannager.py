import logging
import colorlog
import os

color={
    "INFO":"green",
    "WARNING":"yellow",
    "ERROR":"red"
}

class CustomFilter(logging.Filter):
    """Custom filter to filter out logs from HTTP requests and third-party libraries."""
    
    def filter(self, record):
        # Logger name patterns to filter out
        filtered_loggers = [
            'urllib3',
            'requests',
            'httpx',
            'aiohttp',
            'fastapi',
            'uvicorn',
            'werkzeug',
            'tornado',
            'flask',
            'django',
            'gunicorn',
            'waitress',
            'hypercorn',
            'transformers',  # Filter logs from the transformers library
            'torch',         # Filter logs from pytorch
            'datasets',      # Filter logs from the datasets library
            'tokenizers',    # Filter logs from the tokenizers library
        ]
        
        # Check if the logger name contains any of the patterns to be filtered
        for filtered in filtered_loggers:
            if filtered in record.name.lower():
                return False
        
        # Filter messages containing specific keywords
        message = record.getMessage().lower()
        filtered_messages = [
            'http',
            'request',
            'response',
            'get',
            'post',
            'put',
            'delete',
            'status code',
            'connection',
            'socket',
            '200',
            '404',
            '500',
            'LiteLLM'
        ]
        
        for filtered_msg in filtered_messages:
            if filtered_msg in message:
                return False
        
        return True

class LoggingMannager:
    @staticmethod
    def configure_global():
        """
        Globally configures the logging system, all loggers will inherit this configuration.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
         # Set the log level for third-party libraries to WARNING or higher to reduce noise.

        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
        logging.getLogger('LiteLLM').setLevel(logging.WARNING)

        # Console handler (colored)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Add custom filter
        ch.addFilter(CustomFilter())
        
        ch.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s - %(name)s - %(message)s",
            log_colors=color
        ))
        root_logger.addHandler(ch)
        

    @staticmethod
    def get_logger(logger_name: str) -> logging.Logger:
        """
        Gets a logger, which inherits the global configuration without adding new handlers.
        """
        return logging.getLogger(logger_name)

