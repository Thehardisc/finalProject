import logging
import os
import sys
import json
import traceback
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in structured JSON format."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "service": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Merge extra context if provided
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry)

class CustomLogger:
    """Wrapper around logging.Logger to provide extra helper methods and structured telemetry."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def info(self, msg, *args, **kwargs): self.logger.info(msg, *args, **kwargs)
    def debug(self, msg, *args, **kwargs): self.logger.debug(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs): self.logger.error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs): self.logger.critical(msg, *args, **kwargs)

    def log_exception(self, title: str, e: Exception):
        """Standardized way to log unexpected errors with full stack traces."""
        error_msg = f"{title}: {str(e)}"
        self.logger.error(error_msg, exc_info=True)

    def log_stats(self, title: str, stats: dict, level: int = logging.INFO):
        """Prints a professional-looking boxed table of statistics or structured JSON."""
        # Use simple JSON dump if format is JSON
        if os.environ.get("LOG_FORMAT") == "JSON":
            self.logger.log(level, f"STATS: {title}", extra={"extra_data": stats})
            return

        if not self.logger.isEnabledFor(level):
            return

        width = 60
        header = f" {title} "
        separator = f"+{'-'*(width-2)}+"
        header_line = f"|{header:-^{width-2}}|"
        
        lines = [separator, header_line, separator]
        for key, value in stats.items():
            if isinstance(value, float):
                val_str = f"{value:.4f}"
            else:
                val_str = str(value)
            
            line = f"|  {key:<25} : {val_str:<27}|"
            lines.append(line)
        
        lines.append(separator)
        for line in lines:
            self.logger.log(level, line)

def get_logger(name: str) -> CustomLogger:
    """
    Returns a configured logger instance that outputs to sys.stdout.
    Now configures the ROOT logger on first run to ensure global compliance.
    """
    # 1. Determine global log level from environment
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = logging.INFO

    # 2. Configure ROOT logger once to catch external library logs
    root_logger = logging.getLogger()
    if not getattr(root_logger, '_is_global_configured', False):
        root_logger.setLevel(log_level)
        
        # Add a global stream handler if none exists
        if not root_logger.handlers:
            rh = logging.StreamHandler(sys.stdout)
            rh.setLevel(log_level)
            
            if os.environ.get("LOG_FORMAT") == "JSON":
                rh.setFormatter(JsonFormatter())
            else:
                rh.setFormatter(logging.Formatter(
                    '[%(asctime)s] [%(levelname)-8s] [%(name)-24s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
            root_logger.addHandler(rh)
            
        # Optional: Silence chatty libraries or sync them
        for lib in ["uvicorn", "uvicorn.access", "gunicorn", "transformers", "urllib3", "filelock", "datasets", "fsspec"]:
             lib_logger = logging.getLogger(lib)
             # Force them to INFO to reduce noise, even if we are in DEBUG
             lib_logger.setLevel(logging.INFO)
             lib_logger.propagate = True # Let them bubble up to our root handler
             
        setattr(root_logger, '_is_global_configured', True)

    # 3. Configure the specific named logger
    logger = logging.getLogger(name)
    if getattr(logger, '_is_configured', False):
        # Even if configured, ensure level is synced in case of hot-reload (if supported)
        logger.setLevel(log_level)
        return CustomLogger(logger)

    logger.setLevel(log_level)
    
    # We rely on propagation to the configured ROOT and its handler
    # unless we want specific control/formatting for this service
    logger.propagate = True 
    
    setattr(logger, '_is_configured', True)
    return CustomLogger(logger)

