import logging
import os
from datetime import datetime

def setup_logger(name: str = "MedicalAI", log_file: str = "logs/app.log") -> logging.Logger:
    """Set up and return a configured logger instance."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger '{name}' initialized — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger

# Module-level singleton
logger = setup_logger()
