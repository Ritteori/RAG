import logging
import os

def setup_logger(name="RAG"):
    logger = logging.getLogger(name)

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)

    handler = logging.FileHandler('logs.log')
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
