import time
import logging
from contextlib import contextmanager
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

@contextmanager
def timed(logger: logging.Logger, name: str, request_id: str | None = None):
    start = time.perf_counter()

    try:
        yield
    finally:
        end = time.perf_counter() - start
        prefix = f"[{request_id}] " if request_id else ""
        logger.info(f"{prefix}{name} took {end:.3f}s")