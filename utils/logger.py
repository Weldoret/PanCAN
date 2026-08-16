"""Logging helpers."""

import logging
from pathlib import Path


def setup_logger(name="pancan", log_dir=None, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper()))
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(name="pancan"):
    return logging.getLogger(name)


def log_metrics(logger, metrics, prefix=""):
    logger.info("%s%s", prefix, ", ".join(f"{key}={value}" for key, value in metrics.items()))


def log_config(logger, config):
    logger.info("config=%s", config.to_dict() if hasattr(config, "to_dict") else config)
