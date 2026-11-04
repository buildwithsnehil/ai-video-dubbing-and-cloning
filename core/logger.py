import logging
import os
import sys
from pathlib import Path


def _resolve_log_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_clean = level.upper().strip()
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if level_clean in levels:
            return levels[level_clean]
    
    env_level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    return getattr(logging, env_level, logging.INFO)


def setup_logger(
    name: str = "avatarsync",
    log_file: str | None = None,
    level: int | str | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    resolved_level = _resolve_log_level(level)
    logger.setLevel(resolved_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        target_file = log_file or os.getenv("LOG_FILE", "logs/api.log")
        if target_file:
            log_path = Path(target_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(resolved_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger