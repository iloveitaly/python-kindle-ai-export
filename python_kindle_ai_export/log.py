"""Structured logging setup using structlog-config."""

import logging
import os
import sys
import structlog
from structlog_config import LoggerWithContext, configure_logger


def setup_logging() -> LoggerWithContext:
    """Configure structured logging to stderr."""
    if log_path := os.environ.get("KINDLE_EXPORT_LOG_PATH"):
        os.environ["PYTHON_LOG_PATH"] = log_path

    logging.getLogger("google_genai.models").setLevel(logging.ERROR)

    return configure_logger(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr)
    )


log = setup_logging()
