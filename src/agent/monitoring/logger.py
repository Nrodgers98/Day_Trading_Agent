"""Structured JSON logging setup for the trading agent."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.agent.config import MonitoringConfig

EASTERN = ZoneInfo("US/Eastern")


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=EASTERN).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            entry["stack_info"] = record.stack_info

        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "relativeCreated",
                "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "pathname", "filename", "module", "levelno", "levelname",
                "thread", "threadName", "process", "processName",
                "msecs", "taskName", "message",
            }:
                entry[key] = value

        return json.dumps(entry, default=str)


def _sanitize_log_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    return safe.strip("_-") or "agent"


def setup_logging(config: MonitoringConfig, log_name: str | None = None) -> logging.Logger:
    """Configure the root logger with JSON-formatted console and rotating file handlers."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("trading_agent")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = JsonFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_basename = "agent.log"
    if log_name:
        file_basename = f"agent_{_sanitize_log_name(log_name)}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, file_basename),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def setup_component_logger(
    config: MonitoringConfig,
    *,
    logger_name: str,
    file_name: str,
    level: str | None = None,
) -> logging.Logger:
    """Create a dedicated JSON logger for a component-specific log file."""
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    component_logger = logging.getLogger(logger_name)
    component_logger.setLevel(getattr(logging, (level or config.log_level).upper(), logging.INFO))
    component_logger.handlers.clear()

    formatter = JsonFormatter()

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, file_name),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    component_logger.addHandler(file_handler)

    component_logger.propagate = False
    return component_logger
