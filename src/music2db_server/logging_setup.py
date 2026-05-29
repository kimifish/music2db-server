from __future__ import annotations

import logging
from typing import Any

APP_NAME = "music2db_server"
DEFAULT_TAG = "state"


class DefaultTagFilter(logging.Filter):
    """Adds a default cyberlog tag to app logs that do not have one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith(APP_NAME):
            return True

        message = record.getMessage()
        if message.startswith("`"):
            return True

        record.msg = f"`{DEFAULT_TAG}` {message}"
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    if name == APP_NAME or name.startswith(f"{APP_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{APP_NAME}.{name}")


def setup_logging(config: dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger(APP_NAME)
    logger.handlers.clear()
    logger.filters.clear()
    logger.propagate = False

    try:
        from cyberlog import LoggingConfig as CyberlogLoggingConfig
        from cyberlog import setup_logger

        cyberlog_config = CyberlogLoggingConfig(**config)
        logger = setup_logger(
            cyberlog_config,
            APP_NAME,
            clear_handlers=True,
            propagate=False,
        )
    except Exception:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(_level(config.get("level", "INFO")))

    logger.filters.clear()
    logger.addFilter(DefaultTagFilter())
    apply_external_logger_levels(config.get("loggers", {}))
    route_uvicorn_logs()
    return logger


def apply_external_logger_levels(loggers: dict[str, str]) -> None:
    for logger_name, level_name in loggers.items():
        logging.getLogger(logger_name).setLevel(_level(level_name))


def route_uvicorn_logs() -> None:
    try:
        import uvicorn.config
    except Exception:
        return

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger_config = uvicorn.config.LOGGING_CONFIG.setdefault("loggers", {}).setdefault(
            logger_name, {}
        )
        logger_config["handlers"] = []
        logger_config["propagate"] = True

        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True


def _level(level_name: object) -> int:
    return getattr(logging, str(level_name).upper(), logging.INFO)
