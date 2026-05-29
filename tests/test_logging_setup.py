import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music2db_server.logging_setup import DefaultTagFilter, get_logger, setup_logging


def test_get_logger_returns_app_child() -> None:
    logger = get_logger("module")

    assert logger.name == "music2db_server.module"


def test_setup_logging_applies_external_levels_without_duplicate_handlers() -> None:
    config = {"level": "INFO", "console": True, "loggers": {"httpx": "WARNING"}}

    logger = setup_logging(config)
    first_count = len(logger.handlers)
    logger = setup_logging(config)

    assert logging.getLogger("httpx").level == logging.WARNING
    assert len(logger.handlers) == first_count


def test_default_tag_filter_adds_state_tag_to_untagged_messages() -> None:
    record = logging.LogRecord(
        name="music2db_server.server",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    assert DefaultTagFilter().filter(record) is True
    assert record.msg == "`state` hello"
    assert record.args == ()
