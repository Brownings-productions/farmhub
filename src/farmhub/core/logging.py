"""structlog setup: JSON lines to a file, pretty output when stderr is a tty."""

import logging
import sys
from pathlib import Path
from typing import TextIO

import structlog

_HANDLER_PREFIX = "farmhub-"


def configure_logging(
    level: str = "INFO", log_file: Path | None = None, stream: TextIO | None = None
) -> None:
    """Configure structlog and the stdlib root logger.

    Safe to call repeatedly: handlers installed by an earlier call are replaced.
    Console output is pretty on a tty and JSON otherwise (so journald and pipes stay
    machine-readable).
    """
    out = stream if stream is not None else sys.stderr
    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    def formatter(renderer: structlog.typing.Processor) -> structlog.stdlib.ProcessorFormatter:
        return structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )

    root = logging.getLogger()
    for handler in list(root.handlers):
        if (handler.get_name() or "").startswith(_HANDLER_PREFIX):
            root.removeHandler(handler)
            handler.close()

    tty = out.isatty()
    console = logging.StreamHandler(out)
    console.set_name(f"{_HANDLER_PREFIX}console")
    console.setFormatter(
        formatter(
            structlog.dev.ConsoleRenderer(colors=tty)
            if tty
            else structlog.processors.JSONRenderer()
        )
    )
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.set_name(f"{_HANDLER_PREFIX}file")
        file_handler.setFormatter(formatter(structlog.processors.JSONRenderer()))
        root.addHandler(file_handler)

    root.setLevel(level)


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """Return a bound logger. Bind per-request context with structlog.contextvars."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
