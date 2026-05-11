"""
app/utils/logging.py
====================
structlog configuration for the EEG Research Copilot.

Provides two rendering modes:
  - JSON (production): machine-readable, one event per line, compatible with
    Datadog / CloudWatch / Loki log ingestion pipelines.
  - ConsoleRenderer (development): human-readable, colour-coded output.

The `configure_logging()` function is called exactly once from the FastAPI
lifespan startup hook in `app/main.py`.

Usage
-----
    import structlog
    logger = structlog.get_logger(__name__)
    logger.info("retrieval_complete", chunks=5, latency_ms=42.1)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


# ──────────────────────────────────────────────────────────────────────────────
# Custom processors
# ──────────────────────────────────────────────────────────────────────────────


def _drop_color_message_key(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,       # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """
    Uvicorn adds a 'color_message' key for terminal output; strip it from JSON
    logs so downstream parsers receive clean records.
    """
    event_dict.pop("color_message", None)
    return event_dict


def _add_log_level(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Normalise the log level name to uppercase (structlog uses lowercase)."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _stringify_exc_info(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,       # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """
    If exc_info is a tuple, convert it to a formatted string so JSON logs
    contain the full traceback as a single-line string field.
    """
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        import traceback
        if isinstance(exc_info, tuple):
            formatted = "".join(traceback.format_exception(*exc_info)).strip()
        elif exc_info is True:
            import sys as _sys
            formatted = "".join(
                traceback.format_exception(*_sys.exc_info())
            ).strip()
        else:
            formatted = str(exc_info)
        event_dict["exception"] = formatted
    return event_dict


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
) -> None:
    """
    Configure structlog and the stdlib root logger for the application.

    This function is idempotent; calling it multiple times has no effect
    beyond re-setting the same configuration.

    Parameters
    ----------
    log_level:
        Minimum log level string (DEBUG | INFO | WARNING | ERROR | CRITICAL).
    json_logs:
        True  → JSONRenderer  (production: Datadog / CloudWatch compatible)
        False → ConsoleRenderer (development: coloured, human-readable)
    """
    # ── 1. Shared processors applied to every log event ───────────────────────
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_log_level,
        _drop_color_message_key,
        _stringify_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
        structlog.processors.StackInfoRenderer(),
    ]

    # ── 2. Final renderer ─────────────────────────────────────────────────────
    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # ── 3. Configure structlog ────────────────────────────────────────────────
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ── 4. Bridge stdlib logging → structlog ─────────────────────────────────
    # Uvicorn, FastAPI, httpx, chromadb all emit stdlib logs.
    # This bridge re-routes them through our structlog pipeline.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(log_level.upper()),
    )
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        _logger = logging.getLogger(name)
        _logger.handlers.clear()
        _logger.propagate = True


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a structlog BoundLogger pre-bound with the module name.

    Preferred over calling structlog.get_logger() directly so that the
    `logger_name` key is always populated consistently.

    Example
    -------
        from app.utils.logging import get_logger
        log = get_logger(__name__)
        log.info("node_complete", node="reranking", n_chunks=5)
    """
    return structlog.get_logger(name)


def bind_request_context(request_id: str) -> None:
    """
    Bind a request_id into the structlog context-var context.

    All subsequent log calls on this async task will automatically include
    the request_id key without manual passing.

    Call this at the top of each request handler or middleware.

    Parameters
    ----------
    request_id:
        The UUID trace ID from QueryRequest / X-Request-ID header.
    """
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """
    Clear all structlog context-var bindings for the current async context.

    Call this in a finally block or middleware teardown to prevent context
    leaking between requests in the same worker.
    """
    structlog.contextvars.clear_contextvars()
