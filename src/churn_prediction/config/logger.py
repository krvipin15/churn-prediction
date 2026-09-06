"""Structured Application Logging Module.

Implements a centralized logging system that provides formatted
console output and rotating file logs. It is designed to be thread-safe
and prevents duplicate handler initialization in concurrent environments.
"""

import logging
import logging.handlers
import sys
import threading
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import sentry_sdk
import structlog
from sentry_sdk.integrations.logging import LoggingIntegration

from churn_prediction.config.settings import get_settings


class LoggingLifecycle:
    """Track the process-local state of logging configuration.

    This internal helper provides thread-safe state management for the
    application logging lifecycle. It prevents multiple concurrent callers
    from configuring the logging system more than once.

    Attributes
    ----------
    _lock : threading.Lock
        Lock protecting logging initialization from concurrent access.
    _is_configured : bool
        Whether application-wide logging has been successfully configured.
    """

    _lock = threading.Lock()
    _is_configured = False

    @classmethod
    def is_configured(cls) -> bool:
        """Return whether application logging has been configured.

        Returns
        -------
        bool
            ``True`` when logging configuration has completed successfully;
            otherwise ``False``.
        """
        return cls._is_configured

    @classmethod
    def mark_configured(cls) -> None:
        """Mark application logging as successfully configured."""
        cls._is_configured = True


def _init_sentry(dsn: str, environment: str) -> None:
    """Initialize Sentry error monitoring for the application.

    Configures the Sentry SDK with the application's logging integration.
    Informational log records are retained as breadcrumbs, while error-level
    records are captured as Sentry events.

    Parameters
    ----------
    dsn : str
        Sentry Data Source Name identifying the Sentry project.
    environment : str
        Runtime environment associated with captured events, such as
        ``"development"``, ``"staging"``, or ``"production"``.
    """
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        integrations=[
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            )
        ],
        attach_stacktrace=True,
        send_default_pii=False,
    )


def _add_callsite_to_message(
    logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add source filename and line number to a log event.

    Extracts the originating call site from the logging record and prepends
    the filename and line number to the event message. This provides
    additional context when diagnosing application behavior from logs.

    Parameters
    ----------
    logger : structlog.stdlib.BoundLogger
        Logger instance that emitted the event.
    method_name : str
        Name of the logging method that generated the event.
    event_dict : dict[str, Any]
        Structured logging event dictionary to enrich.

    Returns
    -------
    dict[str, Any]
        The updated event dictionary containing call-site information.
    """
    _, _ = logger, method_name  # Unused parameters
    filename = event_dict.pop("filename", "unknown")
    lineno = event_dict.pop("lineno", "0")

    # Get the original message
    event = event_dict.get("event", "")

    # Re-insert it at the front of the message
    event_dict["event"] = f"{filename}:{lineno} - {event}"
    return event_dict


def _get_shared_processors() -> list[Any]:
    """Create the shared Structlog processing pipeline.

    Builds the ordered processor chain used by both Structlog and standard
    library logging integrations. The processors enrich events with logger
    metadata, context variables, log levels, timestamps, stack information,
    exception details, and Unicode normalization.

    Returns
    -------
    list[Any]
        Ordered Structlog processors shared across logging integrations.
    """
    return [
        # Add callsite information (filename and line number)
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        _add_callsite_to_message,
        # Interpolate standard %s and %d positional arguments into the message
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Merge context variables for structured logging
        structlog.contextvars.merge_contextvars,
        # Add standard library log level
        structlog.stdlib.add_log_level,
        # Add timestamp to the event dictionary
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Add stack info to the event dictionary
        structlog.processors.StackInfoRenderer(),
        # Add exception info to the event dictionary
        structlog.processors.format_exc_info,
        # Decode bytes to Unicode for consistent output
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging(*, force: bool = False) -> None:
    """Configure application-wide structured logging.

    Initializes Structlog, console output, rotating file logging,
    third-party logger handling, standard warning capture, and optional
    Sentry error monitoring.

    Configuration is idempotent by default. If logging has already been
    configured, subsequent calls have no effect unless ``force=True`` is
    specified.

    Parameters
    ----------
    force : bool, default=False
        Whether to rebuild the logging configuration when logging has already
        been initialized. When ``True``, existing root handlers are removed
        before configuration is recreated.
    """
    if LoggingLifecycle.is_configured() and not force:
        return

    with LoggingLifecycle._lock:
        if LoggingLifecycle.is_configured() and not force:
            return

        # Initialize the settings and determine the environment
        settings = get_settings()
        is_production = settings.ENVIRONMENT.value == "production"

        # 0. Initialize Sentry
        if settings.SENTRY_DSN and is_production:
            _init_sentry(str(settings.SENTRY_DSN), settings.ENVIRONMENT.value)

        shared_processors = _get_shared_processors()

        # 1. Configure structlog engine
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # 2. Choose renderer setup
        console_renderer = (
            structlog.processors.JSONRenderer()
            if is_production
            else structlog.dev.ConsoleRenderer(colors=True)
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    console_renderer,
                ],
            )
        )

        # 3. Configure rotating file handler setup
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(
                settings.LOGS_DIR
                / (f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:4]}" + ".log")
            ),
            when=settings.LOG_ROTATION_WHEN,
            interval=1,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
        )

        # 4. Attach handlers to root logger & clear default handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(settings.LOG_LEVEL.value.upper())

        # 5. Clear handlers for third-party loggers to avoid duplicate logs
        for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
            third_party_logger = logging.getLogger(logger_name)
            third_party_logger.handlers.clear()
            third_party_logger.propagate = True

        # Redirect standard library warnings to logging ecosystem
        logging.captureWarnings(capture=True)

        LoggingLifecycle.mark_configured()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured application logger.

    Logging configuration is initialized lazily if it has not already been
    configured.

    Parameters
    ----------
    name : str or None, default=None
        Logger name to bind to the returned logger. If ``None``, the default
        application logger name configured in :class:`Settings` is used.

    Returns
    -------
    structlog.stdlib.BoundLogger
        Configured Structlog logger suitable for structured application
        logging.
    """
    if not LoggingLifecycle.is_configured():
        configure_logging()

    settings = get_settings()
    logger_name = name or settings.LOGGER_NAME
    return structlog.get_logger(logger_name)
