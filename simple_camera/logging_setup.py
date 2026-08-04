import logging
from logging.handlers import (
    QueueHandler,
    QueueListener,
    RotatingFileHandler,
)
from pathlib import Path
from typing import Any


def configure_queue_logging(
    log_queue: Any,
) -> None:
    """
    Configure the current process to send all Python
    logs to the central multiprocessing queue.
    """
    root_logger = logging.getLogger()

    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    root_logger.addHandler(
        QueueHandler(log_queue)
    )

    # Route Uvicorn through the same logging queue.
    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ):
        uvicorn_logger = logging.getLogger(
            logger_name
        )
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    logging.captureWarnings(True)


def start_log_listener(
    log_queue: Any,
    log_directory: Path,
) -> QueueListener:
    """
    Start one listener in the parent process.
    Only this listener writes to the log files.
    """
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(processName)s:%(process)d | "
        "%(name)s | %(message)s"
    )

    app_handler = RotatingFileHandler(
        filename=log_directory / "app.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
        delay=True,
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        filename=log_directory / "errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
        delay=True,
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    listener = QueueListener(
        log_queue,
        app_handler,
        error_handler,
        console_handler,
        respect_handler_level=True,
    )

    listener.start()

    return listener