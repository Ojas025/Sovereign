"""Central logging: JSONL file output with run/turn correlation ids.

Single owner of logging configuration: every module calls ``get_logger`` and
the CLI calls ``configure_logging`` once at startup. Correlation ids are
module globals so TUI worker threads see them without context plumbing.
"""

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

_run_id: str = ""
_turn_id: str = ""

_LOG_MAX_BYTES = 5 * 1024 * 1024
_LOG_BACKUP_COUNT = 3

_STANDARD_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line carrying correlation ids and any ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if _run_id:
            payload["run_id"] = _run_id
        if _turn_id:
            payload["turn_id"] = _turn_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    *,
    log_dir: Path,
    level: str = "INFO",
    console: bool = False,
    run_id: str | None = None,
) -> str:
    """Attach the JSONL file handler to the ``workbench`` logger; return the run id.

    Reconfiguring replaces previous handlers so a run never double-writes.
    """
    global _run_id
    _run_id = run_id or uuid4().hex[:12]

    level_no = logging.getLevelNamesMapping().get(level.upper())
    if level_no is None:
        raise ValueError(f"invalid log level: {level}")

    log_dir.mkdir(parents=True, exist_ok=True)
    workbench_logger = logging.getLogger("workbench")
    for handler in list(workbench_logger.handlers):
        workbench_logger.removeHandler(handler)
        handler.close()
    workbench_logger.setLevel(level_no)
    workbench_logger.propagate = False

    file_handler = RotatingFileHandler(
        log_dir / f"run-{_run_id}.jsonl",
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    workbench_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        workbench_logger.addHandler(stream_handler)

    return _run_id


def set_turn_id(turn: str | None) -> None:
    """Set (or clear with None) the correlation id stamped on later records."""
    global _turn_id
    _turn_id = turn or ""


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the central ``workbench`` logger."""
    return logging.getLogger(f"workbench.{name}")
