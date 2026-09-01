import contextvars
import json
import logging
from datetime import UTC, datetime
from typing import Any

# Set once per request (in app/api.py's POST /api/chat handler) before the
# StreamingResponse is returned. asyncio.create_task() snapshots the current
# contextvars.Context at creation time, so this value is already visible
# inside generate_events() and the _produce() child task it spawns - no
# need to thread request_id through as an explicit function parameter.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

LOGGER_NAME = "gpu_chat"

# Attribute names a bare LogRecord already has - anything else set on a
# record (via logging.info(..., extra={...})) is caller-supplied context
# that should make it into the JSON output, not be silently dropped.
_STANDARD_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
}


class JSONFormatter(logging.Formatter):
    """One JSON object per line, so a failure can be grepped straight out
    of the logs - request_id ties a log line back to the stream events
    (DoneEvent/ErrorEvent) the same request emitted to the client."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id
        for key, value in vars(record).items():
            if key not in _STANDARD_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Attaches the JSON formatter to a dedicated app-level logger, not the
    root/uvicorn loggers - uvicorn manages its own access/error logging via
    its own dictConfig, and fighting that isn't worth it for what this
    needs: one structured log stream for this app's own request lifecycle
    events."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
