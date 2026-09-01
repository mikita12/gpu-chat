import json
import logging

from app.logging_config import JSONFormatter, request_id_var


def _make_record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="gpu_chat", level=logging.INFO, pathname=__file__, lineno=1, msg=msg, args=(), exc_info=None
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_format_produces_valid_json_with_core_fields() -> None:
    formatter = JSONFormatter()
    line = formatter.format(_make_record("hello world"))
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "gpu_chat"
    assert "timestamp" in payload


def test_format_omits_request_id_when_unset() -> None:
    token = request_id_var.set(None)
    try:
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert "request_id" not in payload
    finally:
        request_id_var.reset(token)


def test_format_includes_request_id_when_set() -> None:
    token = request_id_var.set("abc123")
    try:
        payload = json.loads(JSONFormatter().format(_make_record()))
        assert payload["request_id"] == "abc123"
    finally:
        request_id_var.reset(token)


def test_format_includes_extra_fields() -> None:
    payload = json.loads(JSONFormatter().format(_make_record("failed", code="stall", error_message="boom")))
    assert payload["code"] == "stall"
    assert payload["error_message"] == "boom"
