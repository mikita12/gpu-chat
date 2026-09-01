import pytest
from fastapi import HTTPException

from app.api import _check_limits, _trim_to_context
from app.schemas import ChatMessage, ChatRequest

from .helpers import fast_settings


def _msg(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def test_check_limits_rejects_empty_messages() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _check_limits(ChatRequest(messages=[]), fast_settings())
    assert exc_info.value.status_code == 400


def test_check_limits_rejects_too_many_messages() -> None:
    settings = fast_settings(max_messages=2)
    req = ChatRequest(messages=[_msg("user", "a"), _msg("user", "b"), _msg("user", "c")])
    with pytest.raises(HTTPException) as exc_info:
        _check_limits(req, settings)
    assert exc_info.value.status_code == 400


def test_check_limits_rejects_one_message_too_long() -> None:
    settings = fast_settings(max_message_chars=5)
    req = ChatRequest(messages=[_msg("user", "way too long")])
    with pytest.raises(HTTPException) as exc_info:
        _check_limits(req, settings)
    assert exc_info.value.status_code == 400


def test_check_limits_rejects_combined_prompt_too_long() -> None:
    settings = fast_settings(max_message_chars=100, max_prompt_chars=10)
    req = ChatRequest(messages=[_msg("user", "12345"), _msg("assistant", "12345"), _msg("user", "1")])
    with pytest.raises(HTTPException) as exc_info:
        _check_limits(req, settings)
    assert exc_info.value.status_code == 400


def test_check_limits_accepts_within_bounds() -> None:
    settings = fast_settings()
    req = ChatRequest(messages=[_msg("user", "hello")])
    _check_limits(req, settings)  # must not raise


def test_trim_to_context_keeps_system_and_recent_messages() -> None:
    messages = [
        _msg("system", "be nice"),
        _msg("user", "x" * 40),  # oldest - should be dropped
        _msg("assistant", "x" * 40),
        _msg("user", "x" * 40),  # most recent - must survive
    ]
    # budget in tokens = (context_length - reserve) // 4 chars-per-token.
    # Pick a context_length that only leaves room for the system message
    # plus the last one or two non-system messages.
    trimmed = _trim_to_context(messages, context_length=1044)  # reserve=1024 -> 20 token budget -> 80 chars
    assert trimmed[0].role == "system"
    assert trimmed[-1].content == messages[-1].content
    assert len(trimmed) < len(messages)


def test_trim_to_context_always_keeps_latest_message_even_if_oversized() -> None:
    messages = [_msg("user", "x" * 10_000)]
    trimmed = _trim_to_context(messages, context_length=10)
    assert trimmed == messages


def test_trim_to_context_noop_when_everything_fits() -> None:
    messages = [_msg("user", "hi"), _msg("assistant", "hello")]
    trimmed = _trim_to_context(messages, context_length=1_000_000)
    assert trimmed == messages
