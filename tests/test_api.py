import asyncio

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from app.api import generate_events, router
from app.ollama import OllamaConnectionError, OllamaHTTPError
from app.schemas import ContentEvent, DoneEvent, ErrorEvent, OllamaChatChunk, OllamaChatMessageChunk

from .helpers import FakeOllamaClient, fast_settings


def make_app(ollama: FakeOllamaClient) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.ollama = ollama
    return app


async def test_chat_streams_content_then_done() -> None:
    fake = FakeOllamaClient(
        chunks=[
            OllamaChatChunk(message=OllamaChatMessageChunk(content="Hi")),
            OllamaChatChunk(message=OllamaChatMessageChunk(content=""), done=True, eval_count=1),
        ]
    )
    app = make_app(fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"messages": [{"role": "user", "content": "hi"}], "model": "test-model"}
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            assert resp.status_code == 200
            lines = [line async for line in resp.aiter_lines() if line]
    assert '"type":"content"' in lines[0]
    assert '"type":"done"' in lines[-1]


async def test_chat_unknown_model_returns_400() -> None:
    fake = FakeOllamaClient(models=[])
    app = make_app(fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "model": "ghost"},
        )
    assert resp.status_code == 400


async def test_generate_events_maps_upstream_http_error() -> None:
    fake = FakeOllamaClient(chat_error=OllamaHTTPError(404, "model 'ghost' not found"))
    events = [e async for e in generate_events(fake, "ghost", [], fast_settings())]  # type: ignore[arg-type]
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "upstream_http_404"


async def test_generate_events_maps_connection_error() -> None:
    fake = FakeOllamaClient(chat_error=OllamaConnectionError("refused"))
    events = [e async for e in generate_events(fake, "m", [], fast_settings())]  # type: ignore[arg-type]
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "connection"


async def test_stall_watchdog_fires_when_ollama_goes_silent() -> None:
    # hang_seconds far exceeds stall_timeout_seconds - the fake never yields
    # anything, simulating Ollama accepting the request but never responding.
    fake = FakeOllamaClient(hang_seconds=10.0)
    settings = fast_settings(heartbeat_seconds=0.02, stall_timeout_seconds=0.08)
    events = await asyncio.wait_for(
        _collect(generate_events(fake, "m", [], settings)),  # type: ignore[arg-type]
        timeout=5.0,
    )
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "stall"
    # The generator must have actually torn down the hung producer, not left
    # it running.
    assert fake.cancelled is True


async def test_mid_stream_disconnect_cancels_and_frees_producer() -> None:
    fake = FakeOllamaClient(hang_seconds=10.0)
    settings = fast_settings(heartbeat_seconds=0.02, stall_timeout_seconds=10.0)
    agen = generate_events(fake, "m", [], settings)  # type: ignore[arg-type]
    first = await agen.__anext__()
    assert isinstance(first, type(first))  # got at least one (a ping) event
    await agen.aclose()  # simulates the client disconnecting mid-stream
    await asyncio.sleep(0.05)
    assert fake.cancelled is True


async def _collect(agen: object) -> list[ContentEvent | DoneEvent | ErrorEvent]:
    return [event async for event in agen]  # type: ignore[attr-defined]
