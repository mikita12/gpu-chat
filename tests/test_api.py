import asyncio
import contextlib

import httpx
from fastapi import FastAPI
from httpx import ASGITransport

from app.api import _produce, generate_events, router
from app.config import get_settings
from app.limiter import GenerationLimiter
from app.ollama import OllamaConnectionError, OllamaHTTPError
from app.schemas import (
    ContentEvent,
    ErrorEvent,
    OllamaChatChunk,
    OllamaChatMessageChunk,
    QueuedEvent,
    StreamEvent,
)

from .helpers import FakeOllamaClient, fast_settings


def make_app(
    ollama: FakeOllamaClient, limiter: GenerationLimiter | None = None, bearer_token: str = ""
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.ollama = ollama
    app.state.limiter = limiter or GenerationLimiter(max_concurrent=1, max_queue_size=5)
    if bearer_token:
        settings = get_settings()
        app.dependency_overrides[get_settings] = lambda: settings.model_copy(
            update={"bearer_token": bearer_token}
        )
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
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    ticket = limiter.reserve()
    events = [e async for e in generate_events(fake, "ghost", [], fast_settings(), limiter, ticket)]  # type: ignore[arg-type]
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "upstream_http_404"


async def test_generate_events_maps_connection_error() -> None:
    fake = FakeOllamaClient(chat_error=OllamaConnectionError("refused"))
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    ticket = limiter.reserve()
    events = [e async for e in generate_events(fake, "m", [], fast_settings(), limiter, ticket)]  # type: ignore[arg-type]
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "connection"


async def test_stall_watchdog_fires_when_ollama_goes_silent() -> None:
    # hang_seconds far exceeds stall_timeout_seconds - the fake never yields
    # anything, simulating Ollama accepting the request but never responding.
    fake = FakeOllamaClient(hang_seconds=10.0)
    settings = fast_settings(heartbeat_seconds=0.02, stall_timeout_seconds=0.08)
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    ticket = limiter.reserve()
    events = await asyncio.wait_for(
        _collect(generate_events(fake, "m", [], settings, limiter, ticket)),  # type: ignore[arg-type]
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
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    ticket = limiter.reserve()
    await asyncio.sleep(0)  # let the (uncontended) permit acquire settle first
    agen = generate_events(fake, "m", [], settings, limiter, ticket)  # type: ignore[arg-type]
    first = await agen.__anext__()
    assert isinstance(first, type(first))  # got at least one (a ping) event
    await agen.aclose()  # simulates the client disconnecting mid-stream
    await asyncio.sleep(0.05)
    assert fake.cancelled is True


async def test_producer_cancelled_while_blocked_on_full_queue_closes_ollama_stream() -> None:
    # Regression test for the "cannot exit cancel scope in a different task"
    # class of bug: _produce() must use contextlib.aclosing() around the
    # ollama.chat() generator, not a bare `async for`, so a cancellation that
    # lands while blocked on a full queue.put() (not inside chat()'s own
    # await) still tears the generator down deterministically in this task.
    fake = FakeOllamaClient(
        chunks=[
            OllamaChatChunk(message=OllamaChatMessageChunk(content="one")),
            OllamaChatChunk(message=OllamaChatMessageChunk(content="two")),
        ]
    )
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=1)
    await queue.put(ContentEvent(text="filler"))  # first real put() will block

    task = asyncio.create_task(_produce(fake, "m", [], queue))  # type: ignore[arg-type]
    await asyncio.sleep(0.05)  # let _produce reach the blocked queue.put()
    assert not task.done()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert fake.closed is True


async def test_second_request_sees_queued_position_while_first_runs() -> None:
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    settings = fast_settings(heartbeat_seconds=0.05, stall_timeout_seconds=10.0)

    fake1 = FakeOllamaClient(hang_seconds=0.3)
    ticket1 = limiter.reserve()
    task1 = asyncio.create_task(_collect(generate_events(fake1, "m", [], settings, limiter, ticket1)))  # type: ignore[arg-type]

    await asyncio.sleep(0.02)  # let request 1 actually acquire first
    fake2 = FakeOllamaClient(chunks=[OllamaChatChunk(message=OllamaChatMessageChunk(content="hi"), done=True)])
    ticket2 = limiter.reserve()
    events2 = await asyncio.wait_for(
        _collect(generate_events(fake2, "m", [], settings, limiter, ticket2)),  # type: ignore[arg-type]
        timeout=5.0,
    )

    queued = [e for e in events2 if isinstance(e, QueuedEvent)]
    assert queued and queued[0].position == 1
    assert any(isinstance(e, ContentEvent) for e in events2)

    await task1


async def test_cancel_mid_generation_releases_permit_for_next_request() -> None:
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    settings = fast_settings(heartbeat_seconds=0.02, stall_timeout_seconds=10.0)

    fake1 = FakeOllamaClient(hang_seconds=10.0)  # never finishes on its own
    ticket1 = limiter.reserve()
    agen1 = generate_events(fake1, "m", [], settings, limiter, ticket1)  # type: ignore[arg-type]
    await agen1.__anext__()  # let it acquire the permit and start generating
    await agen1.aclose()  # simulates the client disconnecting mid-generation

    ticket2 = limiter.reserve()
    await asyncio.sleep(0.05)
    # The permit from the cancelled first request must have been released
    # promptly - not leaked, which with max_concurrent=1 would deadlock
    # every future request forever.
    assert ticket2.acquire_task.done() is True


async def test_queue_full_returns_429_without_streaming() -> None:
    # capacity = max_concurrent (1, running) + max_queue_size (1, waiting) = 2
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=1)
    fake = FakeOllamaClient(hang_seconds=10.0)
    app = make_app(fake, limiter=limiter)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"messages": [{"role": "user", "content": "hi"}], "model": "test-model"}
        task1 = asyncio.create_task(client.post("/api/chat", json=payload))
        await asyncio.sleep(0.03)
        task2 = asyncio.create_task(client.post("/api/chat", json=payload))
        await asyncio.sleep(0.03)

        resp3 = await client.post("/api/chat", json=payload)
        assert resp3.status_code == 429

        for task in (task1, task2):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, httpx.RequestError):
                await task


async def test_no_bearer_token_configured_requires_no_header() -> None:
    fake = FakeOllamaClient()
    app = make_app(fake)  # bearer_token left unset - today's default
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/loaded")
    assert resp.status_code == 200


async def test_bearer_token_configured_rejects_missing_header() -> None:
    fake = FakeOllamaClient()
    app = make_app(fake, bearer_token="secret")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/loaded")
    assert resp.status_code == 401


async def test_bearer_token_configured_rejects_wrong_token() -> None:
    fake = FakeOllamaClient()
    app = make_app(fake, bearer_token="secret")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/loaded", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


async def test_bearer_token_configured_accepts_correct_token() -> None:
    fake = FakeOllamaClient()
    app = make_app(fake, bearer_token="secret")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/loaded", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


async def test_chat_rejects_too_many_messages_with_400() -> None:
    fake = FakeOllamaClient()
    app = make_app(fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"messages": [{"role": "user", "content": "hi"}] * 1000, "model": "test-model"}
        resp = await client.post("/api/chat", json=payload)
    assert resp.status_code == 400


async def test_chat_trims_history_to_context_window() -> None:
    fake = FakeOllamaClient(
        chunks=[OllamaChatChunk(message=OllamaChatMessageChunk(content="ok"), done=True)],
        # reserve (1024) dwarfs this, so budget falls back to context_length
        # itself: 15 *tokens*. Each 40-char message costs 40//4=10 tokens,
        # so only the single most recent one fits (10 <= 15, two would be 20).
        context_length_value=15,
    )
    app = make_app(fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "messages": [
                {"role": "user", "content": "x" * 40},
                {"role": "assistant", "content": "x" * 40},
                {"role": "user", "content": "x" * 40},
            ],
            "model": "test-model",
        }
        resp = await client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    assert fake.last_chat_messages is not None
    assert len(fake.last_chat_messages) < 3
    assert fake.last_chat_messages[-1].content == "x" * 40


async def _collect(agen: object) -> list[StreamEvent]:
    return [event async for event in agen]  # type: ignore[attr-defined]
