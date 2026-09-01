import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import StreamingResponse

from app.config import Settings, get_settings
from app.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPError,
    OllamaProtocolError,
)
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ContentEvent,
    DoneEvent,
    ErrorEvent,
    LoadedResponse,
    PingEvent,
    StreamEvent,
)

router = APIRouter()


def get_ollama_client(request: Request) -> OllamaClient:
    ollama: OllamaClient = request.app.state.ollama
    return ollama


OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/api/models")
async def list_models(ollama: OllamaDep) -> list[str]:
    try:
        summaries = await ollama.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return sorted(m.name for m in summaries)


@router.get("/api/loaded")
async def loaded(ollama: OllamaDep, settings: SettingsDep) -> LoadedResponse:
    try:
        running = await ollama.running_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return LoadedResponse(loaded=[m.name for m in running], default=settings.ollama_model)


def _error_event(exc: OllamaError) -> ErrorEvent:
    if isinstance(exc, OllamaHTTPError):
        return ErrorEvent(message=exc.message, code=f"upstream_http_{exc.status_code}")
    if isinstance(exc, OllamaConnectionError):
        return ErrorEvent(message="could not reach the model server", code="connection")
    if isinstance(exc, OllamaProtocolError):
        return ErrorEvent(message="unexpected response from the model server", code="protocol")
    return ErrorEvent(message=str(exc), code="unknown")


async def _produce(
    ollama: OllamaClient,
    model: str,
    messages: list[ChatMessage],
    queue: asyncio.Queue[StreamEvent | None],
) -> None:
    """Runs as a background task. Pushes typed events into the queue and
    always pushes a final None sentinel, even on error - the consumer relies
    on that to know when to stop rather than ever waiting forever."""
    try:
        async for chunk in ollama.chat(model, messages):
            if chunk.message.content:
                await queue.put(ContentEvent(text=chunk.message.content))
            if chunk.done:
                await queue.put(
                    DoneEvent(
                        eval_count=chunk.eval_count,
                        eval_duration=chunk.eval_duration,
                        prompt_eval_count=chunk.prompt_eval_count,
                        prompt_eval_duration=chunk.prompt_eval_duration,
                        load_duration=chunk.load_duration,
                        total_duration=chunk.total_duration,
                    )
                )
    except OllamaError as exc:
        await queue.put(_error_event(exc))
    finally:
        await queue.put(None)


async def generate_events(
    ollama: OllamaClient,
    model: str,
    messages: list[ChatMessage],
    settings: Settings,
) -> AsyncGenerator[StreamEvent, None]:
    """Consumes OllamaClient.chat() through a background task + bounded
    queue, so we can interleave heartbeats while genuinely idle without ever
    cancelling the network read itself (which would leave the upstream
    connection to Ollama in an inconsistent state - see _produce/cleanup
    below for the deterministic-cancellation half of that story).

    Deliberately typed as AsyncGenerator (not the narrower AsyncIterator):
    callers - including tests simulating a client disconnect - rely on
    calling .aclose() to trigger the cleanup in the `finally` block below.
    """
    queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=settings.stream_queue_maxsize)
    task = asyncio.create_task(_produce(ollama, model, messages, queue))
    last_progress = time.monotonic()
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=settings.heartbeat_seconds)
            except TimeoutError:
                if time.monotonic() - last_progress >= settings.stall_timeout_seconds:
                    yield ErrorEvent(message="model did not respond in time", code="stall")
                    return
                yield PingEvent()
                continue
            if item is None:
                return
            last_progress = time.monotonic()
            yield item
            if isinstance(item, DoneEvent | ErrorEvent):
                return
    finally:
        # Deterministically stop the producer (and, through it, close the
        # httpx stream to Ollama) whenever we stop consuming for any reason -
        # client disconnect, a `return` above, or our own cancellation. A
        # bare `task.cancel()` without awaiting is fire-and-forget: the
        # upstream connection might still be open, and Ollama would keep
        # generating and holding the GPU for a client that already left.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@router.post("/api/chat")
async def chat(req: ChatRequest, ollama: OllamaDep, settings: SettingsDep) -> StreamingResponse:
    model = req.model or settings.ollama_model
    try:
        available = {m.name for m in await ollama.list_models()}
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if model not in available:
        raise HTTPException(status_code=400, detail=f"unknown model: {model!r}")

    async def stream() -> AsyncIterator[bytes]:
        async for event in generate_events(ollama, model, req.messages, settings):
            yield event.model_dump_json().encode() + b"\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
