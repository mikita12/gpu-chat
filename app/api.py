import asyncio
import contextlib
import hmac
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import StreamingResponse

from app import metrics
from app.config import Settings, get_settings
from app.limiter import GenerationLimiter, QueueFullError, Ticket
from app.logging_config import get_logger, request_id_var
from app.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaGenerationError,
    OllamaHTTPError,
    OllamaProtocolError,
)
from app.schemas import (
    ChatMessage,
    ChatRequest,
    ContentEvent,
    DoneEvent,
    ErrorEvent,
    GeneratingEvent,
    LoadedResponse,
    PingEvent,
    QueuedEvent,
    StreamEvent,
    ThinkingEvent,
)


def get_ollama_client(request: Request) -> OllamaClient:
    ollama: OllamaClient = request.app.state.ollama
    return ollama


def get_limiter(request: Request) -> GenerationLimiter:
    limiter: GenerationLimiter = request.app.state.limiter
    return limiter


OllamaDep = Annotated[OllamaClient, Depends(get_ollama_client)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
LimiterDep = Annotated[GenerationLimiter, Depends(get_limiter)]


def require_auth(request: Request, settings: SettingsDep) -> None:
    """No-op when settings.bearer_token is unset (the default) - preserves
    today's zero-friction LAN access. Otherwise requires a matching
    `Authorization: Bearer <token>` header, compared with hmac.compare_digest
    (constant-time) rather than `==` to avoid a timing side-channel."""
    if not settings.bearer_token:
        return
    expected = f"Bearer {settings.bearer_token}"
    provided = request.headers.get("authorization", "")
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail=_detail("unauthorized", "missing or invalid bearer token"))


router = APIRouter(dependencies=[Depends(require_auth)])


def _detail(code: str, message: str) -> dict[str, str]:
    """Structured HTTPException.detail - FastAPI serializes any JSON value,
    not just a string, as {"detail": ...}. Giving every pre-stream error a
    `code` alongside its message means the frontend never has to string-
    sniff human-readable text to classify a failure, matching the same
    typed-code approach already used for in-stream ErrorEvents below."""
    return {"message": message, "code": code}


@router.get("/api/models")
async def list_models(ollama: OllamaDep) -> list[str]:
    try:
        summaries = await ollama.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=_detail("upstream_unavailable", str(exc))) from exc
    return sorted(m.name for m in summaries)


@router.get("/api/loaded")
async def loaded(ollama: OllamaDep, settings: SettingsDep) -> LoadedResponse:
    try:
        running = await ollama.running_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=_detail("upstream_unavailable", str(exc))) from exc
    return LoadedResponse(loaded=[m.name for m in running], default=settings.ollama_model)


def _error_event(exc: OllamaError) -> ErrorEvent:
    request_id = request_id_var.get()
    if isinstance(exc, OllamaGenerationError):
        return ErrorEvent(message=exc.message, code=exc.code, request_id=request_id)
    if isinstance(exc, OllamaHTTPError):
        return ErrorEvent(message=exc.message, code=f"upstream_http_{exc.status_code}", request_id=request_id)
    if isinstance(exc, OllamaConnectionError):
        return ErrorEvent(message="could not reach the model server", code="connection", request_id=request_id)
    if isinstance(exc, OllamaProtocolError):
        return ErrorEvent(
            message="unexpected response from the model server", code="protocol", request_id=request_id
        )
    return ErrorEvent(message=str(exc), code="unknown", request_id=request_id)


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
        # aclosing() guarantees .aclose() runs on this generator - in this
        # same task - on every exit from the block below, including this
        # task being cancelled while blocked on queue.put() (the bounded
        # queue can legitimately block there). Without it, an abandoned
        # generator still holding chat()'s open `async with client.stream()`
        # only gets finalized later by the event loop's async-generator GC
        # hook, in a *different* task than the one that opened the stream -
        # which is exactly httpx/anyio's "cannot exit cancel scope in a
        # different task" failure, and can leak the upstream connection.
        async with contextlib.aclosing(ollama.chat(model, messages)) as stream:
            async for chunk in stream:
                if chunk.message.thinking:
                    await queue.put(ThinkingEvent(text=chunk.message.thinking))
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
                            request_id=request_id_var.get(),
                        )
                    )
    except OllamaError as exc:
        await queue.put(_error_event(exc))
    finally:
        # Best-effort, non-blocking: during a cancellation-driven teardown
        # (the consumer stopped calling queue.get() because *it* is the one
        # shutting things down, e.g. generate_events()'s finally block
        # cancelling us) the queue can be full with nobody left to drain it.
        # A blocking `await queue.put(None)` here would deadlock - a fresh
        # await entered while already unwinding a CancelledError does not
        # get cancelled again on its own. The sentinel only matters to a
        # consumer that's still actively reading; one that isn't doesn't
        # need it.
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(None)


async def generate_events(
    ollama: OllamaClient,
    model: str,
    messages: list[ChatMessage],
    settings: Settings,
    limiter: GenerationLimiter,
    ticket: Ticket,
) -> AsyncGenerator[StreamEvent, None]:
    """Waits for a generation slot (reported via QueuedEvent while waiting),
    then consumes OllamaClient.chat() through a background task + bounded
    queue, so we can interleave heartbeats while genuinely idle without ever
    cancelling the network read itself (which would leave the upstream
    connection to Ollama in an inconsistent state - see _produce/cleanup
    below for the deterministic-cancellation half of that story).

    Deliberately typed as AsyncGenerator (not the narrower AsyncIterator):
    callers - including tests simulating a client disconnect - rely on
    calling .aclose() to trigger the cleanup in the `finally` block below.
    """
    request_id = request_id_var.get()
    logger = get_logger()
    acquired = False
    queue_wait_start = time.monotonic()
    metrics.QUEUE_DEPTH.set(limiter.queue_depth)
    try:
        # Queueing phase: wait our turn. Not subject to the stall watchdog
        # below - waiting for a free slot under load can legitimately take
        # a long time and isn't a "model stopped responding" condition.
        # ticket.acquire_task is polled non-destructively (asyncio.wait,
        # not wait_for) so it never loses its place in the semaphore's FIFO.
        last_position: int | None = None
        while not ticket.acquire_task.done():
            position = limiter.position(ticket)
            if position != last_position:
                yield QueuedEvent(position=position)
                last_position = position
            done, _pending = await asyncio.wait({ticket.acquire_task}, timeout=settings.heartbeat_seconds)
            if ticket.acquire_task not in done:
                yield PingEvent()
        acquired = True
        limiter.mark_running(ticket)
        metrics.QUEUE_WAIT_SECONDS.observe(time.monotonic() - queue_wait_start)
        metrics.QUEUE_DEPTH.set(limiter.queue_depth)
        metrics.ACTIVE_GENERATIONS.inc()
        logger.info("chat request started", extra={"model": model, "message_count": len(messages)})
        # Distinct from a bare ping: pings fire during the queueing phase
        # too, so without this a client that was queued can't tell "still
        # queued, no position change yet" apart from "already generating,
        # model still loading" - both look like a queued event once
        # followed by a run of pings.
        yield GeneratingEvent()

        gen_start = time.monotonic()
        ttft_recorded = False
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue(maxsize=settings.stream_queue_maxsize)
        task = asyncio.create_task(_produce(ollama, model, messages, queue))
        last_progress = time.monotonic()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=settings.heartbeat_seconds)
                except TimeoutError:
                    if time.monotonic() - last_progress >= settings.stall_timeout_seconds:
                        err = ErrorEvent(message="model did not respond in time", code="stall", request_id=request_id)
                        metrics.ERRORS_TOTAL.labels(code=err.code).inc()
                        logger.error("chat request stalled", extra={"code": err.code})
                        yield err
                        return
                    yield PingEvent()
                    continue
                if item is None:
                    return
                last_progress = time.monotonic()
                if isinstance(item, ContentEvent) and not ttft_recorded:
                    metrics.TTFT_SECONDS.observe(time.monotonic() - gen_start)
                    ttft_recorded = True
                elif isinstance(item, DoneEvent):
                    if item.eval_count is not None and item.eval_duration:
                        metrics.TOKENS_PER_SECOND.observe(item.eval_count / (item.eval_duration / 1e9))
                elif isinstance(item, ErrorEvent):
                    metrics.ERRORS_TOTAL.labels(code=item.code).inc()
                    logger.error(
                        "chat request failed", extra={"code": item.code, "error_message": item.message}
                    )
                yield item
                if isinstance(item, DoneEvent | ErrorEvent):
                    return
        finally:
            # Deterministically stop the producer (and, through it, close the
            # httpx stream to Ollama) whenever we stop consuming for any
            # reason - client disconnect, a `return` above, or our own
            # cancellation. A bare `task.cancel()` without awaiting is
            # fire-and-forget: the upstream connection might still be open,
            # and Ollama would keep generating and holding the GPU for a
            # client that already left.
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    finally:
        if acquired:
            metrics.ACTIVE_GENERATIONS.dec()
        await limiter.release(ticket, acquired)


def _check_limits(req: ChatRequest, settings: Settings) -> None:
    """Hard caps, rejected outright - distinct from _trim_to_context below,
    which adapts history to fit rather than rejecting the request."""
    if not req.messages:
        raise HTTPException(status_code=400, detail=_detail("invalid_request", "messages must not be empty"))
    if len(req.messages) > settings.max_messages:
        raise HTTPException(
            status_code=400,
            detail=_detail("invalid_request", f"too many messages (max {settings.max_messages})"),
        )
    for m in req.messages:
        if len(m.content) > settings.max_message_chars:
            raise HTTPException(
                status_code=400,
                detail=_detail("invalid_request", f"message too long (max {settings.max_message_chars} chars)"),
            )
    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > settings.max_prompt_chars:
        raise HTTPException(
            status_code=400,
            detail=_detail("invalid_request", f"prompt too long (max {settings.max_prompt_chars} chars)"),
        )


# Rough heuristic - there's no real tokenizer available for arbitrary
# Ollama models, so this deliberately errs conservative (trims a bit more
# than strictly necessary) rather than trying to be precise.
CHARS_PER_TOKEN_ESTIMATE = 4
RESPONSE_TOKEN_RESERVE = 1024


def _trim_to_context(messages: list[ChatMessage], context_length: int) -> list[ChatMessage]:
    """Keeps all system messages, then as many of the most recent remaining
    messages as fit in the model's context window (minus headroom for its
    own reply), dropping older ones first. Always keeps at least the single
    most recent non-system message, even if it alone exceeds budget - never
    returns an empty conversation. This adapts history to fit rather than
    rejecting the request (see _check_limits for the hard-reject caps)."""
    # If the reserve would eat the whole (tiny) context window, fall back
    # to using the full window as budget rather than trimming to nothing.
    budget = context_length - RESPONSE_TOKEN_RESERVE
    if budget <= 0:
        budget = context_length
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]

    used = sum(len(m.content) for m in system) // CHARS_PER_TOKEN_ESTIMATE
    kept: list[ChatMessage] = []
    for m in reversed(rest):
        cost = len(m.content) // CHARS_PER_TOKEN_ESTIMATE
        if kept and used + cost > budget:
            break
        used += cost
        kept.append(m)
    kept.reverse()
    return system + kept


@router.post("/api/chat")
async def chat(req: ChatRequest, ollama: OllamaDep, settings: SettingsDep, limiter: LimiterDep) -> StreamingResponse:
    _check_limits(req, settings)
    model = req.model or settings.ollama_model
    try:
        available = {m.name for m in await ollama.list_models()}
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=_detail("upstream_unavailable", str(exc))) from exc
    if model not in available:
        raise HTTPException(status_code=400, detail=_detail("unknown_model", f"unknown model: {model!r}"))

    messages = req.messages
    context_length = await ollama.context_length(model)
    if context_length is not None:
        messages = _trim_to_context(messages, context_length)

    try:
        ticket = limiter.reserve()
    except QueueFullError as exc:
        raise HTTPException(
            status_code=429, detail=_detail("queue_full", "server is busy, try again shortly")
        ) from exc

    # asyncio.create_task() (inside generate_events/_produce) snapshots the
    # current contextvars.Context at creation time, so setting this here -
    # before those tasks exist - makes it visible inside them too, with no
    # need to thread request_id through as an explicit parameter.
    request_id_var.set(uuid.uuid4().hex[:12])

    async def stream() -> AsyncIterator[bytes]:
        async for event in generate_events(ollama, model, messages, settings, limiter, ticket):
            yield event.model_dump_json().encode() + b"\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
