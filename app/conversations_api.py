import contextlib
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import StreamingResponse

from app.api import OllamaDep, _trim_to_context, generate_events
from app.auth import CurrentUser, DbDep, SettingsDep
from app.limiter import GenerationLimiter, QueueFullError
from app.logging_config import request_id_var
from app.models import Conversation, Message, User, utcnow
from app.schemas import (
    ChatMessage,
    ContentEvent,
    ConversationDetailOut,
    ConversationOut,
    CreateConversationRequest,
    DoneEvent,
    MessageOut,
    PostMessageRequest,
    StreamEvent,
    ThinkingEvent,
)

router = APIRouter(prefix="/api/conversations")

TITLE_PREVIEW_CHARS = 60


def _get_limiter(request: Request) -> GenerationLimiter:
    limiter: GenerationLimiter = request.app.state.limiter
    return limiter


async def _load_owned_conversation(db: DbDep, user: User, conversation_id: int) -> Conversation:
    """Ownership is enforced in the query itself, not fetched-then-checked -
    a conversation that doesn't exist and one that belongs to someone else
    both come back 404, so existence isn't leaked either way."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.owner_id == user.id)
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail={"message": "conversation not found", "code": "not_found"})
    return conversation


@router.post("", response_model=ConversationOut, status_code=201)
async def create_conversation(
    req: CreateConversationRequest, user: CurrentUser, db: DbDep, settings: SettingsDep
) -> ConversationOut:
    conversation = Conversation(owner_id=user.id, title=req.title, model=req.model or settings.ollama_model)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return ConversationOut.model_validate(conversation, from_attributes=True)


@router.get("", response_model=list[ConversationOut])
async def list_conversations(user: CurrentUser, db: DbDep) -> list[ConversationOut]:
    result = await db.execute(
        select(Conversation).where(Conversation.owner_id == user.id).order_by(Conversation.updated_at.desc())
    )
    return [ConversationOut.model_validate(c, from_attributes=True) for c in result.scalars().all()]


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(conversation_id: int, user: CurrentUser, db: DbDep) -> ConversationDetailOut:
    conversation = await _load_owned_conversation(db, user, conversation_id)
    # An explicit query rather than the lazy `conversation.messages`
    # relationship - accessing a lazy relationship synchronously (plain
    # attribute access, not an await) isn't valid on an AsyncSession and
    # raises MissingGreenlet.
    result = await db.execute(select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id))
    messages = [MessageOut.model_validate(m, from_attributes=True) for m in result.scalars().all()]
    return ConversationDetailOut(
        id=conversation.id,
        title=conversation.title,
        model=conversation.model,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=messages,
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: int, user: CurrentUser, db: DbDep) -> None:
    conversation = await _load_owned_conversation(db, user, conversation_id)
    await db.delete(conversation)
    await db.commit()


async def _start_generation(
    request: Request, conversation: Conversation, messages: list[ChatMessage], ollama: OllamaDep, settings: SettingsDep
) -> StreamingResponse:
    context_length = await ollama.context_length(conversation.model)
    if context_length is not None:
        messages = _trim_to_context(messages, context_length)

    limiter = _get_limiter(request)
    try:
        ticket = limiter.reserve()
    except QueueFullError as exc:
        raise HTTPException(
            status_code=429, detail={"message": "server is busy, try again shortly", "code": "queue_full"}
        ) from exc

    request_id_var.set(uuid.uuid4().hex[:12])
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.db_session_factory
    events = generate_events(ollama, conversation.model, messages, settings, limiter, ticket)

    return StreamingResponse(
        _stream_and_persist(events, session_factory, conversation.id),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _history_to_chat_messages(rows: list[Message]) -> list[ChatMessage]:
    # Only `content` ever crosses back into what's sent to Ollama -
    # ChatMessage has no `thinking` field, so a stored thinking trace is
    # structurally impossible to resend upstream, not just something this
    # code happens not to do.
    return [ChatMessage(role=m.role, content=m.content) for m in rows]  # type: ignore[arg-type]


@router.post("/{conversation_id}/messages")
async def post_message(
    conversation_id: int,
    req: PostMessageRequest,
    request: Request,
    user: CurrentUser,
    db: DbDep,
    ollama: OllamaDep,
    settings: SettingsDep,
) -> StreamingResponse:
    conversation = await _load_owned_conversation(db, user, conversation_id)
    if not req.content:
        raise HTTPException(
            status_code=400, detail={"message": "message must not be empty", "code": "invalid_request"}
        )
    if len(req.content) > settings.max_message_chars:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"message too long (max {settings.max_message_chars} chars)",
                "code": "invalid_request",
            },
        )

    now = utcnow()
    db.add(Message(conversation_id=conversation.id, role="user", content=req.content, created_at=now))
    conversation.updated_at = now
    if conversation.title is None:
        conversation.title = req.content[:TITLE_PREVIEW_CHARS]
    await db.commit()

    history_result = await db.execute(
        select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id)
    )
    messages = _history_to_chat_messages(list(history_result.scalars().all()))
    return await _start_generation(request, conversation, messages, ollama, settings)


@router.post("/{conversation_id}/retry")
async def retry_message(
    conversation_id: int,
    request: Request,
    user: CurrentUser,
    db: DbDep,
    ollama: OllamaDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Regenerates a reply for the conversation's current trailing,
    unanswered user turn - unlike post_message, this does NOT add a new
    user message row. The user's turn was already persisted the moment it
    was first sent (see post_message); a naive "retry" that resubmitted it
    as a new message would duplicate that turn in the conversation every
    time generation failed and was retried."""
    conversation = await _load_owned_conversation(db, user, conversation_id)
    history_result = await db.execute(
        select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id)
    )
    rows = list(history_result.scalars().all())
    if not rows or rows[-1].role != "user":
        raise HTTPException(
            status_code=400, detail={"message": "nothing to retry", "code": "invalid_request"}
        )
    messages = _history_to_chat_messages(rows)
    return await _start_generation(request, conversation, messages, ollama, settings)


async def _stream_and_persist(
    events: AsyncGenerator[StreamEvent, None],
    session_factory: async_sessionmaker[AsyncSession],
    conversation_id: int,
) -> AsyncGenerator[bytes, None]:
    """Forwards every event to the client unchanged while accumulating the
    reply, then persists the assistant turn ONLY on a clean finish. A
    partial reply from an abort/timeout/mid-stream error must never be
    saved as if it were complete - it would poison every later turn's
    context with a truncated, possibly mid-sentence "answer" (the bb872fe
    history-poisoning bug, made durable in the database if this is wrong
    instead of just wrong on one page load).

    Pulled out of the route handler as its own function (rather than a
    closure) so a test can drive it directly against a synthetic event
    generator - including one that ends via `.aclose()` mid-stream to
    simulate a real client disconnect, the same way tests/test_api.py
    exercises generate_events()'s own cancellation handling.

    Opens its own DB session rather than reusing a request-scoped one:
    this keeps running after the route handler has already returned its
    StreamingResponse, for however long generation takes - a
    request-scoped session shouldn't be assumed to still be valid by then.

    Wraps `events` in contextlib.aclosing() for the same reason
    app/api.py's _produce() wraps ollama.chat(): if a client disconnects
    (or a test simulates one via .aclose() on this generator), the
    GeneratorExit lands at the `yield` below and unwinds through this
    `async for` - which, on its own, does not call events.aclose(). Without
    the explicit wrapper, generate_events()'s own cleanup (cancelling its
    producer task, releasing the limiter permit) would only happen later,
    whenever the abandoned generator gets garbage-collected, in whatever
    task that happens to run in.
    """
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    finished_cleanly = False
    async with contextlib.aclosing(events):
        async for event in events:
            yield event.model_dump_json().encode() + b"\n"
            if isinstance(event, ContentEvent):
                content_parts.append(event.text)
            elif isinstance(event, ThinkingEvent):
                thinking_parts.append(event.text)
            elif isinstance(event, DoneEvent):
                finished_cleanly = True

    if finished_cleanly and content_parts:
        async with session_factory() as write_db:
            write_db.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content="".join(content_parts),
                    thinking="".join(thinking_parts) or None,
                    created_at=utcnow(),
                )
            )
            conv_result = await write_db.execute(select(Conversation).where(Conversation.id == conversation_id))
            conv = conv_result.scalar_one_or_none()
            if conv is not None:
                conv.updated_at = utcnow()
            await write_db.commit()
