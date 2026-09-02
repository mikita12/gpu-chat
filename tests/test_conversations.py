import contextlib
import tempfile

import httpx
from httpx import ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import generate_events
from app.conversations_api import _stream_and_persist
from app.db import create_engine, create_session_factory
from app.limiter import GenerationLimiter
from app.models import Base, Conversation, Message
from app.ollama import OllamaHTTPError
from app.schemas import OllamaChatChunk, OllamaChatMessageChunk

from .helpers import FakeOllamaClient, fast_settings, make_account_app


async def _register(client: httpx.AsyncClient, username: str = "alice") -> None:
    resp = await client.post("/api/auth/register", json={"username": username, "password": "hunter22"})
    assert resp.status_code == 201


async def test_create_list_get_delete_conversation(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)

        create_resp = await client.post("/api/conversations", json={"title": "hello"})
        assert create_resp.status_code == 201
        conv = create_resp.json()
        assert conv["title"] == "hello"

        list_resp = await client.get("/api/conversations")
        assert list_resp.status_code == 200
        assert [c["id"] for c in list_resp.json()] == [conv["id"]]

        get_resp = await client.get(f"/api/conversations/{conv['id']}")
        assert get_resp.status_code == 200
        assert get_resp.json()["messages"] == []

        delete_resp = await client.delete(f"/api/conversations/{conv['id']}")
        assert delete_resp.status_code == 204

        assert (await client.get(f"/api/conversations/{conv['id']}")).status_code == 404


async def test_conversation_routes_require_a_session(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/api/conversations")).status_code == 401
        assert (await client.post("/api/conversations", json={})).status_code == 401


async def test_user_cannot_read_or_delete_another_users_conversation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as alice:
        await _register(alice, "alice")
        conv = (await alice.post("/api/conversations", json={})).json()

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as bob:
        await _register(bob, "bob")
        get_resp = await bob.get(f"/api/conversations/{conv['id']}")
        delete_resp = await bob.delete(f"/api/conversations/{conv['id']}")

    assert get_resp.status_code == 404
    assert delete_resp.status_code == 404

    # Bob's failed attempts didn't touch it - it's still there for its
    # actual owner.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as alice_again:
        await alice_again.post("/api/auth/login", json={"username": "alice", "password": "hunter22"})
        still_there = await alice_again.get(f"/api/conversations/{conv['id']}")
    assert still_there.status_code == 200


async def test_assistant_reply_persisted_only_on_clean_done(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    fake = FakeOllamaClient(
        chunks=[
            OllamaChatChunk(message=OllamaChatMessageChunk(content="hi there")),
            OllamaChatChunk(message=OllamaChatMessageChunk(content=""), done=True, eval_count=1, eval_duration=1),
        ]
    )
    app = make_account_app(session_factory, ollama=fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)
        conv = (await client.post("/api/conversations", json={})).json()

        resp = await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "hello"})
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().splitlines() if line]
        assert any('"type":"done"' in line for line in lines)

    async with session_factory() as db:
        result = await db.execute(select(Message).where(Message.conversation_id == conv["id"]))
        messages = result.scalars().all()
    roles = [m.role for m in messages]
    assert roles == ["user", "assistant"]
    assistant = next(m for m in messages if m.role == "assistant")
    assert assistant.content == "hi there"


async def test_assistant_reply_not_persisted_on_mid_stream_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    fake = FakeOllamaClient(
        chunks=[OllamaChatChunk(message=OllamaChatMessageChunk(content="partial answer"))],
        chat_error=OllamaHTTPError(500, "boom"),
    )
    app = make_account_app(session_factory, ollama=fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)
        conv = (await client.post("/api/conversations", json={})).json()

        resp = await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "hello"})
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().splitlines() if line]
        assert any('"type":"error"' in line for line in lines)

    async with session_factory() as db:
        result = await db.execute(select(Message).where(Message.conversation_id == conv["id"]))
        messages = result.scalars().all()
    # Only the user's turn exists - the partial "partial answer" text must
    # never be saved as if it were a complete reply.
    assert [m.role for m in messages] == ["user"]


async def test_assistant_reply_not_persisted_on_client_disconnect() -> None:
    """Drives _stream_and_persist directly against a synthetic event
    generator that is aclose()'d mid-stream (the same technique
    tests/test_api.py uses to simulate a real client disconnect), rather
    than trying to force an httpx-level disconnect through ASGITransport -
    which, like the existing streaming tests in this repo, doesn't reliably
    simulate one."""
    fake = FakeOllamaClient(hang_seconds=10.0)
    settings = fast_settings(heartbeat_seconds=0.02, stall_timeout_seconds=10.0)
    limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    ticket = limiter.reserve()
    events = generate_events(fake, "m", [], settings, limiter, ticket)  # type: ignore[arg-type]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite+aiosqlite:///{tmpdir}/test.db"
        engine = create_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = create_session_factory(engine)
        async with factory() as db:
            conversation = Conversation(owner_id=1, model="m")
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            conversation_id = conversation.id

        agen = _stream_and_persist(events, factory, conversation_id)
        await agen.__anext__()  # consumes the GeneratingEvent, starts the producer
        with contextlib.suppress(StopAsyncIteration):
            await agen.__anext__()  # a PingEvent - _produce's task now exists
        await agen.aclose()  # simulates the client disconnecting mid-stream

        async with factory() as db:
            result = await db.execute(select(Message).where(Message.conversation_id == conversation_id))
            assert result.scalars().all() == []

        await engine.dispose()


async def test_history_sent_to_ollama_excludes_thinking(session_factory: async_sessionmaker[AsyncSession]) -> None:
    fake = FakeOllamaClient(
        chunks=[
            OllamaChatChunk(message=OllamaChatMessageChunk(thinking="pondering...")),
            OllamaChatChunk(message=OllamaChatMessageChunk(content="the answer")),
            OllamaChatChunk(message=OllamaChatMessageChunk(content=""), done=True, eval_count=1, eval_duration=1),
        ]
    )
    app = make_account_app(session_factory, ollama=fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)
        conv = (await client.post("/api/conversations", json={})).json()

        await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "first"})
        # Second turn: the history rebuilt from the DB and sent to Ollama
        # must contain the first assistant reply's *content* only.
        await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "second"})

    assert fake.last_chat_messages is not None
    assert not hasattr(fake.last_chat_messages[0], "thinking")
    contents = [m.content for m in fake.last_chat_messages]
    assert "the answer" in contents
    assert not any("pondering" in c for c in contents)


async def test_reply_stores_thinking_separately(session_factory: async_sessionmaker[AsyncSession]) -> None:
    fake = FakeOllamaClient(
        chunks=[
            OllamaChatChunk(message=OllamaChatMessageChunk(thinking="pondering...")),
            OllamaChatChunk(message=OllamaChatMessageChunk(content="the answer")),
            OllamaChatChunk(message=OllamaChatMessageChunk(content=""), done=True, eval_count=1, eval_duration=1),
        ]
    )
    app = make_account_app(session_factory, ollama=fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)
        conv = (await client.post("/api/conversations", json={})).json()
        await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "hi"})

    async with session_factory() as db:
        result = await db.execute(
            select(Message).where(Message.conversation_id == conv["id"], Message.role == "assistant")
        )
        assistant = result.scalar_one()
    assert assistant.content == "the answer"
    assert assistant.thinking == "pondering..."


async def test_retry_does_not_duplicate_the_user_turn(session_factory: async_sessionmaker[AsyncSession]) -> None:
    fake = FakeOllamaClient(chat_error=OllamaHTTPError(500, "boom"))
    app = make_account_app(session_factory, ollama=fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)
        conv = (await client.post("/api/conversations", json={})).json()
        await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": "hello"})

        # First attempt failed (no assistant reply yet) - retry should
        # regenerate against the SAME user turn, not add a second one.
        fake.chat_error = None
        fake.chunks = [
            OllamaChatChunk(message=OllamaChatMessageChunk(content="ok now"), done=True, eval_count=1, eval_duration=1)
        ]
        retry_resp = await client.post(f"/api/conversations/{conv['id']}/retry")
        assert retry_resp.status_code == 200

    async with session_factory() as db:
        result = await db.execute(select(Message).where(Message.conversation_id == conv["id"]).order_by(Message.id))
        messages = result.scalars().all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "hello"
    assert messages[1].content == "ok now"


async def test_retry_with_no_unanswered_turn_returns_400(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)
        conv = (await client.post("/api/conversations", json={})).json()
        resp = await client.post(f"/api/conversations/{conv['id']}/retry")
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "invalid_request"


async def test_title_auto_set_from_first_message(session_factory: async_sessionmaker[AsyncSession]) -> None:
    fake = FakeOllamaClient(
        chunks=[OllamaChatChunk(message=OllamaChatMessageChunk(content="ok"), done=True, eval_count=1)]
    )
    app = make_account_app(session_factory, ollama=fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await _register(client)
        conv = (await client.post("/api/conversations", json={})).json()
        assert conv["title"] is None
        question = "what is the capital of France?"
        await client.post(f"/api/conversations/{conv['id']}/messages", json={"content": question})
        detail = (await client.get(f"/api/conversations/{conv['id']}")).json()
    assert detail["title"] == "what is the capital of France?"
