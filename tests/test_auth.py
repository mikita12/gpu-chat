import httpx
from httpx import ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import create_engine, create_session_factory
from app.models import Base, User

from .helpers import make_account_app


async def test_register_sets_cookie_and_returns_username(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})
        assert resp.status_code == 201
        assert resp.json() == {"username": "alice"}
        assert "gpu_chat_session" in resp.cookies


async def test_register_duplicate_username_returns_409(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})
        resp = await client.post("/api/auth/register", json={"username": "alice", "password": "different1"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "username_taken"


async def test_register_rejects_short_password(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/auth/register", json={"username": "alice", "password": "short"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "invalid_request"


async def test_password_hash_never_stores_plaintext(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})

    async with session_factory() as db:
        result = await db.execute(select(User).where(User.username == "alice"))
        user = result.scalar_one()
    assert user.password_hash != "hunter22"
    assert "hunter22" not in user.password_hash
    assert user.password_hash.startswith("$argon2id$")


async def test_login_with_correct_password_succeeds(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})
        await client.post("/api/auth/logout")
        resp = await client.post("/api/auth/login", json={"username": "alice", "password": "hunter22"})
    assert resp.status_code == 200
    assert resp.json() == {"username": "alice"}


async def test_login_with_wrong_password_returns_generic_401(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})
        resp = await client.post("/api/auth/login", json={"username": "alice", "password": "wrongpass"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == {"message": "invalid username or password", "code": "invalid_credentials"}


async def test_login_with_unknown_username_returns_same_generic_401(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        wrong_password_resp = await _register_then_wrong_login(client)
        unknown_user_resp = await client.post(
            "/api/auth/login", json={"username": "ghost", "password": "whatever1"}
        )
    # Same status and message shape either way - never reveals which of
    # username/password was the problem.
    assert wrong_password_resp.status_code == unknown_user_resp.status_code == 401
    assert wrong_password_resp.json() == unknown_user_resp.json()


async def _register_then_wrong_login(client: httpx.AsyncClient) -> httpx.Response:
    await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})
    return await client.post("/api/auth/login", json={"username": "alice", "password": "wrongpass"})


async def test_me_requires_session(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/auth/me")
    assert resp.status_code == 401
    assert resp.json()["detail"]["code"] == "unauthorized"


async def test_logout_invalidates_session(session_factory: async_sessionmaker[AsyncSession]) -> None:
    app = make_account_app(session_factory)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})
        assert (await client.get("/api/auth/me")).status_code == 200
        await client.post("/api/auth/logout")
        resp = await client.get("/api/auth/me")
    assert resp.status_code == 401


async def test_session_survives_a_process_restart(db_path: str) -> None:
    """Sessions live in SQLite, not an in-process dict - proven here by
    authenticating against one app/engine instance, discarding it entirely
    (simulating a process restart), and presenting the same cookie to a
    brand new app/engine instance pointed at the same database file."""
    engine1 = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine1.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory1 = create_session_factory(engine1)
    app1 = make_account_app(factory1)
    transport1 = ASGITransport(app=app1)
    async with httpx.AsyncClient(transport=transport1, base_url="http://test") as client:
        await client.post("/api/auth/register", json={"username": "alice", "password": "hunter22"})
        cookie = client.cookies["gpu_chat_session"]
    await engine1.dispose()

    # Brand new engine and app, same underlying file, same cookie.
    engine2 = create_engine(f"sqlite+aiosqlite:///{db_path}")
    factory2 = create_session_factory(engine2)
    app2 = make_account_app(factory2)
    transport2 = ASGITransport(app=app2)
    async with httpx.AsyncClient(
        transport=transport2, base_url="http://test", cookies={"gpu_chat_session": cookie}
    ) as client:
        resp = await client.get("/api/auth/me")
    await engine2.dispose()

    assert resp.status_code == 200
    assert resp.json() == {"username": "alice"}
