import asyncio
from collections.abc import AsyncIterator

from sqlalchemy import delete, event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

from app.models import Session as SessionRow
from app.models import utcnow


def create_engine(database_url: str) -> AsyncEngine:
    # A small, fixed pool - this app targets a resource-constrained single
    # box (e.g. a Raspberry Pi), not a high-concurrency web server; SQLite
    # itself only ever lets one writer through at a time regardless.
    engine = create_async_engine(database_url, pool_size=5, max_overflow=0)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: object, _record: ConnectionPoolEntry) -> None:
        # WAL lets readers proceed while a write is in progress instead of
        # blocking on the single database file lock; NORMAL synchronous is
        # the standard, safe pairing with WAL (fewer fsyncs than FULL, still
        # crash-consistent) - an acceptable trade-off for chat history, not
        # for financial data.
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def sweep_expired_sessions(session_factory: async_sessionmaker[AsyncSession]) -> int:
    """Deletes every session row past its expiry. Called opportunistically
    on lookup (see app/auth.py) and periodically in the background (see
    run_session_sweep_loop below) so a client that never comes back doesn't
    leave its row behind forever - sessions live in SQLite, not an
    in-process dict, so nothing else ever reclaims them."""
    async with session_factory() as db:
        result = await db.execute(delete(SessionRow).where(SessionRow.expires_at < utcnow()))
        await db.commit()
        return result.rowcount or 0  # type: ignore[attr-defined]  # CursorResult at runtime


async def run_session_sweep_loop(session_factory: async_sessionmaker[AsyncSession], interval_seconds: float) -> None:
    """Runs forever as a background task started in app/main.py's lifespan;
    cancelled (via task.cancel()) on shutdown, which is why the sleep is the
    only await in the loop body - a CancelledError there just ends the
    loop, nothing left to clean up."""
    while True:
        await asyncio.sleep(interval_seconds)
        await sweep_expired_sessions(session_factory)


async def get_db_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with session_factory() as db:
        yield db
