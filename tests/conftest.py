from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import create_engine, create_session_factory
from app.models import Base


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture
async def session_factory(db_path: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    # Schema created directly from the ORM metadata rather than via
    # `alembic upgrade head` - fast, and what these tests actually want to
    # exercise is the app's query/auth/persistence logic against a real
    # schema, not the migration tooling itself (see test_migrations.py for
    # the one test that specifically checks the migration matches the
    # models).
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()
