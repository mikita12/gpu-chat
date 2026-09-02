import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from alembic import command
from app.api import router
from app.auth_api import router as auth_router
from app.config import get_settings
from app.conversations_api import router as conversations_router
from app.db import create_engine, create_session_factory, run_session_sweep_loop
from app.health import router as health_router
from app.limiter import GenerationLimiter
from app.logging_config import configure_logging
from app.ollama import OllamaClient

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def _run_migrations(database_url: str) -> None:
    # Runs synchronously against a plain sqlite3 connection (see
    # alembic/env.py) - fine to block the event loop briefly at startup,
    # before the app accepts any traffic. A migration that fails raises
    # here, which keeps /readyz from ever returning 200 - update.sh's
    # health-gated deploy then discards the candidate release instead of
    # cutting over to a broken schema.
    alembic_cfg = AlembicConfig(os.path.join(REPO_ROOT, "alembic.ini"))
    alembic_cfg.set_main_option("script_location", os.path.join(REPO_ROOT, "alembic"))
    command.upgrade(alembic_cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    _run_migrations(settings.database_url)
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    app.state.db_session_factory = session_factory
    sweep_task = asyncio.create_task(
        run_session_sweep_loop(session_factory, settings.session_cleanup_interval_seconds)
    )
    # One shared client for the process lifetime, not one per request - a
    # per-request httpx.AsyncClient means a new connection pool (and no
    # keep-alive reuse) on every single chat message.
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)) as client:
            app.state.ollama = OllamaClient(
                settings.ollama_url, client, cache_ttl_seconds=settings.ollama_cache_ttl_seconds
            )
            app.state.limiter = GenerationLimiter(
                settings.max_concurrent_generations, settings.max_queue_size
            )
            yield
    finally:
        sweep_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sweep_task
        await engine.dispose()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.include_router(auth_router)
    app.include_router(conversations_router)
    app.include_router(health_router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
