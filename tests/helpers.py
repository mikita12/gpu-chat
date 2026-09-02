import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth_api import router as auth_router
from app.config import Settings
from app.conversations_api import router as conversations_router
from app.limiter import GenerationLimiter
from app.ollama import OllamaClient, OllamaError
from app.schemas import ChatMessage, OllamaChatChunk, OllamaModelSummary, OllamaRunningModel


@dataclass
class FakeOllamaClient:
    """Stands in for OllamaClient in tests that exercise app/api.py's
    consumer logic (heartbeat, stall watchdog, cancellation) without needing
    to fake Ollama's exact wire format over a mocked transport."""

    chunks: list[OllamaChatChunk] = field(default_factory=list)
    models: list[OllamaModelSummary] = field(default_factory=lambda: [OllamaModelSummary(name="test-model")])
    running: list[OllamaRunningModel] = field(default_factory=list)
    chat_error: OllamaError | None = None
    hang_seconds: float | None = None
    context_length_value: int | None = None
    cancelled: bool = False
    closed: bool = False
    last_chat_messages: list[ChatMessage] | None = None

    async def list_models(self) -> list[OllamaModelSummary]:
        return self.models

    async def running_models(self) -> list[OllamaRunningModel]:
        return self.running

    async def context_length(self, model: str) -> int | None:
        return self.context_length_value

    async def chat(self, model: str, messages: list[ChatMessage]) -> AsyncIterator[OllamaChatChunk]:
        self.last_chat_messages = messages
        try:
            if self.hang_seconds is not None:
                await asyncio.sleep(self.hang_seconds)
            for chunk in self.chunks:
                yield chunk
                await asyncio.sleep(0)
            if self.chat_error is not None:
                raise self.chat_error
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            # Set regardless of how we got here (normal completion,
            # CancelledError, or GeneratorExit from an explicit aclose()) -
            # proves the generator was actually torn down, not abandoned.
            self.closed = True


def fast_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "heartbeat_seconds": 0.02,
        "stall_timeout_seconds": 0.08,
        "stream_queue_maxsize": 64,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def make_account_app(
    session_factory: async_sessionmaker[AsyncSession],
    ollama: OllamaClient | FakeOllamaClient | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Builds an app exposing the auth + conversations routers against a
    real (temp-file) SQLite database via `session_factory` - mirrors
    `tests/test_api.py`'s make_app(), extended with the DB state the new
    routers need on app.state instead of going through the real lifespan
    (which also runs Alembic migrations - not needed here, see
    tests/conftest.py's session_factory fixture)."""
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(conversations_router)
    app.state.db_session_factory = session_factory
    app.state.limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    # Every conversation route declares an OllamaDep, resolved eagerly by
    # FastAPI before the handler body runs even on a path that never
    # actually calls it (e.g. retry's "nothing to retry" 400) - app.state.ollama
    # must exist regardless.
    app.state.ollama = ollama if ollama is not None else FakeOllamaClient()
    if settings is not None:
        from app.config import get_settings

        app.dependency_overrides[get_settings] = lambda: settings
    return app
