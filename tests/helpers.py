import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.config import Settings
from app.ollama import OllamaError
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
