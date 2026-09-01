import json
import time
from collections.abc import AsyncGenerator
from typing import Any, Generic, TypeVar

import httpx
from pydantic import ValidationError

from app.schemas import (
    ChatMessage,
    OllamaChatChunk,
    OllamaModelSummary,
    OllamaPsResponse,
    OllamaRunningModel,
    OllamaShowResponse,
    OllamaTagsResponse,
)


class OllamaError(Exception):
    """Base class for all errors raised by OllamaClient."""


class OllamaConnectionError(OllamaError):
    """Could not reach Ollama at all (connection refused, timeout, dropped mid-stream)."""


class OllamaHTTPError(OllamaError):
    """Ollama responded with a non-2xx status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Ollama returned {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class OllamaProtocolError(OllamaError):
    """Ollama's response didn't look like valid NDJSON chat output."""


class OllamaGenerationError(OllamaError):
    """Ollama returned 200 and started streaming, then failed mid-generation
    (e.g. an OOM while loading the model onto the GPU) via an inline
    {"error": ...} line instead of a chat chunk. Distinct from
    OllamaHTTPError (a real non-200 response): the request *was* accepted,
    so this is a different failure mode - and a different thing for a
    client to do about it - than a rejected request.

    Classifies itself from Ollama's own wording (observed live: "model
    requires more system memory (24.0 GiB) than is available (16.0 GiB)")
    into a distinguishable `code` a frontend can act on, rather than the
    generic upstream_http_200 an OllamaHTTPError would otherwise produce.
    """

    _OOM_MARKERS = ("out of memory", "requires more system memory", "requires more memory")

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        lowered = message.lower()
        self.code = "oom" if any(marker in lowered for marker in self._OOM_MARKERS) else "generation_failed"


T = TypeVar("T")


class _CacheMiss:
    """Sentinel distinct from None, so a cached None value is still a hit."""

    __slots__ = ()


_MISS = _CacheMiss()


class _TTLCache(Generic[T]):
    """A tiny fixed-TTL cache. Not thread-safe; fine for a single-process
    asyncio app with one event loop."""

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._store: dict[object, tuple[float, T]] = {}

    def get(self, key: object) -> T | _CacheMiss:
        entry = self._store.get(key)
        if entry is None:
            return _MISS
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return _MISS
        return value

    def set(self, key: object, value: T) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)

    def invalidate(self) -> None:
        self._store.clear()


def _extract_error(body: bytes) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")[:500]
    if isinstance(data, dict) and "error" in data:
        return str(data["error"])
    return body.decode(errors="replace")[:500]


class OllamaClient:
    """Thin async wrapper around the Ollama HTTP API.

    Knows nothing about FastAPI/Starlette or HTTP-to-the-browser concerns
    (heartbeats, cancellation orchestration, backpressure) - those live in
    app/api.py. This class just talks to Ollama correctly: it raises typed
    errors instead of returning empty/partial data on failure.
    """

    def __init__(self, base_url: str, client: httpx.AsyncClient, cache_ttl_seconds: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._models_cache: _TTLCache[list[OllamaModelSummary]] = _TTLCache(cache_ttl_seconds)
        self._context_length_cache: _TTLCache[int | None] = _TTLCache(cache_ttl_seconds)

    def invalidate_cache(self) -> None:
        """Explicit invalidation hook - nothing calls this yet, but later
        phases (e.g. after a model finishes loading/unloading) will want to
        force a fresh read instead of waiting out the TTL."""
        self._models_cache.invalidate()
        self._context_length_cache.invalidate()

    async def list_models(self) -> list[OllamaModelSummary]:
        cached = self._models_cache.get("models")
        if not isinstance(cached, _CacheMiss):
            return cached
        data = await self._get_json("/api/tags")
        models = OllamaTagsResponse.model_validate(data).models
        self._models_cache.set("models", models)
        return models

    async def running_models(self) -> list[OllamaRunningModel]:
        data = await self._get_json("/api/ps")
        return OllamaPsResponse.model_validate(data).models

    async def show_model(self, name: str) -> OllamaShowResponse:
        # Send both keys: older Ollama versions expect "name", newer ones
        # "model" - verified both are accepted by the current live version,
        # so sending both is forward-compatible with no downside.
        data = await self._post_json("/api/show", {"model": name, "name": name})
        return OllamaShowResponse.model_validate(data)

    async def context_length(self, name: str) -> int | None:
        """Best-effort context window size for a model.

        Ollama's /api/show buries this under a family-prefixed key in
        model_info, e.g. "qwen35.context_length" - the key name varies by
        model architecture, so we scan for the suffix rather than assume one.
        """
        cached = self._context_length_cache.get(name)
        if not isinstance(cached, _CacheMiss):
            return cached
        show = await self.show_model(name)
        result: int | None = None
        for key, value in show.model_info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                result = value
                break
        self._context_length_cache.set(name, result)
        return result

    async def chat(
        self, model: str, messages: list[ChatMessage]
    ) -> AsyncGenerator[OllamaChatChunk, None]:
        payload = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
        }
        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/api/chat", json=payload
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise OllamaHTTPError(response.status_code, _extract_error(body))
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        raw: Any = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaProtocolError(
                            f"malformed line from Ollama: {line!r}"
                        ) from exc
                    # Ollama can return 200 and start streaming, then hit a
                    # failure mid-generation (e.g. an OOM while loading the
                    # model onto the GPU) and emit a bare {"error": ...}
                    # line instead of a chat chunk. Must be checked before
                    # model_validate(), which would otherwise either drop it
                    # (extra fields are ignored) or raise a confusing
                    # ValidationError depending on shape.
                    if isinstance(raw, dict) and "error" in raw:
                        raise OllamaGenerationError(str(raw["error"]))
                    try:
                        chunk = OllamaChatChunk.model_validate(raw)
                    except ValidationError as exc:
                        raise OllamaProtocolError(
                            f"unexpected chat chunk shape from Ollama: {raw!r}"
                        ) from exc
                    yield chunk
        except httpx.TransportError as exc:
            raise OllamaConnectionError(str(exc)) from exc

    async def _get_json(self, path: str) -> Any:
        try:
            resp = await self._client.get(f"{self._base_url}{path}")
        except httpx.TransportError as exc:
            raise OllamaConnectionError(str(exc)) from exc
        if resp.status_code != 200:
            raise OllamaHTTPError(resp.status_code, _extract_error(resp.content))
        return resp.json()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            resp = await self._client.post(f"{self._base_url}{path}", json=payload)
        except httpx.TransportError as exc:
            raise OllamaConnectionError(str(exc)) from exc
        if resp.status_code != 200:
            raise OllamaHTTPError(resp.status_code, _extract_error(resp.content))
        return resp.json()
