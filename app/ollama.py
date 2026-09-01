import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

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

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def list_models(self) -> list[OllamaModelSummary]:
        data = await self._get_json("/api/tags")
        return OllamaTagsResponse.model_validate(data).models

    async def running_models(self) -> list[OllamaRunningModel]:
        data = await self._get_json("/api/ps")
        return OllamaPsResponse.model_validate(data).models

    async def show_model(self, name: str) -> OllamaShowResponse:
        data = await self._post_json("/api/show", {"name": name})
        return OllamaShowResponse.model_validate(data)

    async def context_length(self, name: str) -> int | None:
        """Best-effort context window size for a model.

        Ollama's /api/show buries this under a family-prefixed key in
        model_info, e.g. "qwen35.context_length" - the key name varies by
        model architecture, so we scan for the suffix rather than assume one.
        """
        show = await self.show_model(name)
        for key, value in show.model_info.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return value
        return None

    async def chat(
        self, model: str, messages: list[ChatMessage]
    ) -> AsyncIterator[OllamaChatChunk]:
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
                    yield OllamaChatChunk.model_validate(raw)
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
