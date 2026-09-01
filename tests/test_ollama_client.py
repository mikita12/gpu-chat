import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from app.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaHTTPError,
    OllamaProtocolError,
)

BASE_URL = "http://ollama.test"


@pytest.fixture
async def ollama() -> AsyncIterator[OllamaClient]:
    async with httpx.AsyncClient() as client:
        yield OllamaClient(BASE_URL, client)


@respx.mock
async def test_list_models_success(ollama: OllamaClient) -> None:
    respx.get(f"{BASE_URL}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3.8:27b", "size": 1}]})
    )
    models = await ollama.list_models()
    assert [m.name for m in models] == ["qwen3.8:27b"]


@respx.mock
async def test_list_models_http_error(ollama: OllamaClient) -> None:
    respx.get(f"{BASE_URL}/api/tags").mock(
        return_value=httpx.Response(500, json={"error": "internal error"})
    )
    with pytest.raises(OllamaHTTPError) as exc_info:
        await ollama.list_models()
    assert exc_info.value.status_code == 500
    assert "internal error" in exc_info.value.message


@respx.mock
async def test_list_models_connection_refused(ollama: OllamaClient) -> None:
    respx.get(f"{BASE_URL}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(OllamaConnectionError):
        await ollama.list_models()


@respx.mock
async def test_chat_success_yields_content_then_done(ollama: OllamaClient) -> None:
    body = (
        '{"message": {"role": "assistant", "content": "Hi"}, "done": false}\n'
        '{"message": {"role": "assistant", "content": "!"}, "done": false}\n'
        '{"message": {"role": "assistant", "content": ""}, "done": true, '
        '"eval_count": 2, "eval_duration": 1000000}\n'
    )
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})
    )
    chunks = [c async for c in ollama.chat("qwen3.8:27b", [])]
    assert [c.message.content for c in chunks] == ["Hi", "!", ""]
    assert chunks[-1].done is True
    assert chunks[-1].eval_count == 2


@respx.mock
async def test_chat_upstream_404_raises_before_yielding(ollama: OllamaClient) -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "model 'ghost' not found"})
    )
    with pytest.raises(OllamaHTTPError) as exc_info:
        async for _ in ollama.chat("ghost", []):
            pass
    assert exc_info.value.status_code == 404
    assert "not found" in exc_info.value.message


@respx.mock
async def test_chat_connection_refused(ollama: OllamaClient) -> None:
    respx.post(f"{BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(OllamaConnectionError):
        async for _ in ollama.chat("qwen3.8:27b", []):
            pass


@respx.mock
async def test_chat_malformed_line_raises_protocol_error(ollama: OllamaClient) -> None:
    body = '{"message": {"role": "assistant", "content": "ok"}, "done": false}\nnot json at all\n'
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})
    )
    seen = []
    with pytest.raises(OllamaProtocolError):
        async for chunk in ollama.chat("qwen3.8:27b", []):
            seen.append(chunk)
    assert len(seen) == 1  # the valid line was yielded before the bad one blew up


@respx.mock
async def test_context_length_scans_family_prefixed_key(ollama: OllamaClient) -> None:
    respx.post(f"{BASE_URL}/api/show").mock(
        return_value=httpx.Response(
            200,
            json={
                "details": {"family": "qwen35"},
                "model_info": {"qwen35.context_length": 262144, "qwen35.embedding_length": 2048},
            },
        )
    )
    assert await ollama.context_length("qwen3.8:27b") == 262144


@respx.mock
async def test_chat_mid_stream_oom_error_is_classified_as_oom(ollama: OllamaClient) -> None:
    # Ollama can return 200, stream some content, then hit an OOM while
    # loading the model and emit a bare {"error": ...} line instead of a
    # chat chunk. This must not leak as a pydantic ValidationError, and must
    # be distinguishable from other mid-stream failures (code == "oom"),
    # not the generic/misleading "upstream_http_200" an OllamaHTTPError
    # would otherwise produce for a request that already got a 200.
    body = (
        '{"message": {"role": "assistant", "content": "Hi"}, "done": false}\n'
        '{"error": "model requires more system memory (24.0 GiB) than is available (16.0 GiB)"}\n'
    )
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})
    )
    seen = []
    with pytest.raises(OllamaGenerationError) as exc_info:
        async for chunk in ollama.chat("qwen3.8:27b", []):
            seen.append(chunk)
    assert len(seen) == 1  # the good line was yielded before the error line
    assert "more system memory" in exc_info.value.message
    assert exc_info.value.code == "oom"


@respx.mock
async def test_chat_mid_stream_generic_error_is_classified_as_generation_failed(ollama: OllamaClient) -> None:
    body = '{"error": "an unexpected internal error occurred"}\n'
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})
    )
    with pytest.raises(OllamaGenerationError) as exc_info:
        async for _ in ollama.chat("qwen3.8:27b", []):
            pass
    assert exc_info.value.code == "generation_failed"


@respx.mock
async def test_chat_unexpected_chunk_shape_raises_protocol_error(ollama: OllamaClient) -> None:
    # Well-formed JSON that isn't a chat chunk at all (e.g. "message" as a
    # string instead of an object) must not crash the generator uncaught.
    body = '{"message": "not an object", "done": false}\n'
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})
    )
    with pytest.raises(OllamaProtocolError):
        async for _ in ollama.chat("qwen3.8:27b", []):
            pass


@respx.mock
async def test_show_model_sends_both_name_and_model_keys(ollama: OllamaClient) -> None:
    route = respx.post(f"{BASE_URL}/api/show").mock(
        return_value=httpx.Response(200, json={"details": {}, "model_info": {}})
    )
    await ollama.show_model("qwen3.8:27b")
    sent = json.loads(route.calls.last.request.content)
    assert sent == {"model": "qwen3.8:27b", "name": "qwen3.8:27b"}


@respx.mock
async def test_list_models_is_cached_within_ttl(ollama: OllamaClient) -> None:
    route = respx.get(f"{BASE_URL}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m", "size": 1}]})
    )
    await ollama.list_models()
    await ollama.list_models()
    assert route.call_count == 1
    ollama.invalidate_cache()
    await ollama.list_models()
    assert route.call_count == 2


@respx.mock
async def test_list_models_cache_expires_after_ttl() -> None:
    route = respx.get(f"{BASE_URL}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m", "size": 1}]})
    )
    async with httpx.AsyncClient() as client:
        short_ttl_ollama = OllamaClient(BASE_URL, client, cache_ttl_seconds=0.05)
        await short_ttl_ollama.list_models()
        await asyncio.sleep(0.1)
        await short_ttl_ollama.list_models()
    assert route.call_count == 2


@respx.mock
async def test_context_length_is_cached_within_ttl(ollama: OllamaClient) -> None:
    route = respx.post(f"{BASE_URL}/api/show").mock(
        return_value=httpx.Response(
            200, json={"details": {}, "model_info": {"qwenx.context_length": 4096}}
        )
    )
    assert await ollama.context_length("m") == 4096
    assert await ollama.context_length("m") == 4096
    assert route.call_count == 1
