import httpx
from fastapi import FastAPI
from httpx import ASGITransport

from app.api import router as api_router
from app.health import router as health_router
from app.limiter import GenerationLimiter
from app.ollama import OllamaConnectionError
from app.schemas import OllamaModelSummary

from .helpers import FakeOllamaClient


def make_app(ollama: FakeOllamaClient, bearer_token: str = "") -> FastAPI:
    from app.config import get_settings

    app = FastAPI()
    app.include_router(api_router)
    app.include_router(health_router)
    app.state.ollama = ollama
    app.state.limiter = GenerationLimiter(max_concurrent=1, max_queue_size=5)
    if bearer_token:
        settings = get_settings()
        app.dependency_overrides[get_settings] = lambda: settings.model_copy(
            update={"bearer_token": bearer_token}
        )
    return app


async def test_readyz_ok_when_ollama_reachable() -> None:
    app = make_app(FakeOllamaClient())
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/readyz")
    assert resp.status_code == 200


async def test_readyz_503_when_ollama_unreachable() -> None:
    fake = FakeOllamaClient()

    async def failing_list_models() -> list[OllamaModelSummary]:
        raise OllamaConnectionError("refused")

    fake.list_models = failing_list_models  # type: ignore[method-assign]
    app = make_app(fake)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/readyz")
    assert resp.status_code == 503


async def test_readyz_does_not_require_bearer_token() -> None:
    app = make_app(FakeOllamaClient(), bearer_token="secret")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # No Authorization header at all - must still succeed, proving
        # /readyz is on the unauthenticated router, not caught by
        # require_auth the way /api/* routes are.
        resp = await client.get("/readyz")
    assert resp.status_code == 200


async def test_healthz_ok_with_no_ollama_dependency() -> None:
    app = make_app(FakeOllamaClient())
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200


async def test_healthz_does_not_require_bearer_token() -> None:
    app = make_app(FakeOllamaClient(), bearer_token="secret")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200


async def test_metrics_endpoint_returns_prometheus_format() -> None:
    app = make_app(FakeOllamaClient())
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "gpu_chat_active_generations" in resp.text


async def test_metrics_does_not_require_bearer_token() -> None:
    app = make_app(FakeOllamaClient(), bearer_token="secret")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 200
