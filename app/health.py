from fastapi import APIRouter, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api import OllamaDep
from app.ollama import OllamaError

# Deliberately its own router, included separately in app/main.py from
# app.api.router (which has require_auth applied). A health/deploy/scrape
# check must work without a bearer token even when one is configured -
# update.sh polls /readyz before cutting a new release over, and a
# monitoring system shouldn't need API credentials just to ask "are you up".
router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Process liveness only - no Ollama call. Distinct from /readyz: this
    answers "is the process up", not "can it actually serve"."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(ollama: OllamaDep) -> dict[str, str]:
    """Ready to serve real traffic: process is up AND Ollama is reachable.
    Used by update.sh to health-gate a deploy before switching over."""
    try:
        await ollama.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
