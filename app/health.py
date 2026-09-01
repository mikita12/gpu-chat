from fastapi import APIRouter, HTTPException

from app.api import OllamaDep
from app.ollama import OllamaError

# Deliberately its own router, included separately in app/main.py from
# app.api.router (which has require_auth applied). A health/deploy check
# must work without a bearer token even when one is configured - update.sh
# polls this before cutting a new release over, and a monitoring system
# shouldn't need API credentials just to ask "are you up".
router = APIRouter()


@router.get("/readyz")
async def readyz(ollama: OllamaDep) -> dict[str, str]:
    """Ready to serve real traffic: process is up AND Ollama is reachable.
    Used by update.sh to health-gate a deploy before switching over."""
    try:
        await ollama.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}
