import json
import os

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

# Defaults assume this app runs on the same machine as Ollama.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.8:27b")

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str | None = None


@app.get("/api/models")
async def models():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{OLLAMA_URL}/api/tags")
        resp.raise_for_status()
        data = resp.json()
    return sorted(m["name"] for m in data.get("models", []))


@app.get("/api/loaded")
async def loaded():
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{OLLAMA_URL}/api/ps")
        resp.raise_for_status()
        data = resp.json()
    return {
        "loaded": [m["name"] for m in data.get("models", [])],
        "default": MODEL,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    model = req.model or MODEL

    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": req.messages, "stream": True},
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

    return StreamingResponse(stream(), media_type="text/plain")


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
