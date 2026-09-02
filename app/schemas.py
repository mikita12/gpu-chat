from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# --- Chat request from the browser -----------------------------------------

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None


# --- Stream event protocol (backend -> browser, one JSON object per line) --


class ContentEvent(BaseModel):
    type: Literal["content"] = "content"
    text: str


class ThinkingEvent(BaseModel):
    """A chunk of a reasoning model's thinking trace (Ollama's
    `message.thinking`), kept distinct from ContentEvent so a client can
    render it separately (e.g. collapsed) and so it's obviously never the
    text that should be appended to the visible reply."""

    type: Literal["thinking"] = "thinking"
    text: str


class PingEvent(BaseModel):
    type: Literal["ping"] = "ping"


class QueuedEvent(BaseModel):
    type: Literal["queued"] = "queued"
    position: int


class GeneratingEvent(BaseModel):
    """Sent exactly once, the moment a generation slot is actually acquired
    - before any content/ping. Without this, a client that was queued has
    no way to distinguish "still queued, no position change yet" from
    "already generating, model still loading": both look identical from
    the wire (a queued event once, then a run of bare pings), since pings
    fire during the queueing phase too. Found live: a genuinely queued
    client stayed stuck showing "Queued" after its turn actually started."""

    type: Literal["generating"] = "generating"


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    eval_count: int | None = None
    eval_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    load_duration: int | None = None
    total_duration: int | None = None
    request_id: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    code: str
    request_id: str | None = None


StreamEvent = Annotated[
    ContentEvent | ThinkingEvent | PingEvent | QueuedEvent | GeneratingEvent | DoneEvent | ErrorEvent,
    Field(discriminator="type"),
]


class LoadedResponse(BaseModel):
    loaded: list[str]
    default: str


# --- Accounts ----------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    username: str


# --- Conversations -------------------------------------------------------


class CreateConversationRequest(BaseModel):
    model: str | None = None
    title: str | None = None


class ConversationOut(BaseModel):
    id: int
    title: str | None
    model: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    role: Role
    content: str
    thinking: str | None
    created_at: datetime


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut]


class PostMessageRequest(BaseModel):
    content: str


# --- Ollama's own response shapes (only the fields this app reads) ---------


class OllamaModelSummary(BaseModel):
    name: str
    size: int = 0


class OllamaTagsResponse(BaseModel):
    models: list[OllamaModelSummary] = []


class OllamaRunningModel(BaseModel):
    name: str
    expires_at: str | None = None


class OllamaPsResponse(BaseModel):
    models: list[OllamaRunningModel] = []


class OllamaShowDetails(BaseModel):
    parameter_size: str | None = None
    quantization_level: str | None = None
    family: str | None = None


class OllamaShowResponse(BaseModel):
    details: OllamaShowDetails = OllamaShowDetails()
    # Keyed like "<family>.context_length" - see OllamaClient.show_model().
    model_info: dict[str, int | str | float | bool | None] = {}


class OllamaChatMessageChunk(BaseModel):
    role: str = "assistant"
    content: str = ""
    # Present when Ollama is running a reasoning model with thinking
    # enabled. Never sent back upstream - see ChatMessage above, which has
    # no such field at all.
    thinking: str = ""


class OllamaChatChunk(BaseModel):
    """One line of Ollama's streaming /api/chat response."""

    message: OllamaChatMessageChunk = OllamaChatMessageChunk()
    done: bool = False
    eval_count: int | None = None
    eval_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    load_duration: int | None = None
    total_duration: int | None = None
