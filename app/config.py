from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for gpu-chat, read from the environment.

    Env var names match what systemd/gpu-chat.service already sets, so
    existing deployments keep working unchanged when this lands.
    """

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Assume Ollama runs on the same machine as this app by default.
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.8:27b"
    ollama_cache_ttl_seconds: float = 5.0

    # Streaming behaviour (app/api.py).
    heartbeat_seconds: float = 10.0
    stall_timeout_seconds: float = 90.0
    # Small on purpose: on a memory-constrained host (e.g. a Raspberry Pi
    # proxying to Ollama on a separate machine), a slow or backgrounded
    # client should only ever buffer a handful of events in *this*
    # process's RAM, not accumulate unbounded backlog.
    stream_queue_maxsize: int = 16

    # Concurrency (Phase 3).
    max_concurrent_generations: int = 1
    max_queue_size: int = 10

    # Validation / limits (Phase 4).
    max_messages: int = 50
    max_message_chars: int = 8_000
    max_prompt_chars: int = 24_000

    # Auth (Phase 4). Empty string (default) means auth is disabled - anyone
    # can use the API, matching today's zero-friction LAN behaviour. Setting
    # this env var to a non-empty value turns on bearer-token enforcement.
    # Orthogonal to the per-user session system below - this gates the
    # stateless /api/chat passthrough for scripted/curl access.
    bearer_token: str = ""

    # Accounts and per-user conversation persistence.
    #
    # Relative path is fine for local/dev use (`uvicorn app.main:app` from
    # the repo root). A systemd deployment MUST override this to an
    # absolute path outside any releases/<sha>/ worktree - see README - or
    # the database gets deleted on the next auto-deploy.
    database_url: str = "sqlite+aiosqlite:///./gpu_chat.db"
    session_cookie_name: str = "gpu_chat_session"
    session_ttl_seconds: float = 60 * 60 * 24 * 30  # 30 days
    session_cleanup_interval_seconds: float = 60 * 60  # hourly sweep


@lru_cache
def get_settings() -> Settings:
    return Settings()
