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
    stream_queue_maxsize: int = 64

    # Concurrency (Phase 3).
    max_concurrent_generations: int = 1
    max_queue_size: int = 10

    # Validation / limits (Phase 4).
    max_messages: int = 50
    max_message_chars: int = 8_000
    max_prompt_chars: int = 24_000
    rate_limit_per_minute: int = 30

    # Auth (Phase 4). Empty string (default) means auth is disabled - anyone
    # can use the API, matching today's zero-friction LAN behaviour. Setting
    # this env var to a non-empty value turns on bearer-token enforcement.
    bearer_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
