from prometheus_client import Counter, Gauge, Histogram

# No FastAPI/Starlette imports here, and app/limiter.py never imports this
# module either - these are plain module-level objects that app/api.py
# updates at the points it already sees every relevant state transition,
# keeping the concurrency primitive (GenerationLimiter) free of any
# observability concern.

TTFT_SECONDS = Histogram("gpu_chat_ttft_seconds", "Time to first token, once generation actually starts")
TOKENS_PER_SECOND = Histogram("gpu_chat_tokens_per_second", "Generation throughput, derived from each done event")
QUEUE_DEPTH = Gauge("gpu_chat_queue_depth", "Requests currently waiting for a free generation slot")
QUEUE_WAIT_SECONDS = Histogram("gpu_chat_queue_wait_seconds", "Time spent waiting for a free generation slot")
ACTIVE_GENERATIONS = Gauge("gpu_chat_active_generations", "Generations currently running against Ollama")
ERRORS_TOTAL = Counter("gpu_chat_errors_total", "Stream errors, by code", ["code"])
