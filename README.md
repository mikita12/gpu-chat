# gpu-chat

Minimal web chat UI for a local Ollama model, meant to run **on the same
machine as Ollama** (e.g. giga2) so anyone on the LAN can open a browser and
prompt it — no SSH tunnel needed.

## What this is

- `app/main.py` — FastAPI backend. One endpoint, `POST /api/chat`, that
  streams straight through to Ollama's `/api/chat`.
- `app/static/index.html` — single-page chat UI, no build step, no
  dependencies.
- Talks to Ollama at `http://127.0.0.1:11434` by default (override with the
  `OLLAMA_URL` env var) and uses the `qwen3.8:27b` model by default (override
  with `OLLAMA_MODEL`) — both are set in `systemd/gpu-chat.service`.

## Stream event protocol

`POST /api/chat` responds with `application/x-ndjson`: one JSON object per
line, each with a `type` field. This is the contract between
`app/api.py` (backend, emits these) and `app/static/index.html` (frontend,
must handle them) - keeping both sides in sync here is the point of writing
it down.

| type | fields | backend guarantees | frontend must do |
|---|---|---|---|
| `content` | `text: string` | one per generated token/chunk, in order | append `text` to the displayed reply |
| `ping` | - | sent every `HEARTBEAT_SECONDS` (default 10s) while otherwise idle (e.g. during model load), purely to keep the connection alive | safe to ignore for content purposes; may use it to detect "still working" |
| `queued` | `position: number` | *(not yet emitted - Phase 3)* will be sent while a request waits for a free generation slot | show `position` in the status line; expect more `content`/`ping` once generation actually starts |
| `done` | `eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`, `load_duration`, `total_duration` (all nanoseconds except counts, all nullable) | sent exactly once, last, on a clean finish - and *only* then | treat its absence as a truncated/interrupted stream, not a clean end; the timing fields are for diagnostics (currently just logged to the console) |
| `error` | `message: string`, `code: string` | sent on any failure (upstream HTTP error, connection failure, malformed/unexpected data from Ollama, or a stall with no progress for `STALL_TIMEOUT_SECONDS`) - always terminates the stream | show `message`; do not treat the exchange as successful |

Example line: `{"type":"content","text":"Hello"}`

A stream that ends without ever sending `done` or `error` (connection just
dropped) must be treated by the frontend as interrupted - that's what the
`[stream interrupted]` marker in `index.html` is for.

## Install (run once, on giga2)

```bash
git clone https://github.com/mikita12/gpu-chat.git
cd gpu-chat
./install.sh
```

This creates a venv, installs a **user-level** systemd service
(`gpu-chat.service`) serving the app on `0.0.0.0:8000`, and a timer
(`gpu-chat-update.timer`) that checks `origin/main` for new commits every 5
minutes.

Open `http://<giga2-lan-ip>:8000` from any machine on the LAN.

## Heads up: it auto-updates itself

Every 5 minutes, `gpu-chat-update.timer` runs `update.sh`, which does
`git fetch` + compares to `origin/main`. If there's a new commit, it does a
**hard reset to `origin/main`** (any local edits to files in this checkout
will be discarded), reinstalls dependencies, and restarts the service. This
means pushing to `origin/main` deploys to giga2 within 5 minutes, with no
manual step on the server.

- Update log: `journalctl --user -u gpu-chat-update.service -f`
- App log: `journalctl --user -u gpu-chat.service -f`
- To pause auto-updates: `systemctl --user disable --now gpu-chat-update.timer`
- To turn them back on: `systemctl --user enable --now gpu-chat-update.timer`

If you need to change something on the server directly, either commit it to
the repo (so it survives the next update) or disable the timer first.

## Keeping it running after logout

User-level systemd services stop when you log out unless lingering is
enabled. `install.sh` will tell you if this is needed; a machine admin runs:

```bash
sudo loginctl enable-linger <your-username>
```

## Configuration

Env vars, set in `systemd/gpu-chat.service` (or passed directly when
running `uvicorn` by hand). All have sane defaults - nothing here is
required.

| Var | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where Ollama is reachable |
| `OLLAMA_MODEL` | `qwen3.8:27b` | Default model when a request doesn't specify one |
| `OLLAMA_CACHE_TTL_SECONDS` | `5.0` | How long `/api/tags`/`/api/show` responses are cached |
| `HEARTBEAT_SECONDS` | `10.0` | How often to ping an otherwise-idle stream |
| `STALL_TIMEOUT_SECONDS` | `90.0` | Give up if Ollama produces nothing for this long once generating |
| `MAX_CONCURRENT_GENERATIONS` | `1` | How many chats run against the GPU at once |
| `MAX_QUEUE_SIZE` | `10` | How many more requests may *wait* beyond that before getting HTTP 429 |
| `MAX_MESSAGES` | `50` | Reject a request with more messages than this |
| `MAX_MESSAGE_CHARS` | `8000` | Reject a request with any single message longer than this |
| `MAX_PROMPT_CHARS` | `24000` | Reject a request whose messages sum to more than this |
| `BEARER_TOKEN` | *(unset)* | See below |

Message history longer than the selected model's actual context window is
trimmed automatically (oldest non-system messages dropped first, using a
rough chars-per-token estimate - there's no real tokenizer for arbitrary
Ollama models) - that's separate from `MAX_MESSAGES`/`MAX_MESSAGE_CHARS`/
`MAX_PROMPT_CHARS` above, which are hard rejects rather than trimming.

## Security note

By default this has **no authentication** — anyone who can reach port 8000
on the LAN can prompt the model. Fine for a trusted home/office network; do
not expose this port beyond the LAN (e.g. via port-forwarding on a router)
without adding auth in front of it.

Setting `BEARER_TOKEN` turns on auth for `/api/models`, `/api/loaded`, and
`/api/chat` (not the static page itself) - callers must send
`Authorization: Bearer <token>`, e.g.:

```bash
curl http://<host>:8000/api/chat -H "Authorization: Bearer <token>" ...
```

**Note:** `app/static/index.html` has no way to enter or send a token - if
you set `BEARER_TOKEN`, the browser UI itself will get 401s unless you put
something in front of it that injects the header (a reverse proxy, a
browser extension, etc.). This is meant for protecting the API from
scripts/automation on a less-trusted network, not for gating the browser
chat UI on today's trusted-LAN deployment.
