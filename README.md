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
| `ping` | - | sent every `HEARTBEAT_SECONDS` (default 10s) while otherwise idle (e.g. during model load or while queued), purely to keep the connection alive | safe to ignore for content purposes; may use it to detect "still working" |
| `queued` | `position: number` | sent while a request waits for a free generation slot (`MAX_CONCURRENT_GENERATIONS`), re-sent whenever `position` changes | show `position` in the status line; expect a `generating` event once a slot is acquired |
| `generating` | - | sent exactly once, the moment a generation slot is actually acquired - before any `content`/`ping`. Needed because `ping` fires during *both* the queueing and generating phases: without this, a client that was queued has no way to tell "still queued, no position change yet" from "already generating, model still loading" - both look identical on the wire (a `queued` event once, then a run of bare pings). Found live: a genuinely queued browser tab stayed stuck showing "Queued" after its turn had actually started | transition out of the queued UI state on receipt, even though no content has arrived yet |
| `done` | `eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`, `load_duration`, `total_duration` (all nanoseconds except counts, all nullable), `request_id: string \| null` | sent exactly once, last, on a clean finish - and *only* then | treat its absence as a truncated/interrupted stream, not a clean end; the timing fields are for diagnostics (currently just logged to the console) |
| `error` | `message: string`, `code: string`, `request_id: string \| null` | sent on any failure (upstream HTTP error, connection failure, malformed/unexpected data from Ollama, or a stall with no progress for `STALL_TIMEOUT_SECONDS`) - always terminates the stream | show `message`; do not treat the exchange as successful |

`request_id` (on `done`/`error`) identifies the request in the server's
structured logs (see Observability below) - the same id is stable across
every event one request emits, and differs between requests.

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

This creates the first release under `releases/<git-sha>/` (its own git
checkout + venv, symlinked as `current`), a **user-level** systemd service
(`gpu-chat.service`) serving `current` on `0.0.0.0:8000`, and a timer
(`gpu-chat-update.timer`) that checks `origin/main` for new commits every 5
minutes.

Open `http://<giga2-lan-ip>:8000` from any machine on the LAN.

## Heads up: it auto-updates itself - but only if the new version is healthy

Every 5 minutes, `gpu-chat-update.timer` runs `update.sh`, which does
`git fetch` and compares `origin/main` against whatever `current` points at.
If there's a new commit, it does **not** touch the live service right away:

1. Builds the new commit as its own release (`releases/<sha>/` - a fresh
   git worktree + venv).
2. Starts *that* release on a scratch port (`127.0.0.1:8099`) using the
   same Ollama config as the real service, and polls its `GET /readyz`
   for up to 60s.
3. **Only if that succeeds**, repoints `current` at the new release and
   restarts `gpu-chat.service` (one real restart - not a zero-downtime
   cutover).
4. If it fails to build or fails the health check, the candidate release
   is discarded and the currently-running one just keeps serving,
   untouched - the failure is visible as a non-zero exit in the update
   log, but nothing about the live chat changes.

This means pushing to `origin/main` deploys to giga2 within 5 minutes *if*
the new commit is actually healthy, with no manual step on the server. Old
releases are pruned automatically, keeping only the current one and the
immediately-previous one (a fast manual-rollback target: repoint `current`
back with `ln -sfn releases/<old-sha> current` and
`systemctl --user restart gpu-chat.service`).

- Update log: `journalctl --user -u gpu-chat-update.service -f`
- App log: `journalctl --user -u gpu-chat.service -f`
- To pause auto-updates: `systemctl --user disable --now gpu-chat-update.timer`
- To turn them back on: `systemctl --user enable --now gpu-chat-update.timer`

If you need to change something on the server directly, either commit it to
the repo (so it survives the next update) or disable the timer first - and
note any change should go in the top-level checkout (where `install.sh`
was run), not inside a `releases/<sha>/` directory, which is disposable.

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

## Observability

Three endpoints, all unauthenticated regardless of `BEARER_TOKEN` (a
health check or metrics scrape shouldn't need API credentials):

- `GET /healthz` - process liveness only, no Ollama call.
- `GET /readyz` - process liveness *and* Ollama is reachable. What
  `update.sh` polls before cutting a deploy over.
- `GET /metrics` - Prometheus exposition format:
  `gpu_chat_ttft_seconds` (time to first token), `gpu_chat_tokens_per_second`,
  `gpu_chat_queue_depth` (currently waiting for a slot),
  `gpu_chat_queue_wait_seconds`, `gpu_chat_active_generations`, and
  `gpu_chat_errors_total{code=...}`.

The app also logs structured JSON (one object per line: timestamp, level,
message, and `request_id` when set) to stdout under the `gpu_chat` logger
name - request start/failure events are tagged with the same `request_id`
that shows up in that request's `done`/`error` stream events, so a
user-reported failure can be grepped straight to its log line
(`journalctl --user -u gpu-chat.service | grep <request_id>`). This is
separate from uvicorn's own access/error logging, which is untouched.

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
