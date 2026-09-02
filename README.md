# gpu-chat

Multi-user web chat for a local Ollama model, with per-user accounts and
persistent conversation history. Built to run on a small, memory-constrained
box (e.g. a Raspberry Pi) that just proxies to Ollama running on a separate
GPU machine over the LAN - the app itself never runs the model.

## Architecture

```
   browser  <--LAN-->  gpu-chat (this app)  <--LAN-->  Ollama (e.g. giga2)
                        - FastAPI + SQLite            - runs the model
                        - accounts, sessions          - the only GPU load
                        - conversation history
```

- `app/main.py` - FastAPI app. Serves the static frontend, the account/
  conversation API, and proxies generation to Ollama.
- `app/static/index.html` - the chat UI. `app/static/login.html` - the
  account creation/sign-in page. Both are single files, no build step, no
  frontend dependencies.
- `OLLAMA_URL` (default `http://127.0.0.1:11434`) is just an HTTP URL - it
  does not need to point at `localhost`. Point it at Ollama's LAN address
  (e.g. `http://giga2.lan:11434`) to run gpu-chat itself on a separate,
  smaller machine. Nothing else about the app changes based on where
  Ollama runs.
- Expected resident memory for the app process itself is small (well under
  100 MB at idle; SQLite and a single shared `httpx.AsyncClient` are the
  only persistent resources) - the model's memory footprint lives entirely
  on the Ollama machine, not here. See "Memory discipline" below for the
  specific knobs that keep it that way under load.

## Accounts and the LAN-trust security model

Every conversation is private to the account that created it - reading or
deleting someone else's conversation isn't just hidden by the UI, it's
rejected by the API itself (a mismatched owner and a nonexistent
conversation both come back `404`, so a guess can't even confirm another
account's conversation IDs exist).

What this **is**: real authentication - passwords hashed with argon2id
(never stored or logged in plaintext), sessions stored server-side in
SQLite (not memory, so they survive a process restart) and presented via
an `HttpOnly`, `SameSite=Lax` cookie.

What this **is not**: internet-facing hardening. There's no email
verification, no password reset flow, and no registration gating - anyone
who can reach the app's port can create an account from `/login.html`.
This is meant for a trusted LAN (home/office), the same trust model the
prototype's original no-auth-at-all version assumed - the difference is
that *data* is now private per account, not that the network perimeter
changed. Don't expose this port to the internet without adding real
perimeter security in front of it.

Creating the first account: open `/login.html`, use the "Create account"
tab. There's no separate admin/setup step - the first account created is
just an account like any other.

The separate `BEARER_TOKEN` mechanism (below) is unrelated to accounts -
it's for scripted/curl access to the stateless `/api/chat` endpoint, and
is checked independently of any session cookie.

## Stream event protocol

Both `POST /api/chat` (stateless, bearer-gated) and
`POST /api/conversations/{id}/messages` / `.../retry` (session-gated,
persisted) respond with `application/x-ndjson`: one JSON object per line,
each with a `type` field. This is the contract between `app/api.py` /
`app/conversations_api.py` (backend, emit these) and `app/static/index.html`
(frontend, must handle them).

| type | fields | backend guarantees | frontend must do |
|---|---|---|---|
| `content` | `text: string` | one per generated reply chunk, in order | append `text` to the displayed reply |
| `thinking` | `text: string` | present only for a reasoning model that returns a thinking trace; interleaved with `content`, always before the reply text it corresponds to | show separately from the reply (this app renders it as a collapsed disclosure) - never treat it as reply text |
| `ping` | - | sent every `HEARTBEAT_SECONDS` (default 10s) while otherwise idle (e.g. during model load or while queued), purely to keep the connection alive | safe to ignore for content purposes; may use it to detect "still working" |
| `queued` | `position: number` | sent while a request waits for a free generation slot (`MAX_CONCURRENT_GENERATIONS`), re-sent whenever `position` changes | show `position` in the status line; expect a `generating` event once a slot is acquired |
| `generating` | - | sent exactly once, the moment a generation slot is actually acquired - before any `content`/`thinking`/`ping`. Needed because `ping` fires during *both* the queueing and generating phases: without this, a client that was queued has no way to tell "still queued, no position change yet" from "already generating, model still loading" | transition out of the queued UI state on receipt, even though no content has arrived yet |
| `done` | `eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`, `load_duration`, `total_duration` (all nanoseconds except counts, all nullable), `request_id: string \| null` | sent exactly once, last, on a clean finish - and *only* then | treat its absence as a truncated/interrupted stream, not a clean end |
| `error` | `message: string`, `code: string`, `request_id: string \| null` | sent on any failure - always terminates the stream. Codes: `stall` (no progress for `STALL_TIMEOUT_SECONDS`), `connection` (couldn't reach Ollama), `upstream_unavailable`, `queue_full`, `oom`, `generation_failed`, `unknown_model`, `invalid_request`, `unauthorized`, `not_found`, `protocol`, `unknown`, plus `upstream_http_<status>` for an unclassified non-2xx from Ollama | show `message`; do not treat the exchange as successful. `stall`/`connection`/`upstream_unavailable`/`queue_full` are worth retrying as-is; the rest generally aren't |

`request_id` (on `done`/`error`) identifies the request in the server's
structured logs (see Observability below).

For the persisted, per-conversation endpoints specifically: the user's
message is saved the moment it's sent, regardless of what happens next.
The assistant's reply is saved **only** on a clean `done` - an abort,
timeout, or mid-stream error leaves that turn without a reply rather than
saving a truncated one, which would otherwise poison every later turn's
context with a partial "answer". `POST .../retry` regenerates a reply for
that still-unanswered turn without creating a duplicate user message.

A stream that ends without ever sending `done` or `error` (connection just
dropped) must be treated by the frontend as interrupted.

## Install (run once, on the machine that will host the app)

```bash
git clone https://github.com/mikita12/gpu-chat.git
cd gpu-chat
./install.sh
```

This creates the first release under `releases/<git-sha>/` (its own git
checkout + venv, symlinked as `current`), a **user-level** systemd service
(`gpu-chat.service`) serving `current` on `0.0.0.0:8000`, a `data/`
directory for the SQLite database (see "Where the database lives" below),
and a timer (`gpu-chat-update.timer`) that checks `origin/main` for new
commits every 5 minutes. Edit `Environment=OLLAMA_URL=...` in
`systemd/gpu-chat.service` before installing if Ollama isn't on the same
machine.

Open `http://<host-lan-ip>:8000` from any machine on the LAN.

## Where the database lives (read this before customizing the deploy)

Every release is its own disposable `git worktree` under `releases/<sha>/`,
pruned automatically once superseded (see "it auto-updates itself" below).
**The database must never live inside one of these** - it would be deleted
on the next deploy. `DATABASE_URL` in `systemd/gpu-chat.service` points at
`data/gpu-chat.db`, a sibling of `releases/` and `current` that survives
every deploy; `install.sh` creates that directory once. If you run
`uvicorn` by hand outside systemd, the default (`./gpu_chat.db`, relative
to the working directory) is fine for local development but **not** for a
real deployment using the release-worktree layout above.

Migrations (Alembic) run automatically at process startup, including
inside `update.sh`'s scratch health-check instance - a migration that
fails keeps `/readyz` from ever returning `200`, so a broken migration is
caught by the same health gate as any other broken deploy, before it ever
reaches the live service.

## Heads up: it auto-updates itself - but only if the new version is healthy

Every 5 minutes, `gpu-chat-update.timer` runs `update.sh`, which does
`git fetch` and compares `origin/main` against whatever `current` points at.
If there's a new commit, it does **not** touch the live service right away:

1. Builds the new commit as its own release (`releases/<sha>/` - a fresh
   git worktree + venv).
2. Starts *that* release on a scratch port (`127.0.0.1:8099`) using the
   same Ollama and database config as the real service (including running
   its migrations - see above), and polls its `GET /readyz` for up to 60s.
3. **Only if that succeeds**, repoints `current` at the new release and
   restarts `gpu-chat.service` (one real restart - not a zero-downtime
   cutover).
4. If it fails to build, fails its migration, or fails the health check,
   the candidate release is discarded and the currently-running one just
   keeps serving, untouched - the failure is visible as a non-zero exit in
   the update log, but nothing about the live chat changes.

This means pushing to `origin/main` deploys within 5 minutes *if* the new
commit is actually healthy, with no manual step on the server. Old
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
required to get started.

| Var | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where Ollama is reachable - point this at a LAN address if Ollama runs on a different machine than this app |
| `OLLAMA_MODEL` | `qwen3.8:27b` | Default model for a new conversation when one isn't specified |
| `OLLAMA_CACHE_TTL_SECONDS` | `5.0` | How long `/api/tags`/`/api/show` responses are cached |
| `HEARTBEAT_SECONDS` | `10.0` | How often to ping an otherwise-idle stream |
| `STALL_TIMEOUT_SECONDS` | `90.0` | Give up if Ollama produces nothing for this long once generating |
| `STREAM_QUEUE_MAXSIZE` | `16` | Internal buffer between Ollama's response and the client - kept small deliberately: a slow/backgrounded client should only ever hold a handful of events in *this process's* RAM, not accumulate unbounded backlog |
| `MAX_CONCURRENT_GENERATIONS` | `1` | How many chats run against the GPU at once |
| `MAX_QUEUE_SIZE` | `10` | How many more requests may *wait* beyond that before getting HTTP 429 |
| `MAX_MESSAGES` | `50` | Reject a request with more messages than this (stateless `/api/chat` only) |
| `MAX_MESSAGE_CHARS` | `8000` | Reject any single message longer than this |
| `MAX_PROMPT_CHARS` | `24000` | Reject a request whose messages sum to more than this (stateless `/api/chat` only) |
| `BEARER_TOKEN` | *(unset)* | See "Security note" below - unrelated to accounts |
| `DATABASE_URL` | `sqlite+aiosqlite:///./gpu_chat.db` | Where the accounts/sessions/conversations database lives - **must** be overridden to a stable path outside `releases/` in any real deployment, see above |
| `SESSION_COOKIE_NAME` | `gpu_chat_session` | Name of the session cookie |
| `SESSION_TTL_SECONDS` | `2592000` (30 days) | How long a session stays valid after login |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `3600` (1 hour) | How often a background task sweeps expired session rows out of SQLite |

Message history longer than the selected model's actual context window is
trimmed automatically (oldest non-system messages dropped first, using a
rough chars-per-token estimate - there's no real tokenizer for arbitrary
Ollama models) for both the stateless and persisted chat paths - that's
separate from `MAX_MESSAGES`/`MAX_MESSAGE_CHARS`/`MAX_PROMPT_CHARS` above,
which are hard rejects on the stateless path rather than trimming.

## Memory discipline

This app is meant to run comfortably on a 512 MB box while Ollama and the
model live elsewhere. The concrete things that keep it that way:

- One shared `httpx.AsyncClient` for the process lifetime (`app/main.py`'s
  `lifespan()`), not one per request - avoids a new connection pool per chat.
- SQLite opened with `PRAGMA journal_mode=WAL` and a small, fixed
  connection pool (`pool_size=5, max_overflow=0` - `app/db.py`) - SQLite
  only ever allows one writer at a time regardless, so a large pool buys
  nothing.
- `STREAM_QUEUE_MAXSIZE` (default 16, see above) bounds how much of a
  reply can sit buffered in this process waiting for a slow client to read
  it, rather than growing unbounded.
- Session cleanup runs as a lightweight periodic sweep (default hourly),
  not an in-process cache that grows with every login.

## Observability

Three endpoints, all unauthenticated regardless of `BEARER_TOKEN` or
session cookies (a health check or metrics scrape shouldn't need credentials):

- `GET /healthz` - process liveness only, no Ollama call.
- `GET /readyz` - process liveness *and* Ollama is reachable (does not
  check the database). What `update.sh` polls before cutting a deploy over.
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

Accounts make conversation data private *between users of this
deployment* - they are not internet-facing hardening (see "Accounts and
the LAN-trust security model" above). Do not expose this port beyond the
LAN (e.g. via port-forwarding on a router) without adding real
authentication/TLS in front of it.

Setting `BEARER_TOKEN` turns on bearer-token enforcement for `/api/models`,
`/api/loaded`, and the stateless `/api/chat` (not the static pages, and
not the session-gated `/api/auth/*` / `/api/conversations/*` routes, which
have their own, independent auth) - callers must send
`Authorization: Bearer <token>`, e.g.:

```bash
curl http://<host>:8000/api/chat -H "Authorization: Bearer <token>" ...
```

This is meant for protecting the stateless API from scripts/automation on
a less-trusted network - it has nothing to do with per-user accounts and
is checked independently of the session cookie.
