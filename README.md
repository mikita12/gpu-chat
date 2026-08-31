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

## Install (run once, on giga2)

```bash
git clone git@github.com:mikita12/gpu-chat.git
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

## Security note

This has **no authentication** — anyone who can reach port 8000 on the LAN
can prompt the model. Fine for a trusted home/office network; do not expose
this port beyond the LAN (e.g. via port-forwarding on a router) without
adding auth in front of it.
