#!/usr/bin/env bash
# Called periodically by the gpu-chat-update timer. Fetches origin/main; if
# there's a new commit, builds it as an isolated release (a git worktree +
# its own venv under releases/<sha>/), health-checks it against the real
# Ollama backend on a scratch port BEFORE touching the live service, and
# only then cuts over by repointing the `current` symlink and restarting.
# A candidate that fails to build OR fails its health check is discarded
# (via the cleanup trap below) and whatever is already running keeps
# serving, untouched.
#
# Do not hand-edit files inside a releases/<sha>/ checkout - each one is a
# disposable worktree that gets pruned once it's no longer current or the
# fallback-previous release.
set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"
CURRENT_LINK="$REPO_DIR/current"
RELEASES_DIR="$REPO_DIR/releases"
SCRATCH_PORT=8099
READYZ_TIMEOUT_SECONDS=60

git fetch origin
NEW_SHA=$(git rev-parse origin/main)

CURRENT_SHA=""
if [ -L "$CURRENT_LINK" ]; then
    CURRENT_SHA=$(basename "$(readlink -f "$CURRENT_LINK")")
fi

if [ "$CURRENT_SHA" = "$NEW_SHA" ]; then
    echo "$(date -Is) up to date ($NEW_SHA)"
    exit 0
fi

NEW_RELEASE="$RELEASES_DIR/$NEW_SHA"
CUTOVER_DONE=0
SCRATCH_PID=""

# Fires on ANY exit from here on - success, a `set -e` failure (bad
# requirements.txt, venv creation failure, ...), or an explicit `exit 1` on
# a failed health check. Kills the scratch instance if one is still up, and
# discards the candidate release unless it actually got cut over to.
cleanup() {
    if [ -n "$SCRATCH_PID" ]; then
        kill "$SCRATCH_PID" 2>/dev/null || true
        wait "$SCRATCH_PID" 2>/dev/null || true
    fi
    if [ "$CUTOVER_DONE" -ne 1 ]; then
        git worktree remove --force "$NEW_RELEASE" 2>/dev/null || rm -rf "$NEW_RELEASE"
    fi
}
trap cleanup EXIT

echo "$(date -Is) building candidate $NEW_SHA (current: ${CURRENT_SHA:-none})"
rm -rf "$NEW_RELEASE"  # in case a previous failed attempt left debris
mkdir -p "$RELEASES_DIR"
git worktree add --force "$NEW_RELEASE" "$NEW_SHA"
python3 -m venv "$NEW_RELEASE/.venv"
"$NEW_RELEASE/.venv/bin/pip" install -q -r "$NEW_RELEASE/requirements.txt"

# Health-check against the *actually configured* Ollama, not assumed
# defaults - read it out of the already-installed unit so a customized
# deployment gets checked against its real backend.
UNIT_FILE="$HOME/.config/systemd/user/gpu-chat.service"
if [ -f "$UNIT_FILE" ]; then
    eval "$(grep '^Environment=' "$UNIT_FILE" | sed 's/^Environment=/export /')"
fi

echo "$(date -Is) health-checking candidate on 127.0.0.1:$SCRATCH_PORT"
(cd "$NEW_RELEASE" && exec "./.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$SCRATCH_PORT") &
SCRATCH_PID=$!

HEALTHY=0
for _ in $(seq 1 "$READYZ_TIMEOUT_SECONDS"); do
    if python3 -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$SCRATCH_PORT/readyz', timeout=2)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        HEALTHY=1
        break
    fi
    sleep 1
done

kill "$SCRATCH_PID" 2>/dev/null || true
wait "$SCRATCH_PID" 2>/dev/null || true
SCRATCH_PID=""

if [ "$HEALTHY" -ne 1 ]; then
    echo "$(date -Is) candidate $NEW_SHA failed /readyz within ${READYZ_TIMEOUT_SECONDS}s - discarding it, ${CURRENT_SHA:-nothing} keeps running"
    exit 1
fi

echo "$(date -Is) candidate healthy - cutting over ${CURRENT_SHA:-none} -> $NEW_SHA"
ln -sfn "$NEW_RELEASE" "$CURRENT_LINK"
systemctl --user restart gpu-chat.service
CUTOVER_DONE=1
echo "$(date -Is) restarted gpu-chat.service on $NEW_SHA"

# Keep only the new release and the immediately-previous one (a fast
# manual-rollback target: repoint current back and restart); prune the rest.
for dir in "$RELEASES_DIR"/*/; do
    sha=$(basename "$dir")
    if [ "$sha" != "$NEW_SHA" ] && [ "$sha" != "$CURRENT_SHA" ]; then
        echo "$(date -Is) pruning old release $sha"
        git worktree remove --force "$dir" 2>/dev/null || rm -rf "$dir"
    fi
done
