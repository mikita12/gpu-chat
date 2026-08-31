#!/usr/bin/env bash
# Called periodically by the gpu-chat-update timer. Pulls latest from origin,
# reinstalls deps and restarts the service only if something actually changed.
# Do not hand-edit files in this repo checkout - a future auto-update will
# overwrite them (git reset --hard origin/main).
set -euo pipefail
cd "$(dirname "$0")"

git fetch origin
BEFORE=$(git rev-parse HEAD)
AFTER=$(git rev-parse origin/main)

if [ "$BEFORE" = "$AFTER" ]; then
    echo "$(date -Is) up to date ($BEFORE)"
    exit 0
fi

echo "$(date -Is) updating $BEFORE -> $AFTER"
git reset --hard origin/main
.venv/bin/pip install -q -r requirements.txt
systemctl --user restart gpu-chat.service
echo "$(date -Is) restarted gpu-chat.service"
