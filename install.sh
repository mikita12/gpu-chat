#!/usr/bin/env bash
# Run this once, from inside the cloned repo, on the machine that will host
# the chat (the box running Ollama, e.g. giga2). Sets up a venv, installs a
# user-level systemd service for the app, and a timer that auto-updates the
# repo from origin/main every 5 minutes (restarting the app if it changed).
set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

mkdir -p ~/.config/systemd/user
for unit in gpu-chat.service gpu-chat-update.service gpu-chat-update.timer; do
    sed "s#__REPO_DIR__#$REPO_DIR#g" "systemd/$unit" > ~/.config/systemd/user/"$unit"
done

systemctl --user daemon-reload
systemctl --user enable --now gpu-chat.service
systemctl --user enable --now gpu-chat-update.timer

if ! loginctl show-user "$(whoami)" 2>/dev/null | grep -q "Linger=yes"; then
    echo "NOTE: services stop when you log out. To keep them running, ask root to run:"
    echo "  sudo loginctl enable-linger $(whoami)"
fi

IP=$(hostname -I | awk '{print $1}')
echo
echo "Installed. Chat running at http://$IP:8000"
echo "This checks origin/main for updates every 5 min and restarts itself automatically - see README.md."
echo "Logs:   journalctl --user -u gpu-chat.service -f"
echo "Update log: journalctl --user -u gpu-chat-update.service -f"
