#!/usr/bin/env bash
# Run this once, from inside the cloned repo, on the machine that will host
# the chat (the box running Ollama, e.g. giga2). Sets up the first release
# (a git-worktree checkout + its own venv under releases/<sha>/, symlinked
# as `current`), a user-level systemd service for it, and a timer that
# health-gates future deploys from origin/main every 5 minutes - see
# update.sh and README.md for what that actually does.
set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

SHA=$(git rev-parse HEAD)
mkdir -p releases
# Sibling to releases/, not inside any one of them - see the DATABASE_URL
# comment in systemd/gpu-chat.service for why.
mkdir -p data
git worktree add "releases/$SHA" "$SHA"
python3 -m venv "releases/$SHA/.venv"
"releases/$SHA/.venv/bin/pip" install -q -r "releases/$SHA/requirements.txt"
ln -sfn "releases/$SHA" current

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
echo "Deploys are health-gated: every 5 min it builds origin/main as a new"
echo "release and checks it's actually healthy against Ollama on a scratch"
echo "port before switching over - a broken commit never touches the live"
echo "service. See README.md."
echo "Logs:   journalctl --user -u gpu-chat.service -f"
echo "Update log: journalctl --user -u gpu-chat-update.service -f"
