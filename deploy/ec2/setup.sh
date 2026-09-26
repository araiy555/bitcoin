#!/usr/bin/env bash
# One-time setup on Amazon Linux 2023. Run from a clone of the repository:
#   bash deploy/ec2/setup.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)

sudo dnf install -y git python3.11 python3.11-pip
id jsboard >/dev/null 2>&1 || sudo useradd --system --create-home --home-dir /opt/jsboard jsboard
sudo rsync -a --delete --exclude .venv "$repo/" /opt/jsboard/app/ 2>/dev/null \
  || { sudo mkdir -p /opt/jsboard/app && sudo cp -a "$repo/." /opt/jsboard/app/; }
sudo chown -R jsboard:jsboard /opt/jsboard
sudo -u jsboard python3.11 -m venv /opt/jsboard/venv
sudo -u jsboard /opt/jsboard/venv/bin/pip install --quiet -e '/opt/jsboard/app[s3]'

sudo install -m 755 "$here/capture.sh" /opt/jsboard/capture.sh
sudo install -m 644 "$here/jsboard-capture@.service" "$here/jsboard-daily.service" \
  "$here/jsboard-daily.timer" /etc/systemd/system/
if [ ! -f /etc/jsboard.env ]; then
  sudo install -m 600 "$here/jsboard.env.example" /etc/jsboard.env
fi
sudo systemctl daemon-reload
echo "done. next: sudo nano /etc/jsboard.env  (paste the Slack URL), then enable the services."
