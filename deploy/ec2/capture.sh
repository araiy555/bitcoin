#!/usr/bin/env bash
# Record one book (with Binance as its lead) for a day, then exit so systemd
# starts the next day's run.   Usage: capture.sh bitbank:ada_jpy
set -euo pipefail
venue=${1%%:*}
symbol=${1#*:}
out=/var/lib/jsboard/${venue}-${symbol}.jsonl
case "$venue" in
  bitbank) cmd=(bbcapture --pair "$symbol") ;;
  gmo) cmd=(gmocapture --symbol "$symbol") ;;
  *) echo "unknown venue: $venue" >&2; exit 2 ;;
esac
exec /opt/jsboard/venv/bin/jsboard "${cmd[@]}" --duration 86400 --out "$out" \
  --s3-bucket "$JSBOARD_BUCKET" --s3-prefix raw/live --rotate-minutes 5
