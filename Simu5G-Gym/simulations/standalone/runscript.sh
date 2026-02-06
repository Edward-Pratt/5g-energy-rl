#!/usr/bin/env bash
set -euo pipefail

cd /home/edward/test-workspace/Simu5G-Gym/simulations/standalone

NEDPATH='../../ned:../../src:/home/edward/test-workspace/Simu5G/src:/home/edward/test-workspace/inet/src'
EXE='../../src/Simu5G-Gym'
PY='/home/edward/test-workspace/Simu5G-python/venv/bin/python'
SERVER='/home/edward/test-workspace/Simu5G-python/server_fixed.py'

for a in 0 1 2; do
  echo "=== Action $a ==="

  # start server fixed-action in background
  "$PY" "$SERVER" --action "$a" --out "rollout_action${a}.csv" &
  SPID=$!
  sleep 0.3

  # sanity check: if server died, fail fast
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "Server died. Check logs above."
    exit 1
  fi

  # run 10 repetitions headless
  CFG="VoIP-DL-A${a}"
  "$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c "$CFG" -r 0..9

  # stop server
  kill "$SPID" 2>/dev/null || true
  wait "$SPID" 2>/dev/null || true
done
