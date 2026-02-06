#!/usr/bin/env bash
set -euo pipefail

cd /home/edward/test-workspace/Simu5G-Gym/simulations/standalone

EXE='/home/edward/test-workspace/Simu5G-Gym/src/Simu5G-Gym'
NEDPATH='/home/edward/test-workspace/Simu5G-Gym/ned:/home/edward/test-workspace/Simu5G-Gym/src:/home/edward/test-workspace/Simu5G/src:/home/edward/test-workspace/inet/src'

PY='/home/edward/test-workspace/venv/bin/python'
SERVER='/home/edward/test-workspace/Simu5G-python/serverppo.py'

# start PPO server
"$PY" "$SERVER" &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT
sleep 0.2

# run OMNeT headless
"$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c VoIP-DL-PPO
