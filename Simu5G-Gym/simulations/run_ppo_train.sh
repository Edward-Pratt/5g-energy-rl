#!/usr/bin/env bash
set -euo pipefail

cd ~/test-workspace/Simu5G-Gym/simulations/standalone

EXE='../../src/Simu5G-Gym'
NEDPATH='../../ned:../../src:~/test-workspace/Simu5G/src:~/test-workspace/inet/src'
PY=~/test-workspace/Simu5G-python/venv/bin/python
SERVER=~/test-workspace/Simu5G-python/serverppo.py

# start PPO server
$PY "$SERVER" --mode train --seed 0 --log rollout_ppo_train.csv --checkpoint ppo_policy.pt &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT
sleep 0.2

# run OMNeT headless, with repeats configured in ini
$EXE -u Cmdenv -n "$NEDPATH" omnetpp.ini -c VoIP-DL-PPO
