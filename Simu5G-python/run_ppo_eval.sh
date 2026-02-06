#!/usr/bin/env bash
set -euo pipefail

# Script is inside: <root>/Simu5G-python/
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OMNET="$ROOT/omnetpp-6.3.0"

# OMNeT++ tools (opp_run, opp_configfilepath, etc.)
export PATH="$OMNET/bin:$PATH"

# OMNeT++ shared libs (liboppcommon.so lives here)
_omnet_ld=""
[[ -d "$OMNET/lib"   ]] && _omnet_ld="$OMNET/lib"
[[ -d "$OMNET/lib64" ]] && _omnet_ld="${_omnet_ld:+$_omnet_ld:}$OMNET/lib64"
export LD_LIBRARY_PATH="${_omnet_ld}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# If your Makefiles ever need it
export OMNETPP_CONFIGFILE="$OMNET/Makefile.inc"

GYM_DIR="$ROOT/Simu5G-Gym/simulations/standalone"
EXE="$ROOT/Simu5G-Gym/src/Simu5G-Gym"

# Absolute NEDPATH = robust regardless of cwd
NEDPATH="$ROOT/Simu5G-Gym/ned:$ROOT/Simu5G-Gym/src:$ROOT/Simu5G/src/simu5g:$ROOT/inet/src"

# Prefer a venv local to the python project
PY="$SCRIPT_DIR/.venv/bin/python"
SERVER="$SCRIPT_DIR/serverppo.py"
CKPT="$SCRIPT_DIR/ppo_policy.pt"
LOG="$SCRIPT_DIR/eval_ppo.csv"

# --- sanity checks ---
[[ -d "$GYM_DIR" ]] || { echo "Missing GYM_DIR: $GYM_DIR" >&2; exit 1; }
[[ -x "$EXE"    ]] || { echo "Missing/Not executable EXE: $EXE" >&2; exit 1; }
[[ -f "$SERVER" ]] || { echo "Missing SERVER: $SERVER" >&2; exit 1; }
[[ -x "$PY"     ]] || { echo "Missing PY interpreter: $PY" >&2; echo "Tip: create it with: python3 -m venv $SCRIPT_DIR/.venv" >&2; exit 1; }

# start eval server
"$PY" "$SERVER" --mode eval --checkpoint "$CKPT" --log "$LOG" &
SPID=$!
trap 'kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true' EXIT INT TERM

cd "$GYM_DIR"

# run headless
"$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c VoIP-DL-PPO
