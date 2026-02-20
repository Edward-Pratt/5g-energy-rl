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
SERVER="$SCRIPT_DIR/server_fixed.py"

# --- Parse command line arguments ---
NUM_EPISODES=1
ACTION=-1  # Default: run all actions

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --action)
      ACTION="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --episodes N     Run N episodes per action (default: 1)"
      echo "  --action A       Run only action A (0/1/2) (default: all)"
      echo "  -h, --help       Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 --episodes 10        # Run 10 episodes for each action"
      echo "  $0 --action 0 --episodes 5  # Run 5 episodes with action=0 only"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use -h or --help for usage" >&2
      exit 1
      ;;
  esac
done

# --- sanity checks ---
[[ -d "$GYM_DIR" ]] || { echo "Missing GYM_DIR: $GYM_DIR" >&2; exit 1; }
[[ -x "$EXE"    ]] || { echo "Missing/Not executable EXE: $EXE" >&2; exit 1; }
[[ -f "$SERVER" ]] || { echo "Missing SERVER: $SERVER" >&2; exit 1; }
[[ -x "$PY"     ]] || { echo "Missing PY interpreter: $PY" >&2; echo "Tip: create it with: python3 -m venv $SCRIPT_DIR/.venv" >&2; exit 1; }

cd "$GYM_DIR"

cleanup() {
  if [[ -n "${SPID:-}" ]]; then
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    SPID=""
  fi
}
trap cleanup EXIT INT TERM

# --- Determine which actions to run ---
if [[ "$ACTION" -eq -1 ]]; then
  ACTIONS=(0 1 2)
else
  ACTIONS=("$ACTION")
fi

# --- Run fixed baseline for each action and episode ---
echo "Starting fixed baseline evaluation"
echo "  Episodes per action: $NUM_EPISODES"
echo "  Actions: ${ACTIONS[*]}"
echo ""

for action in "${ACTIONS[@]}"; do
  # All episodes append to same file per action
  LOG="$SCRIPT_DIR/fixed_${action}.csv"
  
  echo "==> Action $action, $NUM_EPISODES episodes (log: $LOG)"
  
  # Start server ONCE - keep running for all episodes
  "$PY" "$SERVER" --action "$action" --out "$LOG" &
  SPID=$!
  
  sleep 1  # Give server time to start
  
  # Run all episodes in sequence (server persists, episode counter increments)
  for ep in $(seq 0 $((NUM_EPISODES - 1))); do
    echo "    Running episode $ep..."
    # run headless simulation
    "$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c CBR-DL-PPO -r 0
  done
  
  # Kill server after all episodes
  cleanup
done

echo ""
echo "✅ Fixed baseline evaluation complete"
