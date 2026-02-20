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

# --- Parse command line arguments ---
NUM_EPISODES=1
MODE="eval"
GREEDY=false
CHECKPOINT="$CKPT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --greedy)
      GREEDY=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --episodes N       Run N episodes (default: 1)"
      echo "  --mode MODE        Run in 'eval' or 'train' mode (default: eval)"
      echo "  --checkpoint FILE  Use checkpoint FILE (default: ppo_policy.pt)"
      echo "  --greedy           Use deterministic policy (only for eval mode)"
      echo "  -h, --help         Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 --episodes 10              # Run 10 evaluation episodes"
      echo "  $0 --episodes 5 --greedy      # Run 5 episodes with deterministic policy"
      echo "  $0 --mode train --episodes 100  # Continue training for 100 episodes"
      echo "  $0 --checkpoint my_policy.pt # Use custom checkpoint"
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

# Check checkpoint for eval mode
if [[ "$MODE" == "eval" && ! -f "$CHECKPOINT" ]]; then
  echo "Error: Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

cd "$GYM_DIR"

cleanup() {
  if [[ -n "${SPID:-}" ]]; then
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    SPID=""
  fi
}
trap cleanup EXIT INT TERM

# --- Build server command ---
SERVER_CMD=("$PY" "$SERVER" --mode "$MODE" --checkpoint "$CHECKPOINT")

# Add greedy flag if specified and in eval mode
if [[ "$GREEDY" == true && "$MODE" == "eval" ]]; then
  SERVER_CMD+=(--greedy_eval)
fi

# Log file naming
if [[ "$MODE" == "eval" ]]; then
  LOG_BASE="${CHECKPOINT%.pt}"
  LOG_BASE="${LOG_BASE##*/}"  # Extract basename without path
  LOG_BASE="eval_${LOG_BASE}"
else
  LOG_BASE="rollout_ppo"
fi

# Define log file path
LOG="$SCRIPT_DIR/${LOG_BASE}.csv"

echo "Starting PPO $MODE"
echo "  Episodes: $NUM_EPISODES"
echo "  Mode: $MODE"
echo "  Checkpoint: $CHECKPOINT"
[[ "$GREEDY" == true ]] && echo "  Policy: Deterministic (greedy)"
echo "  Log file: $LOG (all episodes appended)"
echo ""

# Start server ONCE - keep running for all episodes
FULL_SERVER_CMD=("${SERVER_CMD[@]}" --log "$LOG")
"${FULL_SERVER_CMD[@]}" &
SPID=$!

sleep 1  # Give server time to start

# Run all episodes in sequence (server persists, episode counter increments)
for ep in $(seq 0 $((NUM_EPISODES - 1))); do
  echo "==> Episode $ep/$((NUM_EPISODES - 1))"
  
  # run headless simulation
  "$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c CBR-DL-PPO -r 0
done

# Kill server after all episodes
cleanup

echo ""
echo "✅ PPO $MODE complete"
echo "📊 Log file: $LOG"
