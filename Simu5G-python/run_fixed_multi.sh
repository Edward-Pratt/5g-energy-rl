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
INI="$GYM_DIR/omnetpp.ini"

# Absolute NEDPATH = robust regardless of cwd
NEDPATH="$ROOT/Simu5G-Gym/ned:$ROOT/Simu5G-Gym/src:$ROOT/Simu5G/src/simu5g:$ROOT/inet/src"

# Prefer a venv local to the python project
PY="$SCRIPT_DIR/.venv/bin/python"
SERVER="$SCRIPT_DIR/server_fixed.py"

# --- Parse command line arguments ---
NUM_EPISODES=1
ACTION=-1  # Default: run all actions
STAGE="all"  # 1|2|all
SCENARIO=""  # Explicit omnetpp.ini [Config ...] name

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
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --scenario)
      SCENARIO="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --episodes N     Run N episodes per action (default: 1)"
      echo "  --action A       Run only action A (0/1/2) (default: all)"
      echo "  --stage S        Stage 1, 2, or all (default: all)"
      echo "  --scenario NAME  Explicit omnetpp.ini config name"
      echo "  -h, --help       Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 --episodes 10                    # Stage1+Stage2, 10 episodes/action"
      echo "  $0 --stage 1 --episodes 5           # Stage1 only (Train-Simple)"
      echo "  $0 --stage 2 --action 0 --episodes 5 # Stage2 only, action=0"
      echo "  $0 --scenario Train-Multi --episodes 3"
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
[[ -f "$INI"    ]] || { echo "Missing omnetpp.ini: $INI" >&2; exit 1; }
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

# --- Determine which scenarios to run ---
if [[ -n "$SCENARIO" ]]; then
  SCENARIOS=("$SCENARIO")
else
  case "$STAGE" in
    1|stage1|Stage1)
      SCENARIOS=("Train-Simple")
      ;;
    2|stage2|Stage2)
      SCENARIOS=("Train-Multi")
      ;;
    all|both)
      SCENARIOS=("Train-Simple" "Train-Multi")
      ;;
    *)
      echo "Invalid --stage value: $STAGE (use: 1, 2, or all)" >&2
      exit 1
      ;;
  esac
fi

for scenario in "${SCENARIOS[@]}"; do
  if ! rg -q "^\[Config ${scenario}\]" "$INI"; then
    echo "Config [$scenario] not found in $INI" >&2
    exit 1
  fi
done

# --- Run fixed baseline for each scenario/action/episode ---
echo "Starting fixed baseline evaluation"
echo "  Episodes per action: $NUM_EPISODES"
echo "  Scenarios: ${SCENARIOS[*]}"
echo "  Actions: ${ACTIONS[*]}"
echo ""

for scenario in "${SCENARIOS[@]}"; do
  case "$scenario" in
    Train-Simple) stage_tag="stage1" ;;
    Train-Multi)  stage_tag="stage2" ;;
    *) stage_tag="$(echo "$scenario" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')" ;;
  esac

  echo "==> Scenario $scenario"

  for action in "${ACTIONS[@]}"; do
    # Keep per-stage logs separate so stage1/stage2 runs don't overwrite each other.
    LOG="$SCRIPT_DIR/fixed_${stage_tag}_${action}.csv"
    echo "  -> Action $action, $NUM_EPISODES episodes (log: $LOG)"

    # Start server ONCE - keep running for all episodes
    "$PY" "$SERVER" --action "$action" --out "$LOG" &
    SPID=$!

    sleep 1  # Give server time to start

    # Run all episodes in sequence (server persists, episode counter increments)
    for ep in $(seq 0 $((NUM_EPISODES - 1))); do
      echo "      Running episode $ep..."
      # run headless simulation
      "$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c "$scenario" -r 0
    done

    # Kill server after all episodes
    cleanup
  done
done

echo ""
echo "✅ Fixed baseline evaluation complete"
