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
CKPT="$SCRIPT_DIR/ppo_policy.pt"
LOG="$SCRIPT_DIR/eval_ppo.csv"
STAGE="${STAGE:-all}"      # 1|2|all
SCENARIO="${SCENARIO:-}"   # Explicit omnetpp.ini [Config ...] name

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
      echo "Invalid STAGE value: $STAGE (use: 1, 2, or all)" >&2
      exit 1
      ;;
  esac
fi

for scenario in "${SCENARIOS[@]}"; do
  if ! rg -q "^\[Config ${scenario}\]" "$INI"; then
    echo "Config [$scenario] not found in $INI" >&2
    exit 1
  fi

  case "$scenario" in
    Train-Simple) stage_tag="stage1" ;;
    Train-Multi)  stage_tag="stage2" ;;
    *) stage_tag="$(echo "$scenario" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '_')" ;;
  esac

  echo "==> Scenario $scenario"
  for action in 0 1 2; do
    echo "  -> Running fixed action ${action}"
    "$PY" "$SERVER" --action "$action" --out "$SCRIPT_DIR/fixed_${stage_tag}_${action}.csv" &
    SPID=$!

    # run headless
    "$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c "$scenario" -r 0

    cleanup
  done
done
