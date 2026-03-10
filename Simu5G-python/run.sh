#!/usr/bin/env bash
set -euo pipefail

# Script is inside: <root>/Simu5G-python/
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OMNET="$ROOT/omnetpp-6.3.0"

# OMNeT++ tools
export PATH="$OMNET/bin:$PATH"

# OMNeT++ shared libs
_omnet_ld=""
[[ -d "$OMNET/lib"   ]] && _omnet_ld="$OMNET/lib"
[[ -d "$OMNET/lib64" ]] && _omnet_ld="${_omnet_ld:+$_omnet_ld:}$OMNET/lib64"
export LD_LIBRARY_PATH="${_omnet_ld}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export OMNETPP_CONFIGFILE="$OMNET/Makefile.inc"

GYM_DIR="$ROOT/Simu5G-Gym/simulations/standalone"
EXE="$ROOT/Simu5G-Gym/src/Simu5G-Gym"
NEDPATH="$ROOT/Simu5G-Gym/ned:$ROOT/Simu5G-Gym/src:$ROOT/Simu5G/src/simu5g:$ROOT/inet/src"

PY="$SCRIPT_DIR/.venv/bin/python"

# ── Defaults ────────────────────────────────────────────────────────────────
AGENT="ppo"            # ppo | dqn
NUM_EPISODES=1
MODE="eval"
GREEDY=false
CHECKPOINT=""          # set automatically if not supplied via --checkpoint

# DQN-specific overrides (passed through only when AGENT=dqn)
DQN_N_ACTIONS=21
DQN_LR="1e-3"
DQN_GAMMA="0.99"
DQN_BUF_SIZE=50000
DQN_BATCH_SIZE=64
DQN_TARGET_UPDATE=500
DQN_EPS_START="1.0"
DQN_EPS_END="0.05"
DQN_EPS_DECAY=10000

# PPO-specific overrides (passed through only when AGENT=ppo)
PPO_UPDATE_EVERY=512
PPO_SAVE_EVERY=1

# ── Usage ────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: \$0 [OPTIONS]

General options:
  --agent AGENT          Which agent to use: ppo | dqn  (default: ppo)
  --episodes N           Number of episodes to run      (default: 1)
  --mode MODE            train | eval                   (default: eval)
  --checkpoint FILE      Checkpoint path (default: <agent>_policy.pt)
  --greedy               Deterministic policy in eval   (PPO only)
  -h, --help             Show this help

PPO-specific options:
  --update_every N       Buffer size before PPO update  (default: 512)
  --save_every N         Save checkpoint every N eps    (default: 1)

DQN-specific options:
  --n_actions N          Discrete action bins           (default: 21)
  --lr LR                Learning rate                  (default: 1e-3)
  --gamma G              Discount factor                (default: 0.99)
  --buf_size N           Replay buffer capacity         (default: 50000)
  --batch_size N         Minibatch size                 (default: 64)
  --target_update N      Hard target-net update period  (default: 500)
  --eps_start E          Initial epsilon                (default: 1.0)
  --eps_end E            Final epsilon                  (default: 0.05)
  --eps_decay N          Epsilon decay steps            (default: 10000)

Examples:
  # PPO – evaluation (1 episode, stochastic)
  \$0 --agent ppo --mode eval --episodes 1

  # PPO – evaluation (greedy / deterministic)
  \$0 --agent ppo --mode eval --episodes 5 --greedy

  # PPO – training (100 episodes, auto-saves ppo_policy.pt)
  \$0 --agent ppo --mode train --episodes 100

  # DQN – training (200 episodes)
  \$0 --agent dqn --mode train --episodes 200

  # DQN – evaluation (load specific checkpoint)
  \$0 --agent dqn --mode eval --episodes 10 --checkpoint dqn_policy.pt
EOF
  exit 0
}

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    # General
    --agent)           AGENT="$2";            shift 2 ;;
    --episodes)        NUM_EPISODES="$2";     shift 2 ;;
    --mode)            MODE="$2";             shift 2 ;;
    --checkpoint)      CHECKPOINT="$2";       shift 2 ;;
    --greedy)          GREEDY=true;           shift   ;;
    # PPO
    --update_every)    PPO_UPDATE_EVERY="$2"; shift 2 ;;
    --save_every)      PPO_SAVE_EVERY="$2";   shift 2 ;;
    # DQN
    --n_actions)       DQN_N_ACTIONS="$2";    shift 2 ;;
    --lr)              DQN_LR="$2";           shift 2 ;;
    --gamma)           DQN_GAMMA="$2";        shift 2 ;;
    --buf_size)        DQN_BUF_SIZE="$2";     shift 2 ;;
    --batch_size)      DQN_BATCH_SIZE="$2";   shift 2 ;;
    --target_update)   DQN_TARGET_UPDATE="$2";shift 2 ;;
    --eps_start)       DQN_EPS_START="$2";    shift 2 ;;
    --eps_end)         DQN_EPS_END="$2";      shift 2 ;;
    --eps_decay)       DQN_EPS_DECAY="$2";    shift 2 ;;
    -h|--help)         usage ;;
    *)
      echo "Unknown option: \$1" >&2
      echo "Use -h or --help for usage" >&2
      exit 1
      ;;
  esac
done

# ── Validate agent ───────────────────────────────────────────────────────────
case "$AGENT" in
  ppo) SERVER="$SCRIPT_DIR/serverppo.py" ;;
  dqn) SERVER="$SCRIPT_DIR/serverdqn.py" ;;
  *)
    echo "Error: --agent must be 'ppo' or 'dqn', got '$AGENT'" >&2
    exit 1
    ;;
esac

# ── Default checkpoint name ──────────────────────────────────────────────────
if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$SCRIPT_DIR/${AGENT}_policy.pt"
fi

# ── Validate mode ────────────────────────────────────────────────────────────
case "$MODE" in
  train|eval) ;;
  *)
    echo "Error: --mode must be 'train' or 'eval', got '$MODE'" >&2
    exit 1
    ;;
esac

# ── Sanity checks ────────────────────────────────────────────────────────────
[[ -d "$GYM_DIR" ]] || { echo "Missing GYM_DIR: $GYM_DIR" >&2; exit 1; }
[[ -x "$EXE"     ]] || { echo "Missing/not-executable EXE: $EXE" >&2; exit 1; }
[[ -f "$SERVER"  ]] || { echo "Missing SERVER script: $SERVER" >&2; exit 1; }
[[ -x "$PY"      ]] || {
  echo "Missing Python interpreter: $PY" >&2
  echo "Tip: python3 -m venv $SCRIPT_DIR/.venv && $SCRIPT_DIR/.venv/bin/pip install -r requirements.txt" >&2
  exit 1
}

if [[ "$MODE" == "eval" && ! -f "$CHECKPOINT" ]]; then
  echo "Error: checkpoint not found for eval: $CHECKPOINT" >&2
  exit 1
fi

if [[ "$GREEDY" == true && "$AGENT" == "dqn" ]]; then
  echo "Note: --greedy has no effect for DQN (eval is always greedy)" >&2
fi

# ── Build server command ─────────────────────────────────────────────────────
SERVER_CMD=("$PY" "$SERVER" --mode "$MODE" --checkpoint "$CHECKPOINT")

case "$AGENT" in
  ppo)
    SERVER_CMD+=(--update_every "$PPO_UPDATE_EVERY")
    SERVER_CMD+=(--save_every_episodes "$PPO_SAVE_EVERY")
    [[ "$GREEDY" == true && "$MODE" == "eval" ]] && SERVER_CMD+=(--greedy_eval)
    ;;
  dqn)
    SERVER_CMD+=(
      --n_actions    "$DQN_N_ACTIONS"
      --lr           "$DQN_LR"
      --gamma        "$DQN_GAMMA"
      --buf_size     "$DQN_BUF_SIZE"
      --batch_size   "$DQN_BATCH_SIZE"
      --target_update "$DQN_TARGET_UPDATE"
      --eps_start    "$DQN_EPS_START"
      --eps_end      "$DQN_EPS_END"
      --eps_decay    "$DQN_EPS_DECAY"
    )
    ;;
esac

# ── Log file naming ──────────────────────────────────────────────────────────
CKPT_BASE="$(basename "${CHECKPOINT%.pt}")"
if [[ "$MODE" == "eval" ]]; then
  LOG="$SCRIPT_DIR/eval_${AGENT}_${CKPT_BASE}.csv"
else
  LOG="$SCRIPT_DIR/rollout_${AGENT}.csv"
fi

SERVER_CMD+=(--log "$LOG")

# ── Summary ──────────────────────────────────────────────────────────────────
echo "┌─────────────────────────────────────────────"
echo "│  Agent      : $AGENT"
echo "│  Mode       : $MODE"
echo "│  Episodes   : $NUM_EPISODES"
echo "│  Checkpoint : $CHECKPOINT"
echo "│  Log        : $LOG"
[[ "$GREEDY" == true && "$AGENT" == "ppo" ]] && echo "│  Policy     : deterministic (greedy)"
echo "└─────────────────────────────────────────────"
echo ""

# ── Cleanup ──────────────────────────────────────────────────────────────────
SPID=""
cleanup() {
  if [[ -n "${SPID:-}" ]]; then
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    SPID=""
  fi
}
trap cleanup EXIT INT TERM

# ── Run ──────────────────────────────────────────────────────────────────────
cd "$GYM_DIR"

"${SERVER_CMD[@]}" &
SPID=$!
sleep 1   # give the server time to bind the socket

for ep in $(seq 0 $((NUM_EPISODES - 1))); do
  echo "==> Episode $((ep + 1)) / $NUM_EPISODES"
  "$EXE" -u Cmdenv -n "$NEDPATH" omnetpp.ini -c CBR-DL-PPO -r 0
done

cleanup

echo ""
echo "✅  ${AGENT^^} $MODE complete — $NUM_EPISODES episode(s)"
echo "📊  Log : $LOG"