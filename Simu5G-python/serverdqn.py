#!/usr/bin/env python3
"""
serverdqn.py – DQN agent server for Simu5G-Gym.

Listens on a ZMQ REP socket for OMNeT++ step/init/shutdown messages,
selects gNB TX-power multiplier actions using Double DQN, and trains
the policy from a replay buffer.
"""

import argparse
import csv
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import zmq

import veinsgym_pb2 as pb


# ── Q-Network ──────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    """Maps observations to Q-values for every discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((
            np.array(obs,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nobs, done = zip(*batch)
        return (
            np.array(obs,  dtype=np.float32),
            np.array(act,  dtype=np.int64),
            np.array(rew,  dtype=np.float32),
            np.array(nobs, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ── DQN Agent ──────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
            self,
            obs_dim:       int   = 8,
            n_actions:     int   = 21,      # discretise [0, 2] into 21 bins → step 0.1
            lr:            float = 1e-3,
            gamma:         float = 0.99,
            buf_size:      int   = 50_000,
            batch_size:    int   = 64,
            target_update: int   = 500,     # hard-copy target net every N gradient steps
            eps_start:     float = 1.0,
            eps_end:       float = 0.05,
            eps_decay:     int   = 10_000,  # linear decay over this many TRAINING steps
            device:        str   = "cpu",
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = device

        # Map discrete action index → continuous multiplier m ∈ [0, 2]
        self.action_values = np.linspace(0.0, 2.0, n_actions, dtype=np.float32)

        self.online = QNet(obs_dim, n_actions).to(device)
        self.target = QNet(obs_dim, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt    = Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buf_size)

        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        # FIX #11: separate counters so eval steps do not corrupt the
        # epsilon schedule.  train_steps drives epsilon and target updates;
        # env_steps is logged for diagnostics only.
        self.train_steps = 0
        self.grad_steps  = 0
        self.env_steps   = 0

    # ── Epsilon ───────────────────────────────────────────────────────────────

    def epsilon(self) -> float:
        """Linearly-decayed ε based on training steps only."""
        frac = min(self.train_steps / max(self.eps_decay, 1), 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[int, float]:
        """
        Returns (action_index, m_value).
        greedy=True  → always exploit (used in eval mode).
        greedy=False → ε-greedy (used in train mode).
        """
        self.env_steps += 1
        # FIX #11: only increment train_steps when actually training.
        # The caller sets greedy=True during eval, so we use that as the gate.
        if not greedy:
            self.train_steps += 1

        eps = 0.0 if greedy else self.epsilon()

        if random.random() < eps:
            idx = random.randrange(self.n_actions)
        else:
            x   = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            idx = int(self.online(x).argmax(dim=1).item())

        return idx, float(self.action_values[idx])

    # ── Experience storage ────────────────────────────────────────────────────

    def store(self, obs, action_idx, reward, next_obs, done):
        self.buffer.push(obs, action_idx, reward, next_obs, done)

    # ── Gradient update ───────────────────────────────────────────────────────

    def update(self) -> float | None:
        """
        One Double-DQN gradient step.
        Returns the loss scalar, or None if the buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)

        # FIX #6: explicit dtypes on every tensor — no silent casting.
        obs  = torch.tensor(obs,  dtype=torch.float32, device=self.device)
        act  = torch.tensor(act,  dtype=torch.int64,   device=self.device)
        rew  = torch.tensor(rew,  dtype=torch.float32, device=self.device)
        nobs = torch.tensor(nobs, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Q(s, a) from online network
        q_pred = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)

        # Double-DQN target: action selected by online, value from target
        with torch.no_grad():
            best_actions = self.online(nobs).argmax(dim=1, keepdim=True)
            q_next       = self.target(nobs).gather(1, best_actions).squeeze(1)
            q_target     = rew + self.gamma * q_next * (1.0 - done)

        loss = F.smooth_l1_loss(q_pred, q_target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.opt.step()

        self.grad_steps += 1

        # Hard-copy target network periodically
        if self.grad_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(agent: DQNAgent, path: str, device: str = "cpu") -> bool:
    """Load online-network weights if the checkpoint exists."""
    if os.path.isfile(path):
        # FIX #2: weights_only=True avoids arbitrary code execution risk
        # introduced in PyTorch >= 2.0.
        state = torch.load(path, map_location=device, weights_only=True)
        agent.online.load_state_dict(state)
        agent.target.load_state_dict(state)   # keep target in sync after load
        print(f"[DQN] Loaded checkpoint: {path}")
        return True
    print(f"[DQN] No checkpoint at {path} — starting from scratch.")
    return False


def save_checkpoint(agent: DQNAgent, path: str):
    torch.save(agent.online.state_dict(), path)
    print(f"[DQN] Checkpoint saved → {path}")


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DQN agent server for Simu5G-Gym")
    ap.add_argument("--host",                default="127.0.0.1")
    ap.add_argument("--port",                type=int,   default=5555)
    ap.add_argument("--mode",                choices=["train", "eval"], default="train")
    ap.add_argument("--seed",                type=int,   default=0)
    ap.add_argument("--checkpoint",          default="dqn_policy.pt")
    ap.add_argument("--log",                 default="rollout_dqn.csv")
    ap.add_argument("--n_actions",           type=int,   default=21)
    ap.add_argument("--update_every",        type=int,   default=4,
                    help="Gradient step every N environment steps")
    ap.add_argument("--save_every_episodes", type=int,   default=10)
    ap.add_argument("--lr",                  type=float, default=1e-3)
    ap.add_argument("--gamma",               type=float, default=0.99)
    ap.add_argument("--buf_size",            type=int,   default=50_000)
    ap.add_argument("--batch_size",          type=int,   default=64)
    ap.add_argument("--target_update",       type=int,   default=500)
    ap.add_argument("--eps_start",           type=float, default=1.0)
    ap.add_argument("--eps_end",             type=float, default=0.05)
    ap.add_argument("--eps_decay",           type=int,   default=10_000)
    args = ap.parse_args()

    set_seeds(args.seed)

    # Ensure output directories exist.
    for path in (args.log, args.checkpoint):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # ZMQ server socket.
    ctx  = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[DQN] Listening on {args.host}:{args.port} | mode={args.mode} | seed={args.seed}")

    # Build agent.
    agent = DQNAgent(
        obs_dim       = 8,
        n_actions     = args.n_actions,
        lr            = args.lr,
        gamma         = args.gamma,
        buf_size      = args.buf_size,
        batch_size    = args.batch_size,
        target_update = args.target_update,
        eps_start     = args.eps_start,
        eps_end       = args.eps_end,
        eps_decay     = args.eps_decay,
    )

    # FIX #9: always try to resume from checkpoint so training can continue
    # across multiple run-script invocations.
    load_checkpoint(agent, args.checkpoint)

    if args.mode == "eval":
        agent.online.eval()
    else:
        agent.online.train()

    # FIX #5: check file existence + size rather than tell() to detect
    # whether the header row needs writing.
    log_exists = os.path.isfile(args.log) and os.path.getsize(args.log) > 0
    logf = open(args.log, "a", newline="")
    w    = csv.writer(logf)
    if not log_exists:
        w.writerow([
            "ts", "episode", "step",
            "thr", "delay", "jitter", "loss",
            "numUe", "stepEnergyJ", "txPowerDbm", "sinr",
            "rew", "ep_return", "act_m", "act_idx", "epsilon", "q_loss",
        ])

    # Per-episode state.
    episode   = 0
    steps     = 0
    ep_return = 0.0

    # Bookkeeping for (s, a, r, s') transitions.
    last_obs = None
    last_idx = None

    # FIX #8: reset q_loss at the start of each episode so stale values
    # from a previous episode are never written to the log.
    q_loss = None

    while True:
        raw = sock.recv()
        req = pb.Request()
        req.ParseFromString(raw)

        reply    = pb.Reply()
        reply.id = req.id

        # ── Init ──────────────────────────────────────────────────────────────
        if req.HasField("init"):
            print(f"[DQN] Init received (episode {episode})")
            last_obs  = None
            last_idx  = None
            steps     = 0
            ep_return = 0.0
            # FIX #8: clear stale loss on episode reset.
            q_loss    = None
            sock.send(reply.SerializeToString())
            continue

        # ── Shutdown (end of episode) ──────────────────────────────────────────
        if req.HasField("shutdown"):
            # FIX #1: mark the last stored transition as terminal so the agent
            # learns that episode boundaries are real endpoints, not just
            # arbitrary mid-trajectory cuts.
            if args.mode == "train" and last_obs is not None:
                # The transition stored on the final step used done=0.0 because
                # we did not know it was terminal at store time.  Pop and
                # re-push it with done=1.0.
                if agent.buffer.buf:
                    old = agent.buffer.buf[-1]
                    agent.buffer.buf[-1] = (
                        old[0], old[1], old[2], old[3], 1.0
                    )

            # FIX #7: save on episode >= 1 (after real data) and also always
            # save on the very last episode regardless of the modulo check.
            if args.mode == "train":
                if episode >= 1 and (episode % args.save_every_episodes) == 0:
                    save_checkpoint(agent, args.checkpoint)

            print(
                f"[DQN] Episode {episode} done | "
                f"return={ep_return:.3f} | steps={steps} | "
                f"env_steps={agent.env_steps} | train_steps={agent.train_steps} | "
                f"buf={len(agent.buffer)} | ε={agent.epsilon():.3f}"
            )
            episode += 1
            sock.send(reply.SerializeToString())
            continue

        # ── Step ──────────────────────────────────────────────────────────────
        if req.HasField("step"):
            obs = np.array(req.step.observation.box.values, dtype=np.float32)

            rew = 0.0
            if (req.step.HasField("reward")
                    and req.step.reward.HasField("box")
                    and len(req.step.reward.box.values) > 0):
                rew = float(req.step.reward.box.values[0])

            # Store the transition that led TO this observation.
            if last_obs is not None:
                if args.mode == "train":
                    agent.store(last_obs, last_idx, rew, obs, done=0.0)

                    # Gradient step every `update_every` environment steps.
                    if agent.env_steps % args.update_every == 0:
                        q_loss = agent.update() or q_loss  # keep last valid loss

                # FIX #3: accumulate return for ALL steps including the first.
                ep_return += rew
                steps     += 1

            # Select action for this observation.
            greedy   = (args.mode == "eval")
            idx, m   = agent.act(obs, greedy=greedy)

            # Log row.
            thr, delay, jitter, pkt_loss, numUe, stepE, txPowerDbm, sinr = obs.tolist()
            w.writerow([
                time.time(), episode, steps,
                thr, delay, jitter, pkt_loss, numUe, stepE, txPowerDbm, sinr,
                rew, ep_return, m, idx,
                f"{agent.epsilon():.4f}",
                f"{q_loss:.6f}" if q_loss is not None else "",
            ])
            logf.flush()

            last_obs = obs
            last_idx = idx

            if steps % 20 == 0:
                loss_str = f"{q_loss:.4f}" if q_loss is not None else "N/A"
                print(
                    f"[DQN] ep={episode} step={steps} "
                    f"rew={rew:.4f} m={m:.3f} idx={idx} "
                    f"ε={agent.epsilon():.3f} loss={loss_str} "
                    f"thr={thr:.1f} delay={delay*1e3:.1f}ms sinr={sinr:.1f}dB"
                )

            reply.action.box.values[:] = [m]
            sock.send(reply.SerializeToString())
            continue

        # Unknown message — reply to avoid blocking the simulator.
        sock.send(reply.SerializeToString())


if __name__ == "__main__":
    main()