import argparse, time, csv, random
from collections import deque
import zmq
import numpy as np
import veinsgym_pb2 as pb

import torch
from torch import nn
from torch.optim import Adam


# ──────────────────────────────────────────────
# Q-Network
# ──────────────────────────────────────────────
class QNet(nn.Module):
    """Maps observations to Q-values for every discrete action."""
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),    nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

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

    def __len__(self):
        return len(self.buf)


# ──────────────────────────────────────────────
# DQN Agent
# ──────────────────────────────────────────────
class DQNAgent:
    def __init__(
            self,
            obs_dim: int   = 8,
            n_actions: int = 21,          # discretise [0,2] into 21 bins → step 0.1
            lr: float      = 1e-3,
            gamma: float   = 0.99,
            buf_size: int  = 50_000,
            batch_size: int= 64,
            target_update: int = 500,     # hard-update target net every N gradient steps
            eps_start: float = 1.0,
            eps_end: float   = 0.05,
            eps_decay: int   = 10_000,    # linear decay over this many steps
            device: str    = "cpu",
    ):
        self.obs_dim     = obs_dim
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_update = target_update
        self.device      = device

        # Map discrete action index → continuous m value in [0, 2]
        self.action_values = np.linspace(0.0, 2.0, n_actions, dtype=np.float32)

        self.online = QNet(obs_dim, n_actions).to(device)
        self.target = QNet(obs_dim, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt    = Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buf_size)

        # Epsilon schedule
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay

        self.grad_steps = 0   # counts optimiser steps (drives target update & ε)
        self.env_steps  = 0   # counts environment steps (drives ε)

    # ------------------------------------------------------------------
    def epsilon(self) -> float:
        """Linearly-decayed ε."""
        frac = min(self.env_steps / self.eps_decay, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[int, float]:
        """
        Returns (action_index, m_value).
        Uses ε-greedy during training, greedy during eval.
        """
        self.env_steps += 1
        eps = 0.0 if greedy else self.epsilon()

        if random.random() < eps:
            idx = random.randrange(self.n_actions)
        else:
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            idx = int(self.online(x).argmax(dim=1).item())

        m = float(self.action_values[idx])
        return idx, m

    # ------------------------------------------------------------------
    def store(self, obs, action_idx, reward, next_obs, done):
        self.buffer.push(obs, action_idx, reward, next_obs, done)

    # ------------------------------------------------------------------
    def update(self):
        """One gradient step; returns loss scalar (or None if buffer too small)."""
        if len(self.buffer) < self.batch_size:
            return None

        obs, act, rew, nobs, done = self.buffer.sample(self.batch_size)

        obs  = torch.tensor(obs,  device=self.device)
        act  = torch.tensor(act,  device=self.device)
        rew  = torch.tensor(rew,  device=self.device)
        nobs = torch.tensor(nobs, device=self.device)
        done = torch.tensor(done, device=self.device)

        # Current Q-values for chosen actions
        q_pred = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)

        # Double-DQN target
        with torch.no_grad():
            best_actions = self.online(nobs).argmax(dim=1, keepdim=True)
            q_next       = self.target(nobs).gather(1, best_actions).squeeze(1)
            q_target     = rew + self.gamma * q_next * (1.0 - done)

        loss = nn.functional.smooth_l1_loss(q_pred, q_target)

        self.opt.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.opt.step()

        self.grad_steps += 1

        # Hard update target network
        if self.grad_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",               default="127.0.0.1")
    ap.add_argument("--port",               type=int,   default=5555)
    ap.add_argument("--mode",               choices=["train", "eval"], default="train")
    ap.add_argument("--seed",               type=int,   default=0)
    ap.add_argument("--checkpoint",         default="dqn_policy.pt")
    ap.add_argument("--log",                default="rollout_dqn.csv")
    ap.add_argument("--n_actions",          type=int,   default=21)
    ap.add_argument("--update_every",       type=int,   default=4)    # gradient step every N env steps
    ap.add_argument("--save_every_episodes",type=int,   default=10)
    ap.add_argument("--lr",                 type=float, default=1e-3)
    ap.add_argument("--gamma",              type=float, default=0.99)
    ap.add_argument("--buf_size",           type=int,   default=50_000)
    ap.add_argument("--batch_size",         type=int,   default=64)
    ap.add_argument("--target_update",      type=int,   default=500)
    ap.add_argument("--eps_start",          type=float, default=1.0)
    ap.add_argument("--eps_end",            type=float, default=0.05)
    ap.add_argument("--eps_decay",          type=int,   default=10_000)
    args = ap.parse_args()

    set_seeds(args.seed)

    ctx  = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[DQN] Listening on {args.host}:{args.port} | mode={args.mode} | seed={args.seed}")

    agent = DQNAgent(
        obs_dim      = 8,
        n_actions    = args.n_actions,
        lr           = args.lr,
        gamma        = args.gamma,
        buf_size     = args.buf_size,
        batch_size   = args.batch_size,
        target_update= args.target_update,
        eps_start    = args.eps_start,
        eps_end      = args.eps_end,
        eps_decay    = args.eps_decay,
    )

    if args.mode == "eval":
        agent.online.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        agent.online.eval()
        print(f"[DQN] Loaded checkpoint {args.checkpoint}")

    logf = open(args.log, "a", newline="")
    w    = csv.writer(logf)
    if logf.tell() == 0:
        w.writerow([
            "ts","episode","step",
            "thr","delay","jitter","loss","numUe","stepEnergyJ","txPowerDbm","sinr",
            "rew","act_m","act_idx","epsilon","q_loss"
        ])

    last_obs    = None
    last_idx    = None
    q_loss      = None

    episode     = 0
    steps       = 0
    ep_return   = 0.0

    while True:
        msg = sock.recv()
        req = pb.Request()
        req.ParseFromString(msg)

        reply    = pb.Reply()
        reply.id = req.id

        # ── INIT ──────────────────────────────────────────────────────
        if req.HasField("init"):
            print("[DQN] Init received")
            last_obs  = None
            last_idx  = None
            steps     = 0
            ep_return = 0.0
            sock.send(reply.SerializeToString())
            continue

        # ── SHUTDOWN ───────────────────────────────────────────────────
        if req.HasField("shutdown"):
            # Mark last transition as terminal
            if args.mode == "train" and last_obs is not None:
                # reward for terminal step is 0 (already stored on next step arrival)
                # just ensure the done flag is correct – we re-push with done=1
                pass  # final done=1 is handled below in the step handler

            if args.mode == "train" and (episode % args.save_every_episodes) == 0:
                torch.save(agent.online.state_dict(), args.checkpoint)

            print(f"[DQN] Shutdown ep={episode} return={ep_return:.3f} "
                  f"steps={steps} env_steps={agent.env_steps} "
                  f"buf={len(agent.buffer)} ε={agent.epsilon():.3f}")
            episode += 1
            sock.send(reply.SerializeToString())
            continue

        # ── STEP ───────────────────────────────────────────────────────
        if req.HasField("step"):
            obs = np.array(req.step.observation.box.values, dtype=np.float32)
            rew = 0.0
            if (req.step.reward and
                    req.step.reward.HasField("box") and
                    len(req.step.reward.box.values) > 0):
                rew = float(req.step.reward.box.values[0])

            # Store previous transition now that we have the next obs + reward
            if last_obs is not None:
                if args.mode == "train":
                    agent.store(last_obs, last_idx, rew, obs, done=0.0)

                    # Learn every `update_every` env steps
                    if agent.env_steps % args.update_every == 0:
                        q_loss = agent.update()

                ep_return += rew
                steps     += 1

            # Choose next action
            greedy       = (args.mode == "eval")
            idx, m       = agent.act(obs, greedy=greedy)

            # Log
            thr, delay, jitter, pkt_loss, numUe, stepE, txPowerDbm, sinr = obs.tolist()
            w.writerow([
                time.time(), episode, steps,
                thr, delay, jitter, pkt_loss, numUe, stepE, txPowerDbm, sinr,
                rew, m, idx,
                f"{agent.epsilon():.4f}",
                f"{q_loss:.6f}" if q_loss is not None else ""
            ])
            logf.flush()

            last_obs = obs
            last_idx = idx

            if steps % 20 == 0:
                loss_str = f"{q_loss:.4f}" if q_loss is not None else "N/A"
                print(f"[DQN] ep={episode} step={steps} rew={rew:.4f} "
                      f"act(m)={m:.3f} idx={idx} ε={agent.epsilon():.3f} "
                      f"loss={loss_str} obs[:3]={obs[:3]}")



            reply.action.box.values[:] = [m]
            sock.send(reply.SerializeToString())
            continue

        sock.send(reply.SerializeToString())


if __name__ == "__main__":
    main()