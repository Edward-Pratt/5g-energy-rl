#!/usr/bin/env python3
"""
serverppo.py – PPO agent server for Simu5G-Gym.

Listens on a ZMQ REP socket for OMNeT++ step/init/shutdown messages,
selects gNB TX-power multiplier actions, and trains a PPO policy.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import zmq

import veinsgym_pb2 as pb


# ── Neural network ─────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """
    Actor-critic network with a continuous Gaussian policy.

    Output:
        mu  – mean of the action distribution (pre-tanh), shape (B, 1)
        v   – state-value estimate,                       shape (B, 1)

    The learnable log_std is action-dimension-sized so it can be extended
    to multi-action scenarios without architecture changes.
    """

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.mu_head  = nn.Linear(hidden, 1)
        self.v_head   = nn.Linear(hidden, 1)
        # log_std is a learnable parameter, not tied to eval/train mode.
        # Deterministic behaviour is controlled by the caller (greedy=True
        # in act()), not by silently clamping std — that was FIX #7.
        self.log_std  = nn.Parameter(torch.tensor([-0.5]))

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.mu_head(h), self.v_head(h)

    def get_std(self) -> torch.Tensor:
        # Clamp log_std to a reasonable range for numerical stability.
        return torch.exp(torch.clamp(self.log_std, min=-5.0, max=1.0))


# ── PPO agent ──────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(
            self,
            obs_dim:    int   = 8,
            lr:         float = 3e-4,
            gamma:      float = 0.99,
            lam:        float = 0.95,
            clip:       float = 0.2,
            ent_coef:   float = 0.01,
            vf_coef:    float = 0.5,
            epochs:     int   = 10,
            batch_size: int   = 64,
            max_grad_norm: float = 0.5,
            device:     str   = "cpu",
    ):
        self.gamma      = gamma
        self.lam        = lam
        self.clip       = clip
        self.ent_coef   = ent_coef
        self.vf_coef    = vf_coef
        self.epochs     = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device     = device

        self.model = PolicyNet(obs_dim).to(device)
        self.opt   = Adam(self.model.parameters(), lr=lr)

        self.reset_buffer()

    # ── Buffer ────────────────────────────────────────────────────────────────

    def reset_buffer(self):
        self.obs_buf:  list = []
        self.u_buf:    list = []   # tanh-squashed actions stored clamped
        self.logp_buf: list = []
        self.rew_buf:  list = []
        self.val_buf:  list = []
        self.done_buf: list = []

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False):
        """
        Returns (m, logp, v, u):
            m    – multiplier sent to simulator, range [0, 2]
            logp – log probability of the action (for PPO ratio)
            v    – value estimate
            u    – tanh-squashed action in [-1, 1], stored in buffer
        """
        x   = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, v = self.model(x)
        std = self.model.get_std().expand_as(mu)
        dist = Normal(mu, std)

        if greedy:
            z = mu                 # deterministic: use the mean
        else:
            z = dist.rsample()     # stochastic: reparameterised sample

        # FIX #1: clamp z BEFORE tanh so u is strictly inside (-1, 1) and
        # can be safely inverted with atanh later.
        z = torch.clamp(z, -5.0, 5.0)
        u = torch.tanh(z)          # u ∈ (-1, 1) — stored in buffer

        m = u + 1.0                # m ∈ (0, 2) — sent to simulator

        # Log-prob with tanh squashing correction (SAC appendix).
        logp = (dist.log_prob(z) - torch.log(1.0 - u.pow(2) + 1e-6)).sum(dim=-1)

        return (
            float(m.item()),
            float(logp.item()),
            float(v.item()),
            float(u.item()),   # stored as the "action" in the buffer
        )

    def store(self, obs, u, logp, rew, val, done):
        self.obs_buf.append(obs)
        self.u_buf.append(u)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.done_buf.append(done)

    # ── GAE return computation ────────────────────────────────────────────────

    def _compute_gae(self, last_val: float = 0.0):
        rews  = np.array(self.rew_buf,  dtype=np.float32)
        vals  = np.array(self.val_buf + [last_val], dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        adv = np.zeros_like(rews)
        gae = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - dones[t]
            delta = rews[t] + self.gamma * vals[t + 1] * nonterminal - vals[t]
            gae   = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae

        ret = adv + vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self, last_val: float = 0.0, min_steps: int = 32):
        """
        Run PPO epochs on the current buffer then clear it.
        Returns True if an update was performed, False if buffer was too small.
        """
        if len(self.obs_buf) < min_steps:
            print(f"[PPO] Buffer too small ({len(self.obs_buf)} < {min_steps}), skipping update.")
            self.reset_buffer()
            return False

        obs      = torch.tensor(np.array(self.obs_buf),  dtype=torch.float32, device=self.device)
        # FIX #1 (continued): u values were clamped at store time, so atanh
        # is safe here.  Extra clamp is a belt-and-braces guard.
        u_stored = torch.tensor(np.array(self.u_buf),    dtype=torch.float32, device=self.device).unsqueeze(1)
        u_stored = torch.clamp(u_stored, -0.9999, 0.9999)
        old_logp = torch.tensor(np.array(self.logp_buf), dtype=torch.float32, device=self.device)

        adv, ret = self._compute_gae(last_val=last_val)
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.tensor(ret, dtype=torch.float32, device=self.device)

        n    = obs.shape[0]
        idxs = np.arange(n)

        for epoch in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                mb = idxs[start:start + self.batch_size]

                mu, v = self.model(obs[mb])
                std   = self.model.get_std().expand_as(mu)
                dist  = Normal(mu, std)

                # Recover pre-tanh sample via atanh (inverse of tanh).
                z_mb = torch.atanh(u_stored[mb])

                # Log-prob with tanh correction.
                logp = (dist.log_prob(z_mb)
                        - torch.log(1.0 - u_stored[mb].pow(2) + 1e-6)).sum(dim=-1)

                ratio = torch.exp(logp - old_logp[mb])

                # FIX #2: standard PPO clipped objective.
                # Both terms of torch.min should be the full surrogate;
                # the original accidentally multiplied adv[mb] in twice.
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss  = self.vf_coef * F.mse_loss(v.squeeze(1), ret[mb])
                entropy = dist.entropy().sum(dim=-1).mean()
                loss    = pi_loss + v_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                # Gradient clipping prevents catastrophic updates on noisy
                # telecom reward signals.
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

        self.reset_buffer()
        return True


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(agent: PPOAgent, path: str, device: str = "cpu"):
    """Load model weights if the checkpoint file exists."""
    if os.path.isfile(path):
        # FIX #6: weights_only=True avoids arbitrary code execution risk
        # introduced in PyTorch >= 2.0.
        state = torch.load(path, map_location=device, weights_only=True)
        agent.model.load_state_dict(state)
        print(f"[PPO] Loaded checkpoint: {path}")
        return True
    print(f"[PPO] No checkpoint found at {path} — starting from scratch.")
    return False


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="PPO agent server for Simu5G-Gym")
    ap.add_argument("--host",               default="127.0.0.1")
    ap.add_argument("--port",               type=int,   default=5555)
    ap.add_argument("--mode",               choices=["train", "eval"], default="train")
    ap.add_argument("--seed",               type=int,   default=0)
    ap.add_argument("--checkpoint",         default="ppo_policy.pt")
    ap.add_argument("--log",                default="rollout_ppo.csv")
    ap.add_argument("--update_every",       type=int,   default=512,
                    help="Steps between PPO updates (train mode only)")
    ap.add_argument("--save_every_episodes",type=int,   default=1,
                    help="Save checkpoint every N completed episodes")
    ap.add_argument("--greedy_eval",        action="store_true",
                    help="Use deterministic (mean) policy during eval")
    # FIX #8: expose batch_size so it matches the DQN interface.
    ap.add_argument("--batch_size",         type=int,   default=64)
    ap.add_argument("--epochs",             type=int,   default=10,
                    help="PPO gradient epochs per update")
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
    print(f"[PPO] Listening on {args.host}:{args.port} | mode={args.mode} | seed={args.seed}")

    # Build agent.
    agent = PPOAgent(
        obs_dim    = 8,
        batch_size = args.batch_size,
        epochs     = args.epochs,
    )

    # FIX #5: set eval_mode and load weights before any forward pass can happen.
    if args.mode == "eval":
        load_checkpoint(agent, args.checkpoint)
        agent.model.eval()
    else:
        # Try to resume training from an existing checkpoint.
        load_checkpoint(agent, args.checkpoint)
        agent.model.train()

    # CSV log — append so resuming training doesn't lose prior data.
    log_exists = os.path.isfile(args.log) and os.path.getsize(args.log) > 0
    logf = open(args.log, "a", newline="")
    w    = csv.writer(logf)
    if not log_exists:
        w.writerow([
            "ts", "episode", "step",
            "thr", "delay", "jitter", "loss",
            "numUe", "stepEnergyJ", "txPowerDbm", "sinr",
            "rew", "ep_return", "act_m",
        ])

    # Per-episode state.
    episode   = 0
    steps     = 0
    ep_return = 0.0

    # Last-step bookkeeping for building (s, a, r, s') transitions.
    last_obs  = None
    last_u    = None
    last_logp = None
    last_val  = None

    while True:
        raw = sock.recv()
        req = pb.Request()
        req.ParseFromString(raw)

        reply    = pb.Reply()
        reply.id = req.id

        # ── Init ──────────────────────────────────────────────────────────────
        if req.HasField("init"):
            print(f"[PPO] Init received (episode {episode})")
            last_obs = last_u = last_logp = last_val = None
            steps     = 0
            ep_return = 0.0
            sock.send(reply.SerializeToString())
            continue

        # ── Shutdown (end of episode) ──────────────────────────────────────────
        if req.HasField("shutdown"):
            if args.mode == "train":
                # Mark the last stored transition as terminal.
                if agent.done_buf:
                    agent.done_buf[-1] = 1.0

                # FIX #3: use last_val as the bootstrap value.
                # If the episode ended normally (sim-time-limit), last_val
                # holds the value estimate of the final observation, which
                # gives a better GAE estimate than hard-coding 0.
                bootstrap = last_val if last_val is not None else 0.0
                agent.update(last_val=bootstrap)

                # FIX #9: save on episode >= 1 (after real training data),
                # not episode == 0 (before any data has been collected).
                if episode >= 1 and (episode % args.save_every_episodes) == 0:
                    torch.save(agent.model.state_dict(), args.checkpoint)
                    print(f"[PPO] Checkpoint saved → {args.checkpoint}")

            print(f"[PPO] Episode {episode} done | return={ep_return:.3f} | steps={steps}")
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

            # FIX #4: accumulate ep_return for ALL steps, including the first.
            # Store the transition that led TO this observation.
            if last_obs is not None:
                if args.mode == "train":
                    agent.store(last_obs, last_u, last_logp, rew, last_val, done=0.0)
                ep_return += rew
                steps     += 1

            # Select action for this observation.
            greedy = args.mode == "eval" and args.greedy_eval
            m, logp, v, u = agent.act(obs, greedy=greedy)

            # Mid-episode update when buffer is full.
            if args.mode == "train" and len(agent.obs_buf) >= args.update_every:
                # FIX #10: pass current value estimate as bootstrap so the
                # partial trajectory at the end of the buffer is handled
                # correctly rather than silently dropping it.
                agent.update(last_val=v)
                print(f"[PPO] Mid-episode update | ep={episode} step={steps} return≈{ep_return:.3f}")

            # Log.
            thr, delay, jitter, loss, numUe, stepE, txPowerDbm, sinr = obs.tolist()
            w.writerow([
                time.time(), episode, steps,
                thr, delay, jitter, loss, numUe, stepE, txPowerDbm, sinr,
                rew, ep_return, m,
            ])
            logf.flush()

            if steps % 20 == 0:
                print(
                    f"[PPO] ep={episode} step={steps} "
                    f"rew={rew:.4f} m={m:.3f} "
                    f"thr={thr:.1f} delay={delay*1e3:.1f}ms sinr={sinr:.1f}dB"
                )

            # Carry forward for next transition.
            last_obs  = obs
            last_u    = u
            last_logp = logp
            last_val  = v

            reply.action.box.values[:] = [m]
            sock.send(reply.SerializeToString())
            continue

        # Unknown message type — send empty reply to avoid blocking the sim.
        sock.send(reply.SerializeToString())


if __name__ == "__main__":
    main()